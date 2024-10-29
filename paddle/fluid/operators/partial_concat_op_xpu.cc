//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/partial_concat_op.h"

namespace paddle {
namespace operators {
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class PartialConcatXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_vars = ctx.MultiInput<Tensor>("X");
    Tensor *out = ctx.Output<Tensor>("Out");
    PADDLE_ENFORCE_EQ(in_vars[0] != nullptr,
                      true,
                      platform::errors::InvalidArgument(
                          "The input of partial concat should not be null."));

    auto input_dim = in_vars[0]->dims();
    PADDLE_ENFORCE_EQ(input_dim.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "Only supports 2-D array with batch size in the 1st "
                          "dimension and data in the 2nd."));
    auto in_size = input_dim[1];
    // may be negative
    auto start_index = ctx.Attr<int>("start_index");
    start_index = ComputeStartIndex(start_index, in_size);

    auto partial_len = ctx.Attr<int>("length");
    if (partial_len < 0) {
      partial_len = in_size - start_index;
    }
    //TODO: what if partial_len > in_size
    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();

    int in_num = in_vars.size();
    int batch_size = input_dim[0];

    std::vector<framework::LoDTensor> tmp_tensors(in_num);
    std::vector<const T*> tmp_tensors_data(in_num);
    std::vector<std::vector<int>> tmp_outs_shape(in_num);
    for (size_t i = 0; i < in_vars.size(); i++) {
        tmp_tensors[i].Resize(phi::make_ddim({batch_size, partial_len}));
        tmp_tensors_data[i] = tmp_tensors[i].mutable_data<T>(ctx.GetPlace());

        tmp_outs_shape[i] = std::vector<int>({batch_size, partial_len});

        const T* input_data = in_vars[i]->data<T>();

        std::vector<int> xshape = phi::vectorize<int>(in_vars[i]->dims());
        std::vector<int> starts = {0, start_index};
        std::vector<int> ends = {batch_size, start_index + partial_len};//要截取的x的每个维度的终止坐标(不包含)

        int r = xpu::slice<T>(xpu_context,
                              input_data,
                              const_cast<T*>(tmp_tensors_data[i]),
                              xshape,
                              starts,
                              ends);
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::External(
                            "The partial_concat XPU OP's slice return wrong value[%d %s]",
                            r, XPUAPIErrorMsg[r]));
    }

    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    int axis = 1;
    int r = xpu::concat<T>(xpu_context,
                           tmp_tensors_data,
                           out_data,
                           tmp_outs_shape,
                           axis);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                    platform::errors::External(
                        "The partial_concat XPU OP's concat return wrong value[%d %s]",
                        r, XPUAPIErrorMsg[r]));
  }
};

template <typename DeviceContext, typename T>
class PartialConcatGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto xs_grad = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(ins[0] != nullptr,
                      true,
                      platform::errors::InvalidArgument(
                          "The input of partial concat should not be null."));
    // all parameters
    int batch_size = ins[0]->dims()[0];
    int in_size = ins[0]->dims()[1];
    // may be negative
    auto start_index = ctx.Attr<int>("start_index");
    start_index = ComputeStartIndex(start_index, in_size);
    auto partial_len = ctx.Attr<int>("length");
    if (partial_len < 0) {
        partial_len = in_size - start_index;
    }

    auto in_num = ins.size();

    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();

    std::vector<framework::LoDTensor> tmp_tensors(in_num);
    std::vector<const T*> tmp_tensors_data(in_num);

    const T* out_grad_data = out_grad->data<T>();
    for (size_t i = 0; i < in_num; i++) {
        tmp_tensors[i].Resize(phi::make_ddim({batch_size, partial_len}));
        tmp_tensors_data[i] = tmp_tensors[i].mutable_data<T>(ctx.GetPlace());

        std::vector<int> xshape = phi::vectorize<int>(out_grad->dims());
        std::vector<int> starts = {0, int(partial_len * i)};
        std::vector<int> ends = {batch_size, int(partial_len * i + partial_len)};//要截取的x的每个维度的终止坐标(不包含)

        int r = xpu::slice<T>(xpu_context,
                              out_grad_data,
                              const_cast<T*>(tmp_tensors_data[i]),
                              xshape,
                              starts,
                              ends);
        PADDLE_ENFORCE_EQ(
            r,
            xpu::Error_t::SUCCESS,
            platform::errors::External("The partial_concat_grad XPU OP's slice "
                                       "return wrong value[%d %s]",
                                       r,
                                       XPUAPIErrorMsg[r]));

        std::vector<int> tmp_shape = {batch_size, partial_len};
        std::vector<int> pad_left = {0, start_index};
        std::vector<int> pad_right = {0, in_size - start_index - partial_len};
        T* xs_grad_data = xs_grad[i]->mutable_data<T>(ctx.GetPlace());

        r = xpu::pad<T>(xpu_context,
                        tmp_tensors_data[i],
                        xs_grad_data,
                        tmp_shape,
                        pad_left,
                        pad_right,
                        T(0));
        PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                        platform::errors::External(
                            "The partial_concat_grad XPU OP's pad return wrong value[%d %s]",
                            r, XPUAPIErrorMsg[r]));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(partial_concat, ops::PartialConcatXPUKernel<paddle::platform::XPUDeviceContext, float>)
REGISTER_OP_XPU_KERNEL(partial_concat_grad, ops::PartialConcatGradXPUKernel<paddle::platform::XPUDeviceContext, float>)
