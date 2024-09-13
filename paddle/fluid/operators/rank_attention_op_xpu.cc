/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/rank_attention_op.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "xpu/runtime.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class RankAttention2XPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* X = ctx.Input<Tensor>("X");
        auto* rank_offset = ctx.Input<Tensor>("RankOffset");
        auto* param = ctx.Input<Tensor>("RankParam");
        int max_rank = ctx.Attr<int>("MaxRank");
        auto* Out = ctx.Output<Tensor>("Out");

        // check dims
        auto x_dims = X->dims();
        auto ins_num = x_dims[0];
        auto x_fea_dim = x_dims[1];
        auto para_dims = param->dims();
        auto para_row = para_dims[0];
        auto para_col = para_dims[1];
        auto rank_offset_dims = rank_offset->dims();

        PADDLE_ENFORCE_EQ(rank_offset_dims[0], ins_num,
                          platform::errors::InvalidArgument("Input(RankOffset) has wrong rows."));
        PADDLE_ENFORCE_EQ(
            (rank_offset_dims[1] - 1) / 2, max_rank,
            platform::errors::InvalidArgument("Input(RankOffset) has wrong columns."));
        PADDLE_ENFORCE_EQ(max_rank * max_rank * x_fea_dim, para_row,
                          platform::errors::InvalidArgument("Input(RankParam) has wrong rows."));

        // get data ptr
        auto& dev_ctx = ctx.template device_context<DeviceContext>();

        T* out_data = Out->mutable_data<T>(ctx.GetPlace());

        int ret = baidu::xpu::xctr::rank_attention2<T>(dev_ctx.x_context(), ins_num, x_fea_dim, X->data<T>(),
                                          max_rank, rank_offset->data<int>(), para_row, para_col,
                                          param->data<T>(), out_data);
        PADDLE_ENFORCE_EQ(
            ret, XPU_SUCCESS,
            platform::errors::External("The rank_attention2 XPU kernel return wrong value[%d %s]",
                                       ret, XPUAPIErrorMsg[ret]));
    }
};

template <typename DeviceContext, typename T>
class RankAttention2GradXPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* X = ctx.Input<Tensor>("X");
        auto* rank_offset = ctx.Input<Tensor>("RankOffset");
        auto* param = ctx.Input<Tensor>("RankParam");
        auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
        auto* drank_para = ctx.Output<Tensor>(framework::GradVarName("RankParam"));

        // get dim
        auto x_dims = X->dims();
        auto ins_num = x_dims[0];
        auto x_fea_dim = x_dims[1];
        auto para_dims = param->dims();
        auto para_row = para_dims[0];
        auto para_col = para_dims[1];
        auto rank_offset_dims = rank_offset->dims();
        auto max_rank = (rank_offset_dims[1] - 1) / 2;

        auto& dev_ctx = ctx.template device_context<DeviceContext>();

        // initialize out grad
        T* drank_para_ptr = drank_para->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, drank_para, 0.0);

        int ret = baidu::xpu::xctr::rank_attention2_grad<T>(
            dev_ctx.x_context(), para_row, para_col, drank_para_ptr, ins_num, x_fea_dim,
            X->data<T>(), max_rank, rank_offset->data<int>(), dout->data<T>());
        PADDLE_ENFORCE_EQ(
            ret, XPU_SUCCESS,
            platform::errors::External("The rank_attention2_grad XPU kernel return wrong value[%d %s]",
                                       ret, XPUAPIErrorMsg[ret]));
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(rank_attention2,
                       ops::RankAttention2XPUKernel<paddle::platform::XPUDeviceContext, float>);

REGISTER_OP_XPU_KERNEL(rank_attention2_grad,
                       ops::RankAttention2GradXPUKernel<paddle::platform::XPUDeviceContext, float>);


