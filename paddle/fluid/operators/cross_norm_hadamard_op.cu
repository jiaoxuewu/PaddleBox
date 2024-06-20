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

#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/cross_norm_hadamard.cu.h"
#include "paddle/fluid/operators/cross_norm_hadamard_op.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#endif

#include "paddle/fluid/framework/fleet/box_wrapper.h"
namespace paddle {
namespace operators {
using framework::Tensor;
using platform::PADDLE_CUDA_NUM_THREADS;

template <typename DeviceContext, typename T>
class CrossNormHadamardCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<framework::LoDTensor>("Input");
    auto* summary_input = ctx.Input<framework::Tensor>("SummaryInput");
    auto* Out = ctx.Output<Tensor>("Out");
    auto* cuda_means = ctx.Output<Tensor>("CudaMeans");
    auto* cuda_scales = ctx.Output<Tensor>("CudaScales");

    auto fields_num = ctx.Attr<int64_t>("fields_num");
    auto embed_dim = ctx.Attr<int64_t>("embed_dim");

    auto cols = (embed_dim * 3 + 1) * fields_num;
    auto input_dims = input->dims();
    auto rows = input_dims[0];

    auto& dev_ctx = ctx.template device_context<GPUCtx>();
    auto stream = ctx.cuda_device_context().stream();

    Out->Resize({rows, cols});
    T* out_data = Out->mutable_data<T>(ctx.GetPlace());
    phi::funcs::set_constant(dev_ctx, Out, static_cast<T>(0));

    cuda_means->Resize({1, cols});
    cuda_scales->Resize({1, cols});
    cuda_means->mutable_data<T>(ctx.GetPlace());
    cuda_scales->mutable_data<T>(ctx.GetPlace());

    nncross_norm_ff<T>(fields_num, embed_dim, rows, input->data<T>(),
                       Out->data<T>(), summary_input->data<T>(),
                       cuda_means->data<T>(), cuda_scales->data<T>(), stream);
  }
};

template <typename DeviceContext, typename T>
class CrossNormHadamardOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* summary_input = ctx.Input<Tensor>("SummaryInput");
    auto* out = ctx.Input<Tensor>("Out");
    auto* means = ctx.Input<Tensor>("CudaMeans");
    auto* scales = ctx.Input<Tensor>("CudaScales");
    auto* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto fields_num = ctx.Attr<int64_t>("fields_num");
    auto embed_dim = ctx.Attr<int64_t>("embed_dim");
    const float epsilon = ctx.Attr<float>("epsilon");
    const float dr = ctx.Attr<float>("summary_decay_rate");
    const bool need_sync_stats = ctx.Attr<bool>("sync_stats");

    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto* summary_grad =
        ctx.Output<Tensor>(framework::GradVarName("SummaryInput"));

    auto& dev_ctx = ctx.template device_context<GPUCtx>();
    auto stream = ctx.cuda_device_context().stream();

    // initialize
    input_grad->mutable_data<T>(ctx.GetPlace());
    phi::funcs::set_constant(dev_ctx, input_grad, static_cast<T>(0));

    summary_grad->mutable_data<T>(ctx.GetPlace());
    phi::funcs::set_constant(dev_ctx, summary_grad, static_cast<T>(0));
    
    auto cols = (embed_dim * 3 + 1) * fields_num;
    auto input_dims = input->dims();
    auto rows = input_dims[0];

    // temperary tensor
    phi::funcs::Transpose<DeviceContext, T, 2> trans;

    Tensor input_help;
    input_help = ctx.AllocateTmpTensor<T, DeviceContext>(
        {fields_num * 2 * embed_dim, rows}, dev_ctx);
    trans(dev_ctx, *input, &input_help, {1, 0});

    Tensor summary_help;
    summary_help = ctx.AllocateTmpTensor<T, DeviceContext>({cols, 3}, dev_ctx);
    trans(dev_ctx, *summary_input, &summary_help, {1, 0});

    Tensor out_help;
    out_help = ctx.AllocateTmpTensor<T, DeviceContext>({cols, rows}, dev_ctx);
    trans(dev_ctx, *out, &out_help, {1, 0});

    Tensor out_grad_help;
    out_grad_help =
        ctx.AllocateTmpTensor<T, DeviceContext>({cols, rows}, dev_ctx);
    trans(dev_ctx, *out_grad, &out_grad_help, {1, 0});

    Tensor input_grad_help;
    input_grad_help = ctx.AllocateTmpTensor<T, DeviceContext>(
        {fields_num * 2 * embed_dim, rows}, dev_ctx);
    trans(dev_ctx, *input_grad, &input_grad_help, {1, 0});

    Tensor summary_grad_help;
    summary_grad_help =
        ctx.AllocateTmpTensor<T, DeviceContext>({cols, 3}, dev_ctx);
    trans(dev_ctx, *summary_grad, &summary_grad_help, {1, 0});

    std::vector<int> sum_offset(cols + 1, rows);
    for (int i = 2; i < sum_offset.size(); ++i) {
      sum_offset[i] += sum_offset[i - 1];
    }
    sum_offset[0] = 0;

    auto tmp_array = memory::Alloc(dev_ctx, sum_offset.size() * sizeof(int));
    memory::Copy(dev_ctx.GetPlace(),
                 tmp_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void*>(sum_offset.data()),
                 sum_offset.size() * sizeof(int), dev_ctx.stream());
    int* g_sum_offset = reinterpret_cast<int*>(tmp_array->ptr());

    auto temp_grad_buf1 = memory::Alloc(dev_ctx, sizeof(T) * cols);
    T* g_grad_buf1 = reinterpret_cast<T*>(temp_grad_buf1->ptr());
    auto temp_grad_buf2 = memory::Alloc(dev_ctx, sizeof(T) * cols);
    T* g_grad_buf2 = reinterpret_cast<T*>(temp_grad_buf2->ptr());

    nncross_norm_bp<T>(fields_num, embed_dim, rows, input_help.data<T>(),
                       out_help.data<T>(), out_grad_help.data<T>(),
                       input_grad_help.data<T>(), summary_grad_help.data<T>(),
                       summary_help.data<T>(), means->data<T>(),
                       scales->data<T>(), epsilon, stream, g_grad_buf1,
                       g_sum_offset, g_grad_buf2, ctx);

    trans(dev_ctx, input_grad_help, input_grad, {1, 0});
    trans(dev_ctx, summary_grad_help, summary_grad, {1, 0});

    int C = 3 * cols;
    T* summary_input_data =
        ctx.Output<Tensor>("SummaryInput")->mutable_data<T>(ctx.GetPlace());

    if (need_sync_stats) {
#if defined(PADDLE_WITH_NCCL)
      auto comm = platform::NCCLCommContext::Instance().Get(0, ctx.GetPlace());
      PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
          reinterpret_cast<const void*>(summary_grad->data<T>()),
          reinterpret_cast<void*>(summary_grad->data<T>()), C,
          platform::ToNCCLDataType(input->type()), ncclSum, comm->comm(),
          stream));
      PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU, and need_sync_stats connot be "
          "supported on windows now."));
#endif
    }
    update_norm_param<T>(stream, C, summary_grad->data<T>(), summary_input_data,
                         dr);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(cross_norm_hadamard,
                        ops::CrossNormHadamardCUDAKernel<GPUCtx, float>,
                        ops::CrossNormHadamardCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(cross_norm_hadamard_grad,
                        ops::CrossNormHadamardOpCUDAKernel<GPUCtx, float>,
                        ops::CrossNormHadamardOpCUDAKernel<GPUCtx, double>);
