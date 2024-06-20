/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_concat_op.h"
#ifdef PADDLE_WITH_BOX_PS
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#else
#include "paddle/fluid/framework/threadpool.h"
#endif
#include <string>

#ifdef TRACE_PROFILE
// The producer side.
#include <scalopus_tracing/tracing.h>
#include <scalopus_transport/transport_loopback.h>
// The catapult recorder side.
#include <scalopus_catapult/catapult_recorder.h>
#include <scalopus_general/endpoint_manager_poll.h>
#include <scalopus_general/general_provider.h>
#include <scalopus_tracing/native_trace_provider.h>
#endif

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FusedConcatOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef TRACE_PROFILE
    TRACE_SCOPE_START("FusedConcatOpXPUKernel Compute", xpu_wait(ctx.template device_context<DeviceContext>().x_context()->xpu_stream));
#endif
    auto output = ctx.Output<framework::Tensor>("Out");
    auto inputs = ctx.MultiInput<LoDTensor>("X");
    const int length = ctx.Attr<int>("length");
    const int offset = ctx.Attr<int>("offset");

    const int x_num = static_cast<int>(inputs.size());
    const int total_cols = x_num * length;
    int batch_size = inputs[0]->dims()[0];
    int dim_size = inputs[0]->dims()[1];

    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();
    auto place = ctx.GetPlace();

    //TODO:r480 l3 have some thing wrong
    static bool use_l3_tensor = std::getenv("XPU_PADDLE_L3_TENSOR")!=NULL ?
                        (std::strcmp(std::getenv("XPU_PADDLE_L3_TENSOR"), "1") == 0 ? true:false) :
                        false;
    
    // input
    std::vector<const T*> cpu_x_addr_vec(x_num, 0);
    for (int i = 0; i < x_num; i++) {
        const auto *input = inputs[i];
        CHECK(batch_size == input->dims()[0])
            << "batch: " << batch_size << ", current: " << input->dims()[0];
        cpu_x_addr_vec[i] = reinterpret_cast<const T*>(input->data<T>());
    }

    // output
    phi::Place l3_place = ctx.template device_context<DeviceContext>().GetL3Place();

    output->Resize({batch_size, total_cols});
    T *cpu_y_addr = reinterpret_cast<T *>(output->mutable_data<T>(place));
    if(use_l3_tensor) {
        cpu_y_addr = reinterpret_cast<T*>(output->mutable_data<T>(l3_place));
    } else {
        cpu_y_addr = reinterpret_cast<T*>(output->mutable_data<T>(place));
    }

#ifdef TRACE_PROFILE
    TRACE_SCOPE_START("xpu::fused_concat", xpu_wait(xpu_context->xpu_stream););
#endif
    int r = xpu::fused_concat<T>(xpu_context,
                                    cpu_x_addr_vec,
                                    cpu_y_addr,
                                    batch_size,
                                    dim_size,
                                    length,
                                    offset);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                     platform::errors::External(
                         "The fused_concat XPU OP return wrong value[%d %s]",
                         r, XPUAPIErrorMsg[r]));
#ifdef TRACE_PROFILE
    TRACE_SCOPE_END("xpu::fused_concat", xpu_wait(xpu_context->xpu_stream););
    TRACE_SCOPE_END("FusedConcatOpXPUKernel Compute", xpu_wait(xpu_context->xpu_stream));
#endif
  }
};

template <typename DeviceContext, typename T>
class FusedConcatGradOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef TRACE_PROFILE
    TRACE_SCOPE_START("FusedConcatGradOpXPUKernel Compute", xpu_wait(ctx.template device_context<DeviceContext>().x_context()->xpu_stream));
#endif

    auto out_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));

    const int length = ctx.Attr<int>("length");
    const int offset = ctx.Attr<int>("offset");
    const int x_num = static_cast<int>(in_grads.size());

    int batch_size = out_grad->dims()[0];
    int dim_size = in_grads[0]->dims()[1];

    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();
    auto place = ctx.GetPlace();

    // input
    std::vector<T*> cpu_dx_list(x_num);
    for (int k = 0; k < x_num; ++k) {
      auto *in_grad = in_grads[k];
      CHECK(batch_size == in_grad->dims()[0])
          << "batch: " << batch_size << ", current: " << in_grad->dims()[0];
      cpu_dx_list[k] = reinterpret_cast<T *>(in_grad->mutable_data<T>(place));
    }
    
    // output
    auto cpu_dy_addr = out_grad->data<T>();
    int r = xpu::fused_concat_grad<T>(xpu_context,
                                        cpu_dy_addr,
                                        cpu_dx_list,
                                        batch_size,
                                        dim_size,
                                        length,
                                        offset);
     PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
            platform::errors::External(
               "The fused_concat_grad XPU OP return wrong value[%d %s]",
               r, XPUAPIErrorMsg[r]));
#ifdef TRACE_PROFILE
    TRACE_SCOPE_END("FusedConcatGradOpXPUKernel Compute", xpu_wait(xpu_context->xpu_stream));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    fused_concat,
    ops::FusedConcatOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    fused_concat_grad,
    ops::FusedConcatGradOpXPUKernel<paddle::platform::XPUDeviceContext, float>);

// TODO
// REGISTER_OP_XPU_KERNEL(
//     fused_seqpool_concat,
//     ops::FusedSeqpoolConcatOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
// REGISTER_OP_XPU_KERNEL(
//     fused_seqpool_concat_grad,
//     ops::FusedSeqpoolConcatGradOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
