/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_seqpool_cvm_with_conv_op.h"
#include "paddle/fluid/operators/fused/fused_seqpool_cvm_utils_xpu.h"
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

DECLARE_bool(check_fused_negative_nan_inf);
namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FusedSeqpoolCVMWithConvOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef TRACE_PROFILE
    TRACE_SCOPE_START("FusedSeqpoolCVMWithConvOpXPUKernel Compute", xpu_wait(ctx.template device_context<DeviceContext>().x_context()->xpu_stream));
#endif
    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto out = ctx.MultiOutput<LoDTensor>("Out");
    auto padding_value = ctx.Attr<float>("pad_value");
    bool use_cvm = ctx.Attr<bool>("use_cvm");
    bool need_filter = ctx.Attr<bool>("need_filter");
    float show_coeff = ctx.Attr<float>("show_coeff");
    float clk_coeff = ctx.Attr<float>("clk_coeff");
    float threshold = ctx.Attr<float>("threshold");
    auto cvm_offset = ctx.Attr<int>("cvm_offset");
    auto show_filter = ctx.Attr<bool>("show_filter");
    const int embedx_concate_size = ctx.Attr<int>("embedx_concate_size");

    auto x0_lod = ins[0]->lod();
    auto x0_dims = ins[0]->dims();
    auto y_dims = out[0]->dims();
    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();
    size_t bs = x0_lod[0].size() - 1;
    int slot_num = static_cast<int>(ins.size());
    framework::LoD y_lod(1);
    y_lod[0].resize(bs + 1);
    for (size_t i = 0; i <= bs; ++i) {
        y_lod[0][i] = i;
    }
    for (int i = 0; i < slot_num; i++) {
      out[i]->Resize({static_cast<int64_t>(bs), y_dims[1]});
    }
    //TODO:r480 l3 have some thing wrong
    static bool use_l3_tensor = std::getenv("XPU_PADDLE_L3_TENSOR")!=NULL ?
                      (std::strcmp(std::getenv("XPU_PADDLE_L3_TENSOR"), "1") == 0 ? true:false) :
                      false;
    auto place = ctx.GetPlace();
    phi::Place l3_place = ctx.template device_context<DeviceContext>().GetL3Place();
    int w = ins[0]->numel() / x0_dims[0];
    if(use_cvm) {
      if (show_filter) w = (w - 1);
      PADDLE_ENFORCE_EQ(y_dims[1] % (w * embedx_concate_size), 0,
                        paddle::platform::errors::InvalidArgument(
                            "The output of dims[1] should be dividable of w"));
    }
    else{
      PADDLE_ENFORCE_EQ(y_dims[1] % ((w - cvm_offset) * embedx_concate_size), 0,
                  paddle::platform::errors::InvalidArgument(
                      "The output of dims[1] should be dividable of (w-2)"));
    }

    std::vector<const T*> cpu_x_addr_vec;
    cpu_x_addr_vec.reserve(slot_num);
    std::vector<T*> cpu_y_addr_vec;
    cpu_y_addr_vec.reserve(slot_num);

    unsigned int sum_lod_size = slot_num * (bs + 1);
    std::vector<int> cpu_lodx;
    cpu_lodx.reserve(sum_lod_size);
    unsigned int lod_index = 0;

    for (int i = 0; i < slot_num; i++) {
        cpu_x_addr_vec[i] = reinterpret_cast<const T*>(ins[i]->data<T>());
        if(use_l3_tensor) {
          cpu_y_addr_vec[i] = reinterpret_cast<T*>(out[i]->mutable_data<T>(l3_place));
        } else {
          cpu_y_addr_vec[i] = reinterpret_cast<T*>(out[i]->mutable_data<T>(place));
        }
        auto& x_lod = ins[i]->lod()[0];
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (size_t j = 0; j < x_lod.size(); j++) {
           cpu_lodx[lod_index + j] = x_lod[j];
        }

        lod_index += x_lod.size();
    }

#ifdef TRACE_PROFILE
    TRACE_SCOPE_START("xpu::sequence_sum_pool_cvm_with_conv", xpu_wait(xpu_context->xpu_stream););
#endif
    if (FLAGS_check_fused_negative_nan_inf) {
      xpu_wait(xpu_context->xpu_stream);
      check_tensors_nan(place, xpu_context, ins, "fused_with_conv-x");
    }
    int r = xpu::sequence_sum_pool_cvm_with_conv<T>(xpu_context,
                                          cpu_x_addr_vec,
                                          cpu_y_addr_vec,
                                          cpu_lodx,
                                          bs,
                                          x0_dims[1],
                                          slot_num,
                                          use_cvm,
                                          need_filter,
                                          show_coeff,
                                          clk_coeff,
                                          threshold,
                                          show_filter,
                                          padding_value,
                                          cvm_offset,
                                          embedx_concate_size);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                     platform::errors::External(
                         "The sequence_sum_pool_cvm_with_conv XPU OP return wrong value[%d %s]",
                         r, XPUAPIErrorMsg[r]));
    if (FLAGS_check_fused_negative_nan_inf) {
      xpu_wait(xpu_context->xpu_stream);
      check_tensors_nan(place, xpu_context, out, "fused_with_conv-y");
    }
#ifdef TRACE_PROFILE
    TRACE_SCOPE_END("xpu::sequence_sum_pool_cvm_with_conv", xpu_wait(xpu_context->xpu_stream););
    TRACE_SCOPE_END("FusedSeqpoolCVMWithConvOpXPUKernel Compute", xpu_wait(xpu_context->xpu_stream));
#endif
  }
};

template <typename DeviceContext, typename T>
class FusedSeqpoolCVMWithConvGradOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef TRACE_PROFILE
    TRACE_SCOPE_START("FusedSeqpoolCVMWithConvGradOpXPUKernel Compute", xpu_wait(ctx.template device_context<DeviceContext>().x_context()->xpu_stream));
#endif
    auto dOut = ctx.MultiInput<framework::LoDTensor>(framework::GradVarName("Out"));
    auto xs = ctx.MultiInput<LoDTensor>("X");
    const framework::Tensor* cvm = ctx.Input<framework::Tensor>("CVM");
    auto dxs = ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    bool show_filter = ctx.Attr<bool>("show_filter");
    auto cvm_offset = ctx.Attr<int>("cvm_offset");
    const int embedx_concate_size = ctx.Attr<int>("embedx_concate_size");

    int slot_num = dxs.size();
    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();
    auto place = ctx.GetPlace();
    T* cvm_data = const_cast<T*>(cvm->data<T>());
    int batch_size = dOut[0]->dims()[0];

    auto item_size = dxs[0]->dims()[1];
    std::vector<T*> cpu_dx_list(slot_num);
    std::vector<const T*> cpu_dy_list(slot_num);
    unsigned int sum_size = slot_num * (batch_size + 1);
    std::vector<int> cpu_lodx(sum_size);
    unsigned int start_index = 0;
    int total_length = 0;
    for (int i = 0; i < slot_num; ++i) {
      if(xs[i]->layout()!=paddle::framework::DataLayout::UNDEFINED) {
        total_length += dxs[i]->numel();
      }
    }
    framework::LoDTensor total_values;
    total_values.Resize(phi::make_ddim({total_length}));
    total_values.mutable_data<T>(place);
    int offset = 0;
    for (int k = 0; k < slot_num; k++) {
        auto dx = dxs[k];
        auto dy = dOut[k];

        if(xs[k]->layout()!=paddle::framework::DataLayout::UNDEFINED) {
          total_values.set_offset(offset);
          dx->ShareBufferWith(total_values);
          offset += dx->numel() * sizeof(T);
        }
        T* dx_data = dx->mutable_data<T>(place);
        T* dy_data = const_cast<T*>(dy->data<T>());
        cpu_dx_list[k] = dx_data;
        cpu_dy_list[k] = (const T*)dy_data;
        auto& lod_level_0 = dx->lod()[0];
        int lod_size = lod_level_0.size();
        for (int i = 0; i < lod_size; i++) {
          cpu_lodx[i + start_index] = lod_level_0[i];
        }
        start_index += lod_size;
    }

    if (FLAGS_check_fused_negative_nan_inf) {
      xpu_wait(xpu_context->xpu_stream);
      check_negative(place, xpu_context, cvm_data, cvm->numel());
      check_tensors_nan(place, xpu_context, dOut, "fused_with_conv-dy");
    }
    int r = xpu::sequence_sum_pool_cvm_with_conv_grad<T>(xpu_context,
                                               cpu_dy_list,
                                               cvm_data,
                                               cpu_dx_list,
                                               cpu_lodx,
                                               use_cvm,
                                               cvm_offset,
                                               show_filter,
                                               item_size,
                                               batch_size,
                                               slot_num,
                                               embedx_concate_size);
     PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
            platform::errors::External(
               "The sequence_sum_pool_cvm_with_conv_grad XPU OP return wrong value[%d %s]",
               r, XPUAPIErrorMsg[r]));
    if (FLAGS_check_fused_negative_nan_inf) {
      xpu_wait(xpu_context->xpu_stream);
      check_tensors_nan(place, xpu_context, dxs, "fused_with_conv-dx");
    }
#ifdef TRACE_PROFILE
    TRACE_SCOPE_END("FusedSeqpoolCVMWithConvGradOpXPUKernel Compute", xpu_wait(xpu_context->xpu_stream));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    fused_seqpool_cvm_with_conv,
    ops::FusedSeqpoolCVMWithConvOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    fused_seqpool_cvm_with_conv_grad,
    ops::FusedSeqpoolCVMWithConvGradOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
