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

#include "paddle/fluid/operators/fused/fused_seqpool_cvm_kernel.h"
#include "paddle/fluid/operators/fused/fused_seqpool_cvm_op.h"
#include "paddle/fluid/operators/fused/fused_seqpool_cvm_utils_xpu.h"
#ifdef PADDLE_WITH_BOX_PS
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#else
#include "paddle/fluid/framework/threadpool.h"
#endif
#include <string>

// The producer side.
#include <scalopus_tracing/tracing.h>
#include <scalopus_transport/transport_loopback.h>
// The catapult recorder side.
#include <scalopus_catapult/catapult_recorder.h>
#include <scalopus_general/endpoint_manager_poll.h>
#include <scalopus_general/general_provider.h>
#include <scalopus_tracing/native_trace_provider.h>
DECLARE_bool(check_fused_negative_nan_inf);
namespace paddle {
namespace operators {

#define CHECK_USE_QUANT  \
    if (need_filter && quant_ratio > 0) { \
     auto &show = input_data[k * embedding_size]; \
     auto &click = input_data[k * embedding_size + 1]; \
     if ((show - click) * show_coeff + click * clk_coeff < threshold) { \
        continue; \
     } \
    }

#define QUANT_VALUE(val)  \
    (static_cast<int>(val * quant_ratio + 0.5) / static_cast<float>(quant_ratio));

#define CHECK_QUANT_AND_GETVAL(in_val)  \
    if (quant_ratio > 0) {  \
      if (need_filter) { \
        auto &show = input_data[k * embedding_size]; \
        auto &click = input_data[k * embedding_size + 1]; \
        if ((show - click) * show_coeff + click * clk_coeff < threshold) {  \
           continue;  \
        }  \
      } \
      val += QUANT_VALUE(in_val); \
    } else { \
      val += in_val; \
    }

template <typename DeviceContext, typename T>
class FusedSeqpoolCVMOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef TRACE_PROFILE
    TRACE_SCOPE_START("FusedSeqpoolCVMOpXPUKernel Compute", xpu_wait(ctx.template device_context<DeviceContext>().x_context()->xpu_stream));
#endif
    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto out = ctx.MultiOutput<LoDTensor>("Out");
    std::string pooltype = ctx.Attr<std::string>("pooltype");
    bool use_cvm = ctx.Attr<bool>("use_cvm");
    bool need_filter = ctx.Attr<bool>("need_filter");
    bool clk_filter = ctx.Attr<bool>("clk_filter");
    auto padding_value = ctx.Attr<float>("pad_value");
    auto quant_ratio = ctx.Attr<int>("quant_ratio");
    auto show_coeff = ctx.Attr<float>("show_coeff");
    auto clk_coeff = ctx.Attr<float>("clk_coeff");
    auto threshold = ctx.Attr<float>("threshold");
    auto cvm_offset = ctx.Attr<int>("cvm_offset");
    bool embed_threshold_filter = ctx.Attr<bool>("embed_threshold_filter");
    float embed_threshold = ctx.Attr<float>("embed_threshold");
    int embed_thres_size = ctx.Attr<int>("embed_thres_size");
    int embedx_concate_size = ctx.Attr<int>("embedx_concate_size");
    bool embedx_concate_filter = ctx.Attr<bool>("embedx_concate_filter");
    bool fix_ctr_to_click = ctx.Attr<bool>("fix_ctr_to_click");

    if (embedx_concate_size != 1 && embed_thres_size != 0) {
      CHECK(1 == 0) << "embedx_concate_size: " << embedx_concate_size
                    << "embed_thres_size: " << embed_thres_size;
    }

    auto x0_lod = ins[0]->lod();
    auto x0_dims = ins[0]->dims();
    auto y_dims = out[0]->dims();
    int embedding_size = ins[0]->numel() / ins[0]->dims()[0];
    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();
    int bs = x0_lod[0].size() - 1;
    int slot_num = static_cast<int>(ins.size());
    framework::LoD y_lod(1);
    y_lod[0].resize(bs + 1);
    for (int i = 0; i <= bs; ++i) {
        y_lod[0][i] = i;
    }
    for (int i = 0; i < slot_num; i++) {
      // out[i]->Resize({static_cast<int64_t>(bs), y_dims[1]});
      if (use_cvm) {
        if (clk_filter) {
          out[i]->Resize({bs, (embedding_size - 1) * embedx_concate_size});
        } else {
          out[i]->Resize({bs, embedding_size});
        }
      } else {
        out[i]->Resize({bs, (embedding_size - cvm_offset - embed_thres_size) * embedx_concate_size});
      }
    }
    //TODO:r480 l3 have some thing wrong
    static bool use_l3_tensor = std::getenv("XPU_PADDLE_L3_TENSOR")!=NULL ?
                      (std::strcmp(std::getenv("XPU_PADDLE_L3_TENSOR"), "1") == 0 ? true:false) :
                      false;
    auto place = ctx.GetPlace();
    phi::Place l3_place = ctx.template device_context<DeviceContext>().GetL3Place();
    int w = ins[0]->numel() / x0_dims[0];
    if(use_cvm) {
      if(clk_filter) w = w - 1;
      PADDLE_ENFORCE_EQ(y_dims[1] % w, 0,
                        paddle::platform::errors::InvalidArgument(
                            "The output of dims[1] should be dividable of w"));
    }
    else{
      PADDLE_ENFORCE_EQ(y_dims[1] % (w - cvm_offset - embed_thres_size), 0,
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
    TRACE_SCOPE_START("xpu::sequence_sum_pool_cvm", xpu_wait(xpu_context->xpu_stream););
#endif

    if (FLAGS_check_fused_negative_nan_inf) {
      xpu_wait(xpu_context->xpu_stream);
      check_tensors_nan(place, xpu_context, ins, "fused-x");
    }
    // int r = xpu::sequence_sum_pool_cvm<T>(xpu_context,
    int r = paddle::framework::sequence_sum_pool_cvm<T>(xpu_context,
                                          cpu_x_addr_vec,
                                          cpu_y_addr_vec,
                                          cpu_lodx,
                                          bs,
                                          x0_dims[1],
                                          slot_num,
                                          use_cvm,
                                          clk_filter,
                                          need_filter,
                                          padding_value,
                                          quant_ratio,
                                          show_coeff,
                                          clk_coeff,
                                          threshold,
                                          cvm_offset,
                                          embed_threshold_filter,
                                          embed_threshold,
                                          embed_thres_size,
                                          embedx_concate_size, 
                                          embedx_concate_filter,
                                          fix_ctr_to_click);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                     platform::errors::External(
                         "The sequence_sum_pool_cvm XPU OP return wrong value[%d %s]",
                         r, XPUAPIErrorMsg[r]));
    if (FLAGS_check_fused_negative_nan_inf) {
      xpu_wait(xpu_context->xpu_stream);
      check_tensors_nan(place, xpu_context, out, "fused-y");
    }
    TRACE_SCOPE_END("xpu::sequence_sum_pool_cvm", xpu_wait(xpu_context->xpu_stream););
    TRACE_SCOPE_END("FusedSeqpoolCVMOpXPUKernel Compute", xpu_wait(xpu_context->xpu_stream));
  }
};

template <typename DeviceContext, typename T>
class FusedSeqpoolCVMGradOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#ifdef TRACE_PROFILE
    TRACE_SCOPE_START("FusedSeqpoolCVMGradOpXPUKernel Compute", xpu_wait(ctx.template device_context<DeviceContext>().x_context()->xpu_stream));
#endif
    auto dOut = ctx.MultiInput<framework::LoDTensor>(framework::GradVarName("Out"));
    auto xs = ctx.MultiInput<LoDTensor>("X");
    const Tensor* cvm = ctx.Input<Tensor>("CVM");
    auto dxs = ctx.MultiOutput<framework::LoDTensor>(framework::GradVarName("X"));
    auto use_cvm = ctx.Attr<bool>("use_cvm");//TODO:
    bool clk_filter = ctx.Attr<bool>("clk_filter");
    auto cvm_offset = ctx.Attr<int>("cvm_offset");
    int embed_thres_size = ctx.Attr<int>("embed_thres_size");
    int embedx_concate_size = ctx.Attr<int>("embedx_concate_size");

    int slot_num = dxs.size();
    auto xpu_context = ctx.template device_context<DeviceContext>().x_context();
    auto place = ctx.GetPlace();
    // phi::Place l3_place = ctx.template device_context<DeviceContext>().GetL3Place();
    T* cvm_data = const_cast<T*>(cvm->data<T>());
    int batch_size = dOut[0]->dims()[0];
    // int dy_offset = dOut[0]->dims()[1];

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
        // T* dx_data = dx->mutable_data<T>(place);
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
      check_tensors_nan(place, xpu_context, dOut, "fused-dy");
    }
    // int r = xpu::sequence_sum_pool_cvm_grad<T>(xpu_context,
    int r = paddle::framework::sequence_sum_pool_cvm_grad<T>(xpu_context,
                                               cpu_dy_list,
                                               cvm_data,
                                               cpu_dx_list,
                                               cpu_lodx,
                                               use_cvm,
                                               cvm_offset,
                                               clk_filter,//split
                                               item_size,
                                               batch_size,
                                               slot_num,
                                               embed_thres_size,
                                               embedx_concate_size);

     PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
            platform::errors::External(
               "The sequence_pool_cvm_grad XPU OP return wrong value[%d %s]",
               r, XPUAPIErrorMsg[r]));
    if (FLAGS_check_fused_negative_nan_inf) {
      xpu_wait(xpu_context->xpu_stream);
      check_tensors_nan(place, xpu_context, dxs, "fused-dx");
    }
    TRACE_SCOPE_END("FusedSeqpoolCVMGradOpXPUKernel Compute", xpu_wait(xpu_context->xpu_stream));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(
    fused_seqpool_cvm,
    ops::FusedSeqpoolCVMOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    fused_seqpool_cvm_grad,
    ops::FusedSeqpoolCVMGradOpXPUKernel<paddle::platform::XPUDeviceContext, float>);
