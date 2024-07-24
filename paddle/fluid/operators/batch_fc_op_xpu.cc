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

#include <xpu/runtime.h>  // NOLINT
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#include "paddle/fluid/platform/device_context.h"

#include "paddle/fluid/operators/batch_fc_op.h"
#include "paddle/fluid/operators/batch_fc_kernel.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/xpu_api_wrapper.h"

namespace paddle {
namespace operators {
using framework::Tensor;

template <typename T>
class BatchFCXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int batchcount = ctx.Attr<int>("batchcount");
    auto transpose_weight = ctx.Attr<bool>("transpose_weight");
    if (transpose_weight) {
      // TODO
      PADDLE_ENFORCE_EQ(
        transpose_weight,
        true,
        platform::errors::Unimplemented("BatchFC not support transpose_weight now."));
      return;
    }
    if (batchcount > 0) {
      // TODO
      PADDLE_ENFORCE_EQ(
        (batchcount > 0),
        true,
        platform::errors::Unimplemented("BatchFC not support transpose_weight now."));
    } else {
      // X.dim = slot_pairs_num * ins_num * in_dim
      // W.dim = slot_pairs_num * in_dim * out_dim
      // b.dim = slot_pairs_num * out_dim
      // output.dim = slot_pairs_num * ins_num * out_dim
      auto* input = ctx.Input<framework::LoDTensor>("Input");
      auto* w = ctx.Input<Tensor>("W");
      auto* bias = ctx.Input<Tensor>("Bias");
      auto* output = ctx.Output<framework::LoDTensor>("Out");
      auto input_dims = input->dims();
      auto w_dims = w->dims();
      auto slot_pairs_num = input_dims[0];
      auto ins_num = input_dims[1];
      auto out_dim = w_dims[2];
  
      // get data ptr
      const XPUType* x_ptr = reinterpret_cast<const XPUType*>(input->data<T>());
      const XPUType* y_ptr = reinterpret_cast<const XPUType*>(w->data<T>());
      const XPUType* bias_data = reinterpret_cast<const XPUType*>(bias->data<T>());
  
      output->Resize({slot_pairs_num, ins_num, out_dim});
      XPUType* out_ptr = reinterpret_cast<XPUType*>(output->mutable_data<T>(ctx.GetPlace()));

      // initialize
      auto& dev_ctx = ctx.template device_context<paddle::platform::XPUDeviceContext>();
      auto xpu_context = dev_ctx.x_context();
      xpu::ctx_guard RAII_GUARD(xpu_context);
      
      // initialize
      phi::funcs::set_constant(dev_ctx, output, static_cast<T>(0));
  
      bool trans_x = false;
      bool trans_y = false;
  
      T alpha = 1;
     
      XpuFcInfo fc_info;
      GetFCInfo(input_dims, w_dims, trans_x, trans_y, &fc_info);
      MatMulXPUFunction<XPUType>(xpu_context, x_ptr, y_ptr, out_ptr, fc_info, alpha);
  
      // add bias
      paddle::framework::add_bias<T>(xpu_context,
                  out_ptr,
                  slot_pairs_num,
                  ins_num,
                  out_dim,
                  bias_data);
      xpu_wait(xpu_context->xpu_stream);
    }
  }
};

template <typename T>
class BatchFCGradOpXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int batchcount = ctx.Attr<int>("batchcount");
    if (batchcount > 0) {
      // TODO
      PADDLE_ENFORCE_EQ(
        (batchcount > 0),
        true,
        platform::errors::Unimplemented("BatchFC not support transpose_weight now."));
    } else {
      auto* input = ctx.Input<Tensor>("Input");
      auto* w = ctx.Input<Tensor>("W");
      auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
  
      auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
      auto* dw = ctx.Output<Tensor>(framework::GradVarName("W"));
      auto* db = ctx.Output<Tensor>(framework::GradVarName("Bias"));
  
      auto input_dims = input->dims();
      auto w_dims = w->dims();
      auto slot_pairs_num = input_dims[0];
      auto ins_num = input_dims[1];
      auto out_dim = w_dims[2];

      const XPUType* dout_ptr = reinterpret_cast<const XPUType*>(dout->data<T>());
      XPUType* x_ptr = reinterpret_cast<XPUType*>(dx->mutable_data<T>(ctx.GetPlace()));
      XPUType* y_ptr = reinterpret_cast<XPUType*>(dw->mutable_data<T>(ctx.GetPlace()));
      XPUType* b_ptr = reinterpret_cast<XPUType*>(db->mutable_data<T>(ctx.GetPlace()));

      auto& dev_ctx = ctx.template device_context<paddle::platform::XPUDeviceContext>();
      auto xpu_context = dev_ctx.x_context();
      xpu::ctx_guard RAII_GUARD(xpu_context);

      // initialize
      phi::funcs::set_constant(dev_ctx, dx, static_cast<T>(0));
      phi::funcs::set_constant(dev_ctx, dw, static_cast<T>(0));
      phi::funcs::set_constant(dev_ctx, db, static_cast<T>(0));

      bool transpose_x = false;
      bool transpose_y = false;
      XpuFcInfo info_forward;
      GetFCInfo(input_dims, w_dims, transpose_x, transpose_y, &info_forward);

      const XPUType* a_1 = reinterpret_cast<const XPUType*>(NULL);
      const XPUType* b_1 = reinterpret_cast<const XPUType*>(NULL);
      const XPUType* a_2 = reinterpret_cast<const XPUType*>(NULL);
      const XPUType* b_2 = reinterpret_cast<const XPUType*>(NULL);
      XPUType* c_1 = (dx == NULL) ? reinterpret_cast<XPUType*>(NULL)
                                  : reinterpret_cast<XPUType*>(dx->data<T>());
      XPUType* c_2 = (dw == NULL) ? reinterpret_cast<XPUType*>(NULL)
                                  : reinterpret_cast<XPUType*>(dw->data<T>());

      // add bias grad
      paddle::framework::add_bias_grad<T>(xpu_context,
                       dout_ptr,
                       slot_pairs_num,
                       ins_num,
                       out_dim,
                       b_ptr);
      xpu_wait(xpu_context->xpu_stream);

      T alpha = 1;
      XpuFcInfo info_dx;
      XpuFcInfo info_dy;
      std::tuple<XpuFcInfo,
                XpuFcInfo,
                const XPUType*,
                const XPUType*,
                const XPUType*,
                const XPUType*>
          fc_info = MatmulGradFcInfo(xpu_context,
                                    &RAII_GUARD,
                                    info_forward,
                                    transpose_x,
                                    transpose_y,
                                    x_ptr,
                                    y_ptr,
                                    dout_ptr);
      std::tie(info_dx, info_dy, a_1, b_1, a_2, b_2) = fc_info;
  
      // dx = dout_data * y^T
      MatMulXPUFunction<XPUType>(xpu_context, a_1, b_1, c_1, info_dx, alpha);
      
      // dy = x^T * dout_data
      MatMulXPUFunction<XPUType>(xpu_context, a_2, b_2, c_2, info_dy, alpha);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform; 

REGISTER_OP_XPU_KERNEL(batch_fc,
                       ops::BatchFCXPUKernel<float>);      
REGISTER_OP_XPU_KERNEL(batch_fc_grad,
                       ops::BatchFCGradOpXPUKernel<float>);  
