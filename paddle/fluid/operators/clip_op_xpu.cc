// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class ClipGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
      auto* x = ctx.Input<Tensor>("X");
      auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
      auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
      dx->mutable_data<T>(ctx.GetPlace());

      auto max = static_cast<T>(ctx.Attr<float>("max"));
      if (ctx.HasInput("Max")) {
        Tensor max_cpu;
        auto* max_t = ctx.Input<Tensor>("Max");
        auto* max_data = max_t->data<T>();
        if (platform::is_xpu_place(max_t->place())) {
            paddle::framework::TensorCopySync(
              *max_t, platform::CPUPlace(), &max_cpu);
            max_data = max_cpu.data<T>();
        }
        max = max_data[0];
      }

      auto min = ctx.Attr<float>("min");
      if (ctx.HasInput("Min")) {
          Tensor min_cpu;
          auto* min_t = ctx.Input<Tensor>("Min");
          auto* min_data = min_t->data<T>();
          if (platform::is_xpu_place(min_t->place())) {
              paddle::framework::TensorCopySync(
                *min_t, platform::CPUPlace(), &min_cpu);
             min_data = min_cpu.data<T>();
          }
          min = min_data[0];
      }

      using XPUDataType = typename XPUTypeTrait<T>::Type;
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      auto dx_data = reinterpret_cast<XPUDataType*>(dx->data<T>());
      auto x_data = reinterpret_cast<const XPUDataType*>(x->data<T>());
      auto dout_data = reinterpret_cast<const XPUDataType*>(dout->data<T>());

      // int clip_grad(Context* ctx, const T* x, const T* dy, T* dx, int64_t len, T min_val, T max_val)
      int r = xpu::clip_grad(
           dev_ctx.x_context(), x_data, dout_data, dx_data, dout->numel(), min, max);
      PADDLE_ENFORCE_EQ(
          r,
          XPU_SUCCESS,
          platform::errors::External("XPU API(clip_v2) return wrong "
                                   "value[%d %s]",
                                   r,
                                   XPUAPIErrorMsg[r]));
   }
};


template <typename DeviceContext, typename T>
class ClipXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    auto max = static_cast<T>(ctx.Attr<float>("max"));
    if (ctx.HasInput("Max")) {
      Tensor max_cpu;
      auto* max_t = ctx.Input<Tensor>("Max");
      auto* max_data = max_t->data<T>();
      if (platform::is_xpu_place(max_t->place())) {
        paddle::framework::TensorCopySync(
            *max_t, platform::CPUPlace(), &max_cpu);
        max_data = max_cpu.data<T>();
      }
      max = max_data[0];
    }

    auto min = ctx.Attr<float>("min");
    if (ctx.HasInput("Min")) {
      Tensor min_cpu;
      auto* min_t = ctx.Input<Tensor>("Min");
      auto* min_data = min_t->data<T>();
      if (platform::is_xpu_place(min_t->place())) {
        paddle::framework::TensorCopySync(
            *min_t, platform::CPUPlace(), &min_cpu);
        min_data = min_cpu.data<T>();
      }
      min = min_data[0];
    }

    using XPUDataType = typename XPUTypeTrait<T>::Type;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto x_data = reinterpret_cast<const XPUDataType*>(x->data<T>());
    auto out_data = reinterpret_cast<XPUDataType*>(out->data<T>());
    int r = xpu::clip_v2(
        dev_ctx.x_context(), x_data, out_data, x->numel(), min, max);
    PADDLE_ENFORCE_EQ(
        r,
        XPU_SUCCESS,
        platform::errors::External("XPU API(clip_v2) return wrong "
                                   "value[%d %s]",
                                   r,
                                   XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_XPU_KERNEL(clip, ops::ClipXPUKernel<plat::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(clip_grad, ops::ClipGradXPUKernel<plat::XPUDeviceContext, float>);

#endif
