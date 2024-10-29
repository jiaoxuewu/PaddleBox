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

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_xpu.h"
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {

template <typename T>
class ElementwiseSubXPUKernel : public framework::OpKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
   auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              XPUType* z,
              const std::vector<int>& xshape,
              const std::vector<int>& yshape) {
    return xpu::broadcast_sub<XPUType>(ctx, x, y, z, xshape, yshape);
  };
    XPUElementwise<T, XPUType>(ctx, f);
  }
};

template <typename T>
class ElementwiseSubGradXPUKernel : public ElemwiseGradKernel<T> {
  using XPUType = typename XPUTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    ElemwiseGradKernel<T>::Compute(ctx);
    auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              const XPUType* z,
              const XPUType* dz,
              XPUType* dy,
              XPUType* dx,
              const std::vector<int>& xshape,
              const std::vector<int>& yshape) {
    return xpu::broadcast_sub_grad<XPUType>(
        ctx, x, y, z, dz, dy, dx, xshape, yshape);
  };
    XPUElementwiseGrad<T, XPUType>(
        ctx, f, false);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(elementwise_sub,
                       ops::ElementwiseSubXPUKernel<float>,
                       ops::ElementwiseSubXPUKernel<paddle::platform::float16>,
                       ops::ElementwiseSubXPUKernel<int64_t>);
REGISTER_OP_XPU_KERNEL(
    elementwise_sub_grad,
    ops::ElementwiseSubGradXPUKernel<float>,
    ops::ElementwiseSubGradXPUKernel<paddle::platform::float16>);

#endif
