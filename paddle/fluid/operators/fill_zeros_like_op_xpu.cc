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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/phi/core/lod_utils.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FillZerosLikeXPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& context) const override {
        auto* out = context.Output<framework::Tensor>("Out");
        auto& dev_ctx = context.template device_context<DeviceContext>();

        T* out_data = out->mutable_data<T>(dev_ctx.GetPlace());
        int ret = xpu::constant<T>(dev_ctx.x_context(), reinterpret_cast<T*>(out_data),
                                   out->numel(), static_cast<T>(0));

        PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                          phi::errors::External("XPU constant API return wrong value[%d %s].", ret,
                                                XPUAPIErrorMsg[ret]));
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(fill_zeros_like,
                       ops::FillZerosLikeXPUKernel<paddle::platform::XPUDeviceContext, float>);