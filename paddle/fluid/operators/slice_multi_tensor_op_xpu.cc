// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/device_memory_aligment.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "xpu/refactor/math.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class SliceMultiTensorOpXPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& context) const override {
        auto fuse_tensor = context.Input<framework::LoDTensor>("Input");
        auto in_tensors = context.MultiInput<framework::LoDTensor>("X");
        // Init the continuous space
        auto out_tensors = context.MultiOutput<framework::LoDTensor>("Output");

        int id = context.Attr<int>("id");
        int num = context.Attr<int>("num");

        size_t in_size = in_tensors.size();
        size_t out_size = out_tensors.size();
        CHECK(in_size == out_size || out_size / num == in_size);

        // Make the outputs point to the continuous space.
        int64_t numel = fuse_tensor->numel();
        int64_t offset = (id * numel) / num;

        // adjust fuse
        auto& fuse_dim = fuse_tensor->dims();
        if (fuse_dim.size() > 1 && fuse_dim[0] != numel) {
            paddle::framework::DDim dim(fuse_dim);
            dim[0] = numel;
            dim[1] = 1;
            const_cast<framework::LoDTensor*>(fuse_tensor)->Resize(dim);
        }

        auto& dev_ctx = context.template device_context<DeviceContext>();
        const T* in_data = reinterpret_cast<const T*>(fuse_tensor->data<T>());

        for (size_t i = 0; i < out_tensors.size(); ++i) {
            size_t idx = i % in_size;
            auto dim = in_tensors[idx]->dims();
            size_t len = static_cast<size_t>(in_tensors[idx]->numel());
            CHECK(static_cast<int64_t>(offset + len) <= numel)
                << "fuse dim: " << fuse_dim.to_str() << ", dim:" << dim.to_str()
                << ", offset:" << offset << ", len:" << len;

            T* out_data = reinterpret_cast<T*>(out_tensors[i]->mutable_data<T>(dev_ctx.GetPlace()));
            int r = xpu::slice<T>(dev_ctx.x_context(), in_data, out_data, {numel, 1}, {offset, 0},
                                  {offset + len, 1});
            PADDLE_ENFORCE_EQ(r, XPU_SUCCESS,
                              phi::errors::External("XPU slice API return wrong value[%d %s].", r,
                                                    XPUAPIErrorMsg[r]));

            out_data.Resize(dim);
            offset += len;
        }
    }
};

}  // namespace operators
}  // namespace paddle
