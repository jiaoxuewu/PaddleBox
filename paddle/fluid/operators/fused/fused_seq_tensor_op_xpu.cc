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

#include "paddle/fluid/operators/fused/fused_seq_tensor_kernel.h"
#include "paddle/fluid/operators/fused/fused_seq_tensor_op.h"

#ifdef PADDLE_WITH_BOX_PS
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#else
#include "paddle/fluid/framework/threadpool.h"
#endif
#include <string>

namespace paddle {
namespace operators {
template <typename DeviceContext, typename T>
class FusedSeqTensorXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
   auto input = ctx.Input<framework::Tensor>("Input");
   PADDLE_ENFORCE_NOT_NULL(input, platform::errors::NotFound("Input not found"));
   auto ad_input = ctx.Input<framework::Tensor>("ADInput");
   PADDLE_ENFORCE_NOT_NULL(ad_input, platform::errors::NotFound("Input not found"));

   auto din_output = ctx.Output<framework::Tensor>("DINOut");
   PADDLE_ENFORCE_NOT_NULL(din_output,
                           platform::errors::NotFound("DINOut not found"));
   T* din_output_data = din_output->mutable_data<T>(ctx.GetPlace());
   auto mask_output = ctx.Output<framework::Tensor>("MaskOut");
   PADDLE_ENFORCE_NOT_NULL(mask_output,
                           platform::errors::NotFound("MaskOut not found"));
   T* mask_output_output_data = mask_output->mutable_data<T>(ctx.GetPlace());
   auto side_info_output = ctx.Output<framework::Tensor>("SideInfoOut");
   PADDLE_ENFORCE_NOT_NULL(side_info_output,
                           platform::errors::NotFound("Output not found"));
   T* side_info_output_data =
      side_info_output->mutable_data<T>(ctx.GetPlace());
   auto ad_slot_session_output =
      ctx.Output<framework::Tensor>("ADSlotSessionOut");
   PADDLE_ENFORCE_NOT_NULL(ad_slot_session_output,
                           platform::errors::NotFound("Output not found"));
   T* ad_slot_session_output_data =
      ad_slot_session_output->mutable_data<T>(ctx.GetPlace());

   auto batch_count = ctx.Attr<int64_t>("batch_count");
   auto max_length = ctx.Attr<int64_t>("max_length");
   auto slot_num = ctx.Attr<int64_t>("slot_num");
   auto fea_emb_dim = ctx.Attr<int64_t>("fea_emb_dim");
   auto ad_slot_num = ctx.Attr<int64_t>("ad_slot_num");
   auto ad_slot_offset = ctx.Attr<int64_t>("ad_slot_offset");
   
   auto input_dims = input->dims();
   size_t ins_num = input_dims[0];
   size_t sideinfo_slot_num = slot_num - ad_slot_num;

   size_t sideinfo_slot_offset = 0;
    if (ad_slot_offset == 0) {
      sideinfo_slot_offset = ad_slot_num;
    }

   auto xpu_context = ctx.template device_context<DeviceContext>().x_context();

   paddle::framework::cal_ad_slot_session<T>(input->data<T>(), ad_input->data<T>(), din_output_data, 
                                          ad_slot_session_output_data, batch_count, ins_num, slot_num, max_length, 
                                          fea_emb_dim, ad_slot_num, ad_slot_offset, xpu_context);

   paddle::framework::cal_sideinfo<T>(input->data<T>(), side_info_output_data, batch_count, ins_num, slot_num, max_length, 
                                   fea_emb_dim, sideinfo_slot_num, sideinfo_slot_offset, xpu_context);

   paddle::framework::cal_ad_mask<T>(input->data<T>(), mask_output_output_data, batch_count, ins_num, slot_num, 
                                  max_length, fea_emb_dim, xpu_context);
                                  
   }
};

} // namespace operators
} // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_XPU_KERNEL(
    fused_seq_tensor,
    ops::FusedSeqTensorXPUKernel<paddle::platform::XPUDeviceContext, float>);