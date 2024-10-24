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

#include "paddle/fluid/operators/fused/fused_causal_mask_op.h"

namespace paddle {
namespace operators {

class FusedCausalMaskOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("SeqInfoInput"),
                   "SeqInfoInput",
                   "SeqInfoInput",
                   "FusedCausalMask");
    OP_INOUT_CHECK(ctx->HasOutput("CausalMaskOutput"),
                   "CausalMaskOutput",
                   "CausalMaskOutput",
                   "FusedCausalMask");

    const framework::DDim seq_info_dims = ctx->GetInputDim("SeqInfoInput");
    PADDLE_ENFORCE_EQ(seq_info_dims.size(),
                      1,
                      platform::errors::InvalidArgument(
                          "The input tensor X's dimensions of FusedCausalMaskOp"
                          "should be 1, but received %d.",
                          seq_info_dims.size()));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "SeqInfoInput"),
        ctx.device_context());
  }
};

class FusedCausalMaskOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("SeqInfoInput",
             "(Tensor), The seq info tensor of fused_causal_mask op.");
    AddOutput("CausalMaskOutput",
              "(Tensor), The causal mask tensor of fused_causal_mask op.");

    AddComment(R"DOC(
Example:
>>> seq_info = paddle.to_tensor([0, s1, (s1+s2)...]) eg: [0, 3, 5] seq:3, 2...
>>> print(fused_causal_mask(seq_info))
array([[ True, False, False, False, False],
        [ True,  True, False, False, False],
        [ True,  True,  True, False, False],
        [ False, False, False,  True, False],
        [ False, False, False,  True,  True]])
"""
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fused_causal_mask,
                  ops::FusedCausalMaskOp,
                  ops::FusedCausalMaskOpMaker);

REGISTER_OP_CPU_KERNEL(fused_causal_mask,
                       ops::FusedCausalMaskOpCPUKernel<int>,
                       ops::FusedCausalMaskOpCPUKernel<int64_t>);