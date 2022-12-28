/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/alias_method_op.h"

namespace paddle {
namespace operators {

class AliasMethodOpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Accept", "(Tensor), The accept input tensor of alias_method op.");
    AddInput("Alias", "(Tensor), The alias input tensor of alias_method op.");
    AddInput("Noids",
             "(Tensor), The no contains ids input tensor of alias_method op.");
    AddAttr<int>("Num", "(int), The number of alias_method op.");
    AddOutput("Out", "(Tensor), The output tensor of alias_method op.");
    AddComment(R"DOC(
AliasMethod Operator. https://en.wikipedia.org/wiki/Alias_method
This operator is a family of efficient algorithms for sampling from"
" a discrete probability distribution.
)DOC");
  }
};

class AliasMethodOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Accept"), true,
        platform::errors::NotFound(
            "Input(Accept) of AliasMethodOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Alias"), true,
                      platform::errors::NotFound(
                          "Input(Alias) of AliasMethodOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("Noids"), true,
                      platform::errors::NotFound(
                          "Input(Noids) of AliasMethodOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of AliasMethodOp should not be null."));

    auto accept_dims = ctx->GetInputDim("Accept");
    auto alias_dims = ctx->GetInputDim("Alias");

    PADDLE_ENFORCE_EQ(
        accept_dims, alias_dims,
        platform::errors::InvalidArgument(
            "Input accept'dim should be equal to Input alias'dim. "
            "But received Accept's shape = [%s], alias's shape = [%s].",
            accept_dims, alias_dims));

    int num = ctx->Attrs().Get<int>("Num");
    PADDLE_ENFORCE_GT(num, 0, platform::errors::InvalidArgument(
                                  "Input(Num) must greater than 0."));
    ctx->SetOutputDim("Out", framework::make_ddim({num}));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_WITHOUT_GRADIENT(alias_method, ops::AliasMethodOp,
                             ops::AliasMethodOpOpMaker)

REGISTER_OP_CPU_KERNEL(alias_method, ops::AliasMethodCPUKernel<float>)
