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

#include "paddle/fluid/operators/fused/fused_concat_op.h"
#include <string>

namespace paddle {
namespace operators {

class FusedSeqpoolConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    const int total_cols = ctx->Attrs().Get<int>("output_dim");
    auto ins_dims = ctx->GetInputsDim("X1");
    auto ins_dims2 = ctx->GetInputsDim("X2");
    PADDLE_ENFORCE_EQ(
        ins_dims.size(), ins_dims2.size(),
        platform::errors::InvalidArgument(
            "The dims slot size of first and second should be equal, "
            "but received value is x1: %d, x2: %d",
            ins_dims.size(), ins_dims2.size()));

    const std::vector<int> idxs =
        ctx->Attrs().Get<std::vector<int>>("output_idx");
    PADDLE_ENFORCE_EQ(
        idxs.size(), total_cols * 3,
        platform::errors::InvalidArgument(
            "The dims slot size of first and second should be equal, "
            "but received value is idx size: %d, cols: %d",
            idxs.size(), total_cols));

    const size_t num_inputs = ins_dims.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(num_inputs);

    PADDLE_ENFORCE_GT(num_inputs, 0UL,
                      platform::errors::InvalidArgument(
                          "Input tensors count should be greater than 0, "
                          "but received value is %d.",
                          num_inputs));

    // The output height should be confirmed in Compute,
    // since input lod is not accessible here.
    PADDLE_ENFORCE_EQ(ins_dims[0].size(), 2,
                      platform::errors::InvalidArgument(
                          "The dims size of first input should be equal to 2, "
                          "but received value is %d.",
                          ins_dims[0].size()));

    std::vector<int64_t> out_dim;
    out_dim = {-1, total_cols};
    for (size_t i = 0; i < num_inputs; ++i) {
      // input lod is not accessible here
      outs_dims[i] = phi::make_ddim(out_dim);
    }
    ctx->SetOutputsDim("Out", outs_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
  }
};

class FusedSeqpoolConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X1",
             "(vector<LoDTensor>) The input tensors of"
             " operator.")
        .AsDuplicable();
    AddInput("X2",
             "(vector<LoDTensor>) The input tensors of"
             " operator.")
        .AsDuplicable();
    AddOutput("Out",
              "(vector<Tensor>) The output of Op does not contain LoD "
              "information.")
        .AsDuplicable();
    AddAttr<std::vector<int>>(
        "output_idx",
        "(std::vector<int>, default {}) The value concat cols idxs.")
        .SetDefault({});
    AddAttr<int>("output_dim",
                 "(int, default 0) The value concat total cols num.")
        .SetDefault(0);
    AddComment(R"DOC(
Fuse multiple pairs of Sequence Pool and CVM Operator.

)DOC");
  }
};

class FusedSeqpoolConcatGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    //    auto og_dims = ctx->GetInputsDim(framework::GradVarName("Out"));
    auto out_x_g_n = framework::GradVarName("X1");
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim("X1"));
    ctx->ShareAllLoD("X1", out_x_g_n);

    out_x_g_n = framework::GradVarName("X2");
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim("X2"));
    ctx->ShareAllLoD("X2", out_x_g_n);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class FusedSeqpoolConcatGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("fused_seqpool_concat_grad");
    op_desc_ptr->SetInput("X1", this->Input("X1"));
    op_desc_ptr->SetInput("X2", this->Input("X2"));
    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));

    op_desc_ptr->SetOutput(framework::GradVarName("X1"),
                           this->InputGrad("X1", false));
    op_desc_ptr->SetOutput(framework::GradVarName("X2"),
                           this->InputGrad("X2", false));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};

//============================== equal dim concat ============================
// [x1, x2, x3, x4] => out
class FusedConcatOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    const int length = ctx->Attrs().Get<int>("length");
    PADDLE_ENFORCE_GT(length, 0,
                      platform::errors::InvalidArgument(
                          "Input length more than zero %d.", length));

    auto ins_dims = ctx->GetInputsDim("X");
    int batch_size = ins_dims[0][0];
    const int input_nums = ins_dims.size();
    PADDLE_ENFORCE_GT(input_nums, 1UL,
                      platform::errors::InvalidArgument(
                          "Input tensors count should be greater than 0, "
                          "but received value is %d.",
                          ins_dims.size()));
    std::vector<int64_t> out_dim;
    out_dim = {batch_size, length * input_nums};
    ctx->SetOutputDim("Out", phi::make_ddim(out_dim));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
  }
};

class FusedConcatOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(vector<LoDTensor>) The input tensors of"
             " operator.")
        .AsDuplicable();
    AddOutput("Out",
              "(Tensor) The output of Op does not contain LoD "
              "information.");
    AddAttr<int>("offset", "(int, default 0) The value concat cols start.")
        .SetDefault(0);
    AddAttr<int>("length", "(int, default 0) The value concat cols num.")
        .SetDefault(0);
    AddComment(R"DOC(
Fuse multiple pairs of Sequence Pool and CVM Operator.

)DOC");
  }
};

class FusedConcatGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    //    auto og_dims = ctx->GetInputsDim(framework::GradVarName("Out"));
    auto out_x_g_n = framework::GradVarName("X");
    ctx->SetOutputsDim(out_x_g_n, ctx->GetInputsDim("X"));
    ctx->ShareAllLoD("X", out_x_g_n);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class FusedConcatGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("fused_concat_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));

    op_desc_ptr->SetOutput(framework::GradVarName("X"),
                           this->InputGrad("X", false));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(
    fused_seqpool_concat, ops::FusedSeqpoolConcatOp,
    ops::FusedSeqpoolConcatOpMaker,
    ops::FusedSeqpoolConcatGradOpMaker<paddle::framework::OpDesc>,
    ops::FusedSeqpoolConcatGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_seqpool_concat_grad, ops::FusedSeqpoolConcatGradOp)
REGISTER_OP_CPU_KERNEL(fused_seqpool_concat,
                       ops::FusedSeqpoolConcatOpCPUKernel<float>)
REGISTER_OP_CPU_KERNEL(fused_seqpool_concat_grad,
                       ops::FusedSeqpoolConcatGradOpCPUKernel<float>)

REGISTER_OPERATOR(fused_concat, ops::FusedConcatOp, ops::FusedConcatOpMaker,
                  ops::FusedConcatGradOpMaker<paddle::framework::OpDesc>,
                  ops::FusedConcatGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_concat_grad, ops::FusedConcatGradOp)
REGISTER_OP_CPU_KERNEL(fused_concat, ops::FusedConcatOpCPUKernel<float>)
REGISTER_OP_CPU_KERNEL(fused_concat_grad,
                       ops::FusedConcatGradOpCPUKernel<float>)
