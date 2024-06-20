#include "paddle/fluid/operators/fused/fused_seq_tensor_op.h"
#include "paddle/fluid/framework/op_registry.h"
#include <string>

namespace paddle {
namespace operators {

class FusedSeqTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input",  "Input", "FusedSeqTensorOp");
    OP_INOUT_CHECK(ctx->HasInput("ADInput"), "ADInput",  "ADInput", "FusedSeqTensorOp");
    
    OP_INOUT_CHECK(ctx->HasOutput("DINOut"), "DINOut", "DINOut", "FusedSeqTensorOp");
    OP_INOUT_CHECK(ctx->HasOutput("MaskOut"), "MaskOut", "MaskOut", "FusedSeqTensorOp");
    OP_INOUT_CHECK(ctx->HasOutput("SideInfoOut"), "SideInfoOut", "SideInfoOut", "FusedSeqTensorOp");
    OP_INOUT_CHECK(ctx->HasOutput("ADSlotSessionOut"), "ADSlotSessionOut", "ADSlotSessionOut", "FusedSeqTensorOp");

    const framework::DDim input_dims = ctx->GetInputDim("Input");
    const framework::DDim ad_input_dims = ctx->GetInputDim("ADInput");

    auto ad_slot_num = ctx->Attrs().Get<int64_t>("ad_slot_num");
    auto batch_count = ctx->Attrs().Get<int64_t>("batch_count");
    auto max_length = ctx->Attrs().Get<int64_t>("max_length");
    auto slot_num = ctx->Attrs().Get<int64_t>("slot_num");
    auto fea_emb_dim = ctx->Attrs().Get<int64_t>("fea_emb_dim");
    auto ad_slot_offset = ctx->Attrs().Get<int64_t>("ad_slot_offset");

    int64_t one_ins_dim = batch_count * max_length * slot_num * fea_emb_dim;
    PADDLE_ENFORCE_EQ(
        input_dims[1], one_ins_dim,
        platform::errors::InvalidArgument(
          "input dims error, %ld != %ld", input_dims[1], one_ins_dim));

    int64_t one_ins_ad_dim = batch_count * 1 * ad_slot_num * fea_emb_dim;
    PADDLE_ENFORCE_EQ(
        ad_input_dims[1], one_ins_ad_dim,
        platform::errors::InvalidArgument(
          "input dims error, %ld != %ld", ad_input_dims[1], one_ins_ad_dim));
    PADDLE_ENFORCE_LT(
        ad_slot_num, slot_num, 
        platform::errors::InvalidArgument(
          "ad_slot_num [%ld] >  slot_num [%ld]", ad_slot_num, slot_num));
    PADDLE_ENFORCE_GT(
        ad_slot_num, 0, 
        platform::errors::InvalidArgument(
          "ad_slot_num [%ld] <= 0", ad_slot_num));
    PADDLE_ENFORCE_LE(
        ad_slot_offset, slot_num - 1, 
        platform::errors::InvalidArgument(
          "ad_slot_num [%ld] > slot_num - 1 [%ld]", ad_slot_offset, slot_num));
    PADDLE_ENFORCE_GE(
        ad_slot_offset, 0, 
        platform::errors::InvalidArgument(
          "ad_slot_offset [%ld] < 0", ad_slot_offset));
    if (ad_slot_offset != 0) {
      PADDLE_ENFORCE_EQ(
          ad_slot_num + ad_slot_offset, slot_num, 
          platform::errors::InvalidArgument(
            "ad_slot_num [%ld] + ad_slot_offset [%ld] !=  slot_num [%ld]", ad_slot_num, ad_slot_offset, slot_num));
    }
    
    auto ins_num = input_dims[0];
    if (batch_count > 1) {
      ctx->SetOutputDim("DINOut", {batch_count, ins_num * max_length, ad_slot_num * fea_emb_dim * 4});
      ctx->SetOutputDim("MaskOut", {batch_count, ins_num, max_length});
      ctx->SetOutputDim("SideInfoOut", {batch_count, ins_num * max_length, (slot_num - ad_slot_num) * fea_emb_dim});
      ctx->SetOutputDim("ADSlotSessionOut", {batch_count, ins_num * max_length, ad_slot_num, fea_emb_dim});
    } else {
      ctx->SetOutputDim("DINOut", {ins_num, max_length, ad_slot_num * fea_emb_dim * 4});
      ctx->SetOutputDim("MaskOut", {ins_num, max_length});
      ctx->SetOutputDim("SideInfoOut", {ins_num, max_length, (slot_num - ad_slot_num) * fea_emb_dim});
      ctx->SetOutputDim("ADSlotSessionOut", {ins_num, max_length, ad_slot_num * fea_emb_dim});
    }
    ctx->ShareLoD("Input", "DINOut");
    ctx->ShareLoD("Input", "MaskOut");
    ctx->ShareLoD("Input", "SideInfoOut");
    ctx->ShareLoD("Input", "ADSlotSessionOut");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class FusedSeqTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "The input tensors of operator.");
    AddInput("ADInput",
             "The input ad tensors of operator. ");
    AddOutput("DINOut",
              "DINOut");
    AddOutput("MaskOut",
              "MaskOut");
    AddOutput("SideInfoOut",
              "SideInfoOut");
    AddOutput("ADSlotSessionOut",
              "ADSlotSessionOut");

    AddAttr<int64_t>("batch_count", "(int, default 1)");
    AddAttr<int64_t>("max_length", "(int, default 1)");
    AddAttr<int64_t>("slot_num", "(int, default 1)");
    AddAttr<int64_t>("fea_emb_dim", "(int, default 1)");
    AddAttr<int64_t>("ad_slot_num", "(int, default 1)");
    AddAttr<int64_t>("ad_slot_offset", "(int, default 1)");

    AddComment(R"DOC(
Fuse seq tensor.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(fused_seq_tensor, 
                  ops::FusedSeqTensorOp, ops::FusedSeqTensorOpMaker);

REGISTER_OP_CPU_KERNEL(
  fused_seq_tensor,
  ops::FusedSeqTensorCPUKernel<phi::CPUContext, float>);
