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

#include "paddle/fluid/operators/rank_attention_op.h"

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T>
void rank_attention2_forward_kernel_cpu(const int ins_num,
                                        const int ins_col,
                                        const T* input,
                                        const int max_rank,
                                        const int* rank_offset,
                                        const int para_row,
                                        const int para_col,
                                        const T* para,
                                        T* out_val) {
    int rank_cols = max_rank * 2 + 1;

    // rank offset 2:1:46:2:44:3:45
    for (int idx = 0; idx < ins_num * para_col; idx++) {
        int row_id = idx / para_col;
        int col_id = idx % para_col;

        int lower = rank_offset[row_id * rank_cols] - 1;
        if (lower < 0) {
            out_val[idx] = 0.0;
            continue;
        }
        assert(lower < max_rank);

        float sum = 0.0;
        for (int k = 0; k < max_rank; ++k) {
            int faster = rank_offset[row_id * rank_cols + 2 * k + 1] - 1;
            // note look rank_offset to know why
            assert(faster < max_rank);
            if (faster < 0) {
                continue;
            }
            int index = rank_offset[row_id * rank_cols + 2 * k + 2];
            assert(index < ins_num);

            int start = (lower * max_rank + faster) * ins_col;
            assert(start + ins_col <= para_row);

            for (int j = 0; j < ins_col; ++j) {
                sum += input[index * ins_col + j] * para[(start + j) * para_col + col_id];
            }
        }
        out_val[idx] = sum;
    }
}

template <typename T>
void rank_attention2_backward_kernel_cpu(const int para_row,
                                         const int para_col,
                                         T* out_para_grad,
                                         const int ins_num,
                                         const int ins_col,
                                         const T* input,
                                         const int max_rank,
                                         const int* rank_offset,
                                         const T* out_grad) {
    int rank_cols = max_rank * 2 + 1;

    // rank offset 2:1:46:2:44:3:45
    for (int idx = 0; idx < ins_num * ins_col * para_col * max_rank; idx++) {
        int ins_id = idx / para_col / ins_col / max_rank;
        int para_col_id = (idx / ins_col / max_rank) % para_col;
        int ins_col_id = (idx / para_col / max_rank) % ins_col;
        int k = (idx / para_col / ins_col) % max_rank;

        int lower = rank_offset[ins_id * rank_cols] - 1;
        if (lower < 0) {
            continue;
        }
        assert(lower < max_rank);

        int faster = rank_offset[ins_id * rank_cols + 2 * k + 1] - 1;
        // note look rank_offset to know why
        assert(faster < max_rank);
        if (faster < 0) {
            continue;
        }

        int index = rank_offset[ins_id * rank_cols + 2 * k + 2];
        assert(index < ins_num);

        int start = (lower * max_rank + faster) * ins_col;
        assert(start + ins_col <= para_row);

        float* tmp =
            reinterpret_cast<float*>(&out_para_grad[(start + ins_col_id) * para_col + para_col_id]);
        out_para_grad[(start + ins_col_id) * para_col + para_col_id] =
            *tmp + input[index * ins_col + ins_col_id] * out_grad[ins_id * para_col + para_col_id];
    }
}

template <typename DeviceContext, typename T>
class RankAttention2CPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* X = ctx.Input<Tensor>("X");
        auto* rank_offset = ctx.Input<Tensor>("RankOffset");
        auto* param = ctx.Input<Tensor>("RankParam");
        int max_rank = ctx.Attr<int>("MaxRank");
        auto* Out = ctx.Output<Tensor>("Out");

        // check dims
        auto x_dims = X->dims();
        auto ins_num = x_dims[0];
        auto x_fea_dim = x_dims[1];
        auto para_dims = param->dims();
        auto para_row = para_dims[0];
        auto para_col = para_dims[1];
        auto rank_offset_dims = rank_offset->dims();

        PADDLE_ENFORCE_EQ(rank_offset_dims[0], ins_num,
                          platform::errors::InvalidArgument("Input(RankOffset) has wrong rows."));
        PADDLE_ENFORCE_EQ(
            (rank_offset_dims[1] - 1) / 2, max_rank,
            platform::errors::InvalidArgument("Input(RankOffset) has wrong columns."));
        PADDLE_ENFORCE_EQ(max_rank * max_rank * x_fea_dim, para_row,
                          platform::errors::InvalidArgument("Input(RankParam) has wrong rows."));

        // get data ptr
        T* out_data = Out->mutable_data<T>(ctx.GetPlace());
        rank_attention2_forward_kernel_cpu<T>(ins_num, x_fea_dim, X->data<T>(), max_rank,
                                              rank_offset->data<int>(), para_row, para_col,
                                              param->data<T>(), out_data);
    }
};

template <typename DeviceContext, typename T>
class RankAttention2GradCPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* X = ctx.Input<Tensor>("X");                     // not use data
        auto* rank_offset = ctx.Input<Tensor>("RankOffset");  // not use data
        auto* param = ctx.Input<Tensor>("RankParam");         // not use data
        auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
        auto* drank_para = ctx.Output<Tensor>(framework::GradVarName("RankParam"));

        // get dim
        auto x_dims = X->dims();
        auto ins_num = x_dims[0];
        auto x_fea_dim = x_dims[1];
        auto para_dims = param->dims();
        auto para_row = para_dims[0];
        auto para_col = para_dims[1];
        auto rank_offset_dims = rank_offset->dims();
        auto max_rank = (rank_offset_dims[1] - 1) / 2;

        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        // initialize out grad
        T* drank_para_ptr = drank_para->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, drank_para, 0.0);

        rank_attention2_backward_kernel_cpu<T>(para_row, para_col, drank_para_ptr, ins_num,
                                               x_fea_dim, X->data<T>(), max_rank,
                                               rank_offset->data<int>(), dout->data<T>());
    }
};

class RankAttentionOp : public framework::OperatorWithKernel {
   public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("X"), true,
            platform::errors::InvalidArgument("Input(X) of RankAttentionOp should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasInput("RankOffset"), true,
                          platform::errors::InvalidArgument(
                              "Input(RankOffset) of RankAttentionOp should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasInput("RankParam"), true,
                          platform::errors::InvalidArgument(
                              "Input(RankParam) of RankAttentionOp should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasOutput("InsRank"), true,
                          platform::errors::InvalidArgument(
                              "Output(InsRank) of RankAttentionOp should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasOutput("InputHelp"), true,
                          platform::errors::InvalidArgument(
                              "Output(InputHelp) of RankAttentionOp should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasOutput("ParamHelp"), true,
                          platform::errors::InvalidArgument(
                              "Output(ParamHelp) of RankAttentionOp should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                          platform::errors::InvalidArgument(
                              "Output(Out) of RankAttentionOp should not be null."));
        auto max_rank = ctx->Attrs().Get<int>("MaxRank");

        auto x_dims = ctx->GetInputDim("X");
        auto ins_num = x_dims[0];
        auto param_dims = ctx->GetInputDim("RankParam");
        auto para_col = param_dims[1];
        auto rank_offset_dims = ctx->GetInputDim("RankOffset");
        auto x_fea_dim = x_dims[1];
        auto block_matrix_row = max_rank * x_fea_dim;

        PADDLE_ENFORCE_EQ(
            (rank_offset_dims[1] - 1) / 2, max_rank,
            platform::errors::InvalidArgument("Input(RankOffset) has wrong columns."));

        ctx->SetOutputDim("Out", {ins_num, para_col});
        ctx->SetOutputDim("InputHelp", {ins_num, block_matrix_row});
        ctx->SetOutputDim("ParamHelp", {ins_num * block_matrix_row, para_col});
        ctx->SetOutputDim("InsRank", {ins_num, 1});
        ctx->ShareLoD("X", /*->*/ "Out");
    }

   protected:
    framework::OpKernelType GetExpectedKernelType(
        const framework::ExecutionContext& ctx) const override {
        return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                                       ctx.device_context());
    }
};

class RankAttentionGradOp : public framework::OperatorWithKernel {
   public:
    using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("RankParam"), true,
                      platform::errors::InvalidArgument(
                          "Input(RankParam) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("RankOffset"), true,
                      platform::errors::InvalidArgument(
                          "Input(RankOffset) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("InputHelp"), true,
                      platform::errors::InvalidArgument(
                          "Input(InputHelp) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("InsRank"), true,
        platform::errors::InvalidArgument("Input(InsRank) should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput("ParamHelp"), true,
                      platform::errors::InvalidArgument(
                          "Input(ParamHelp) should not be null"));

    ctx->SetOutputDim(framework::GradVarName("RankParam"),
                      ctx->GetInputDim("RankParam"));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

   protected:
    framework::OpKernelType GetExpectedKernelType(
        const framework::ExecutionContext& ctx) const override {
        return framework::OpKernelType(
            OperatorWithKernel::IndicateVarDataType(ctx, framework::GradVarName("Out")),
            ctx.device_context());
    }
};

class RankAttentionOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of rank_attention_Op operator.");
    AddInput("RankOffset",
             "(Tensor) Input tensor of rank_attention_Op operator.");
    AddInput("RankParam",
             "(Tensor) Input tensor of rank_attention_Op operator.");
    AddOutput("InputHelp", "Output tensor of rank_attention_Op operator.")
        .AsDispensable();
    AddOutput("ParamHelp", "Output tensor of rank_attention_Op operator.")
        .AsDispensable();
    AddOutput("Out", "Output tensor of rank_attention_Op operator.");
    AddOutput("InsRank", "Output tensor of rank_attention_Op operator.")
        .AsDispensable();
    AddAttr<int>("MaxRank", "(int, default 3) max rank of rank_attention_Op")
        .SetDefault(3);
    AddAttr<int>("MaxSize", "(int, default 0) max rank of rank_attention_Op")
        .SetDefault(0);
    AddAttr<bool>("EnableInputBp", "(bool, default false) input bp switch of rank_attention_Op")
        .SetDefault(false);
    AddComment(R"DOC(
RankAttention Operator.
This Op can calculate rank attention between input and rank_param,
and rank_param gives the organization of data. Notice: It currently supports GPU device.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
    }
};

template <typename T>
class RankAttentionGradOpMaker : public framework::SingleGradOpMaker<T> {
   public:
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

   protected:
    void Apply(GradOpPtr<T> op) const override {
        op->SetType("rank_attention_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("RankOffset", this->Input("RankOffset"));
    op->SetInput("RankParam", this->Input("RankParam"));
    op->SetInput("InputHelp", this->Output("InputHelp"));
    op->SetInput("ParamHelp", this->Output("ParamHelp"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput("InsRank", this->Output("InsRank"));

    op->SetOutput(framework::GradVarName("RankParam"),
                  this->InputGrad("RankParam"));
    op->SetOutput(framework::GradVarName("X"),
                  this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(RankAttentionGradOpNoNeedBufferVarsInference,
                                    "X",
                                    "RankOffset",
                                    "RankParam");

class RankAttention2Op : public framework::OperatorWithKernel {
   public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("X"), true,
            platform::errors::InvalidArgument("Input(X) of RankAttentionOp should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasInput("RankOffset"), true,
                          platform::errors::InvalidArgument(
                              "Input(RankOffset) of RankAttentionOp should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasInput("RankParam"), true,
                          platform::errors::InvalidArgument(
                              "Input(RankParam) of RankAttentionOp should not be null."));
        PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                          platform::errors::InvalidArgument(
                              "Output(Out) of RankAttentionOp should not be null."));
        auto max_rank = ctx->Attrs().Get<int>("MaxRank");
        VLOG(3) << "max_rank:" << max_rank;
        auto x_dims = ctx->GetInputDim("X");
        auto ins_num = x_dims[0];
        auto param_dims = ctx->GetInputDim("RankParam");
        auto para_col = param_dims[1];
        auto rank_offset_dims = ctx->GetInputDim("RankOffset");

        PADDLE_ENFORCE_EQ(
            (rank_offset_dims[1] - 1) / 2, max_rank,
            platform::errors::InvalidArgument("Input(RankOffset) has wrong columns."));

        ctx->SetOutputDim("Out", {ins_num, para_col});
        ctx->ShareLoD("X", /*->*/ "Out");
    }

   protected:
    framework::OpKernelType GetExpectedKernelType(
        const framework::ExecutionContext& ctx) const override {
        return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(ctx, "X"),
                                       ctx.device_context());
    }
};

class RankAttention2GradOp : public framework::OperatorWithKernel {
   public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                          platform::errors::InvalidArgument("Input(X) should not be null"));
        PADDLE_ENFORCE_EQ(ctx->HasInput("RankParam"), true,
                          platform::errors::InvalidArgument("Input(RankParam) should not be null"));
        PADDLE_ENFORCE_EQ(
            ctx->HasInput("RankOffset"), true,
            platform::errors::InvalidArgument("Input(RankOffset) should not be null"));

        ctx->SetOutputDim(framework::GradVarName("RankParam"), ctx->GetInputDim("RankParam"));
    }

   protected:
    framework::OpKernelType GetExpectedKernelType(
        const framework::ExecutionContext& ctx) const override {
        return framework::OpKernelType(
            OperatorWithKernel::IndicateVarDataType(ctx, framework::GradVarName("Out")),
            ctx.device_context());
    }
};

class RankAttention2OpMaker : public framework::OpProtoAndCheckerMaker {
   public:
    void Make() override {
        AddInput("X", "(Tensor) Input tensor of rank_attention_Op operator.");
        AddInput("RankOffset", "(Tensor) Input tensor of rank_attention_Op operator.");
        AddInput("RankParam", "(Tensor) Input tensor of rank_attention_Op operator.");
        AddOutput("Out", "Output tensor of rank_attention_Op operator.");
        AddAttr<int>("MaxRank", "(int, default 3) max rank of rank_attention_Op").SetDefault(3);
        AddComment(R"DOC(
RankAttention Operator.
This Op can calculate rank attention between input and rank_param,
and rank_param gives the organization of data. Notice: It currently supports GPU device.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
    }
};

template <typename T>
class RankAttention2GradOpMaker : public framework::SingleGradOpMaker<T> {
   public:
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

   protected:
    void Apply(GradOpPtr<T> op) const override {
        op->SetType("rank_attention2_grad");

        op->SetInput("X", this->Input("X"));
        op->SetInput("RankOffset", this->Input("RankOffset"));
        op->SetInput("RankParam", this->Input("RankParam"));
        op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

        op->SetOutput(framework::GradVarName("RankParam"), this->InputGrad("RankParam"));
        op->SetAttrMap(this->Attrs());
    }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(RankAttention2GradOpNoNeedBufferVarsInference,
                                    "X",
                                    "RankOffset",
                                    "RankParam");
}  // namespace operators
}  // namespace paddle
using CPUCtx = phi::CPUContext;
namespace ops = paddle::operators;
REGISTER_OPERATOR(rank_attention,
                  ops::RankAttentionOp,
                  ops::RankAttentionOpMaker,
                  ops::RankAttentionGradOpMaker<paddle::framework::OpDesc>,
                  ops::RankAttentionGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(rank_attention_grad,
                  ops::RankAttentionGradOp,
                  ops::RankAttentionGradOpNoNeedBufferVarsInference);

REGISTER_OPERATOR(rank_attention2,
                  ops::RankAttention2Op,
                  ops::RankAttention2OpMaker,
                  ops::RankAttention2GradOpMaker<paddle::framework::OpDesc>,
                  ops::RankAttention2GradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(rank_attention2_grad, ops::RankAttention2GradOp);

// REGISTER_OPERATOR(rank_attention2_grad, ops::RankAttention2GradOp,
//                   ops::RankAttention2GradOpNoNeedBufferVarsInference);

REGISTER_OP_CPU_KERNEL(rank_attention,
                       ops::RankAttentionKernel<CPUCtx, float>,
                       ops::RankAttentionKernel<CPUCtx, double>);

REGISTER_OP_CPU_KERNEL(rank_attention2,
                       ops::RankAttention2CPUKernel<CPUCtx, float>,
                       ops::RankAttentionKernel<CPUCtx, double>);

REGISTER_OP_CPU_KERNEL(rank_attention2_grad,
                       ops::RankAttention2GradCPUKernel<CPUCtx, float>,
                       ops::RankAttentionKernel<CPUCtx, double>);
