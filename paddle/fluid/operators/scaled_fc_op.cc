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

#include "paddle/fluid/operators/scaled_fc_op.h"
#include <string>
#include "paddle/phi/kernels/funcs/blas/blas.h"

using CPUCtx = phi::CPUContext;

namespace paddle {
namespace operators {
using framework::Tensor;

template <typename T>
void cast_and_padding_cpu_kernel(const unsigned int rown_ori,
                                 const unsigned int coln_ori,
                                 const unsigned int rown_pad,
                                 const unsigned int coln_pad,
                                 const T* matrix,
                                 float* matrix_pad,
                                 T grad_scale_factor) {
    int N = rown_pad * coln_pad;
    for (int i = 0; i < N; i++) {
        unsigned int col_idx = i % coln_pad;
        unsigned int row_idx = i / coln_pad;
        if (row_idx < rown_ori && col_idx < coln_ori) {
            int idx = row_idx * coln_ori + col_idx;
            matrix_pad[i] = static_cast<float>(matrix[idx] * grad_scale_factor);
        } else {
            matrix_pad[i] = static_cast<float>(0.0);
        }
    }
}

template <typename T>
void vec_mat_row_add_cpu_kernel(const unsigned int rown,
                                const unsigned int coln,
                                T* matrix,
                                const T* vector,
                                const T bias_scale_factor_use) {
    int N = rown * coln;
    for (int i = 0; i < N; i++) {
        matrix[i] += vector[i % coln] * bias_scale_factor_use;
    }
}

template <typename T>
void cast_and_cut_cpu_kernel(const unsigned int rown_ori,
                             const unsigned int coln_ori,
                             const unsigned int rown_pad,
                             const unsigned int coln_pad,
                             T* matrix,
                             float* matrix_pad,
                             T scale_factor) {
    int N = rown_ori * coln_ori;
    for (int i = 0; i < N; i++) {
        int col_idx = i % coln_ori;
        int row_idx = i / coln_ori;
        int idx = row_idx * coln_pad + col_idx;
        matrix[i] = static_cast<T>(matrix_pad[idx]);
        matrix[i] *= scale_factor;
    }
}

template <typename T>
void col_sum_mat_cpu_kernel(const unsigned int rown,
                            const unsigned int coln,
                            const T* matrix,
                            T* vector,
                            const T bias_scale_factor_use) {
    for (unsigned int i = 0; i < coln; i++) {
        for (unsigned int j = 0; j < rown; j++) {
            vector[i] += matrix[j * coln + i];
        }
    }
}

template <typename DeviceContext, typename T>
class ScaledFCCPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* input = ctx.Input<framework::LoDTensor>("Input");  // framework::Tensor*
        auto* w = ctx.Input<Tensor>("W");
        auto* bias = ctx.Input<Tensor>("Bias");
        auto* output = ctx.Output<framework::LoDTensor>("Out");
        auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
        auto bias_scale_factor = ctx.Attr<float>("bias_scale_factor");

        auto input_dims = input->dims();
        auto w_dims = w->dims();
        auto ins_num = input_dims[0];  // oriinput: ins_num*in_feat, oriweight: in_feat* out_fea,
                                       // output: ins_num* out_feat
        auto in_feat = input_dims[1];
        auto out_feat = w_dims[1];

        output->mutable_data<T>(ctx.GetPlace());
        output->Resize({ins_num, w_dims[1]});

        auto& dev_ctx = ctx.template device_context<CPUCtx>();

        // begin cast and pad
        const unsigned int insnum_ori = ins_num;
        const unsigned int infea_ori = in_feat;
        const unsigned int outfea_ori = out_feat;

        const unsigned int insnum_pad =
            (insnum_ori % 8) == 0 ? insnum_ori : insnum_ori + (8 - insnum_ori % 8);
        const unsigned int infea_pad =
            (infea_ori % 8) == 0 ? infea_ori : infea_ori + (8 - infea_ori % 8);
        const unsigned int outfea_pad =
            (outfea_ori % 8) == 0 ? outfea_ori : outfea_ori + (8 - outfea_ori % 8);

        framework::Tensor input_help;
        input_help = ctx.AllocateTmpTensor<float, DeviceContext>({insnum_pad, infea_pad}, dev_ctx);

        framework::Tensor w_help;
        w_help = ctx.AllocateTmpTensor<float, DeviceContext>({infea_pad, outfea_pad}, dev_ctx);

        framework::Tensor bias_help;
        bias_help = ctx.AllocateTmpTensor<float, DeviceContext>({outfea_pad, 1}, dev_ctx);

        framework::Tensor output_help;
        output_help =
            ctx.AllocateTmpTensor<float, DeviceContext>({insnum_pad, outfea_pad}, dev_ctx);

        T scale = static_cast<T>(1.0);
        cast_and_padding_cpu_kernel<T>(insnum_ori, infea_ori, insnum_pad, infea_pad,
                                       input->data<T>(),
                                       input_help.mutable_data<float>(ctx.GetPlace()), scale);
        cast_and_padding_cpu_kernel<T>(infea_ori, outfea_ori, infea_pad, outfea_pad, w->data<T>(),
                                       w_help.mutable_data<float>(ctx.GetPlace()), scale);
        cast_and_padding_cpu_kernel<T>(outfea_ori, 1, outfea_pad, 1, bias->data<T>(),
                                       bias_help.mutable_data<float>(ctx.GetPlace()), scale);
        VLOG(3) << "input dim0=" << input->dims()[0] << ", input dim1=" << input->dims()[1]
                << ", input_help dim0=" << input_help.dims()[0]
                << ", input_help dim1=" << input_help.dims()[1];
        VLOG(3) << "w dim0=" << w->dims()[0] << ", w dim1=" << w->dims()[1]
                << ", w_help dim0=" << w_help.dims()[0] << ", w_help dim1=" << w_help.dims()[1];
        VLOG(3) << "bias dim0=" << bias->dims()[0] << ", bias dim1=" << bias->dims()[1]
                << ", bias_help dim0=" << bias_help.dims()[0]
                << ", bias_help dim1=" << bias_help.dims()[1];

        // begin fc
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        auto blas = phi::funcs::GetBlas<CPUCtx, float>(dev_ctx);
        float alpha = static_cast<float>(input_scale_factor);
        float bias_scale_factor_use = static_cast<float>(bias_scale_factor);
        float beta = static_cast<float>(0.0);

        blas.GEMM(transA, transB, insnum_pad, outfea_pad, infea_pad, alpha,
                  input_help.data<float>(), w_help.data<float>(), beta,
                  output_help.mutable_data<float>(ctx.GetPlace()));

        // begin row add
        vec_mat_row_add_cpu_kernel<float>(insnum_pad, outfea_pad, output_help.data<float>(),
                                          bias_help.data<float>(), bias_scale_factor_use);

        // begin cast and cut
        T scale_factor = static_cast<T>(1 / input_scale_factor);
        cast_and_cut_cpu_kernel<T>(insnum_ori, outfea_ori, insnum_pad, outfea_pad,
                                   output->data<T>(), output_help.data<float>(), scale_factor);

        VLOG(3) << "input_scale_factor=" << input_scale_factor
                << ", bias_scale_factor_use=" << bias_scale_factor_use
                << ", output scale_factor=" << scale_factor;

        VLOG(3) << "output_help dim0=" << output_help.dims()[0]
                << ", output_help dim1=" << output_help.dims()[1]
                << ", output dim0=" << output->dims()[0] << ", output dim1=" << output->dims()[1];
    }
};

template <typename DeviceContext, typename T>
class ScaledFCGradCPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* input = ctx.Input<Tensor>("Input");
        auto* w = ctx.Input<Tensor>("W");
        auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));  // insnum * outfea

        auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
        auto bias_scale_factor = ctx.Attr<float>("bias_scale_factor");
        auto grad_scale_factor = ctx.Attr<float>("grad_scale_factor");

        T bias_scale_factor_use = static_cast<T>(bias_scale_factor);
        float alpha = static_cast<float>(input_scale_factor);
        float beta = static_cast<float>(0.0);

        auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
        auto* dw = ctx.Output<Tensor>(framework::GradVarName("W"));
        auto* db = ctx.Output<Tensor>(framework::GradVarName("Bias"));

        auto input_dims = input->dims();  // ins_num*in_feat
        auto dout_dims = dout->dims();    // ins_num*out_feat
        auto w_dims = w->dims();          // in_feat*out_feat

        auto dout_coln = dout_dims[1];
        auto ins_num = dout_dims[0];

        auto& dev_ctx = ctx.template device_context<CPUCtx>();

        // initialize
        dx->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, dx, 0.0);

        dw->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, dw, 0.0);

        db->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, db, 0.0);

        // get bias grad
        col_sum_mat_cpu_kernel(ins_num, dout_coln, dout->data<T>(), db->data<T>(),
                               bias_scale_factor_use);

        // pad
        const unsigned int insnum_ori = input_dims[0];
        const unsigned int infea_ori = input_dims[1];
        const unsigned int outfea_ori = w_dims[1];

        const unsigned int insnum_pad =
            (insnum_ori % 8) == 0 ? insnum_ori : insnum_ori + (8 - insnum_ori % 8);
        const unsigned int infea_pad =
            (infea_ori % 8) == 0 ? infea_ori : infea_ori + (8 - infea_ori % 8);
        const unsigned int outfea_pad =
            (outfea_ori % 8) == 0 ? outfea_ori : outfea_ori + (8 - outfea_ori % 8);
        VLOG(3) << "input dim0=" << input_dims[0] << ", input dim1=" << input_dims[1]
                << ", dout dim0=" << dout_dims[0] << ", dout dim1=" << dout_dims[1]
                << ", w dim0=" << w_dims[0] << ", w dim1=" << w_dims[1]
                << ", insnum_ori=" << insnum_ori << ", insnum_pad=" << insnum_pad
                << ", infea_ori=" << infea_ori << ", infea_pad=" << infea_pad
                << ", outfea_ori=" << outfea_ori << ", outfea_pad=" << outfea_pad;

        framework::Tensor dx_help;
        dx_help = ctx.AllocateTmpTensor<float, DeviceContext>({insnum_pad, infea_pad}, dev_ctx);

        framework::Tensor dw_help;
        dw_help = ctx.AllocateTmpTensor<float, DeviceContext>({infea_pad, outfea_pad}, dev_ctx);

        framework::Tensor dout_help;
        dout_help = ctx.AllocateTmpTensor<float, DeviceContext>({insnum_pad, outfea_pad}, dev_ctx);

        framework::Tensor input_help;
        input_help = ctx.AllocateTmpTensor<float, DeviceContext>({insnum_pad, infea_pad}, dev_ctx);

        framework::Tensor w_help;
        w_help = ctx.AllocateTmpTensor<float, DeviceContext>({infea_pad, outfea_pad}, dev_ctx);

        T scale = static_cast<T>(1.0);
        cast_and_padding_cpu_kernel<T>(insnum_ori, infea_ori, insnum_pad, infea_pad,
                                       input->data<T>(),
                                       input_help.mutable_data<float>(ctx.GetPlace()), scale);
        cast_and_padding_cpu_kernel<T>(infea_ori, outfea_ori, infea_pad, outfea_pad, w->data<T>(),
                                       w_help.mutable_data<float>(ctx.GetPlace()), scale);
        T dout_grad_scale_factor =
            static_cast<T>(grad_scale_factor) * static_cast<T>(1 / input_scale_factor);
        cast_and_padding_cpu_kernel<T>(
            insnum_ori, outfea_ori, insnum_pad, outfea_pad, dout->data<T>(),
            dout_help.mutable_data<float>(ctx.GetPlace()), dout_grad_scale_factor);

        auto blas = phi::funcs::GetBlas<CPUCtx, float>(dev_ctx);

        // dx = dy * w^T
        blas.GEMM(CblasNoTrans, CblasTrans, insnum_pad, infea_pad, outfea_pad, alpha,
                  dout_help.data<float>(), w_help.data<float>(), beta,
                  dx_help.mutable_data<float>(ctx.GetPlace()));
        // dw = x^T * dy
        blas.GEMM(CblasTrans, CblasNoTrans, infea_pad, outfea_pad, insnum_pad, alpha,
                  input_help.data<float>(), dout_help.data<float>(), beta,
                  dw_help.mutable_data<float>(ctx.GetPlace()));

        // cast dx dw to fp32 and cut
        T scale_factor = static_cast<T>(1.0 / grad_scale_factor);
        cast_and_cut_cpu_kernel<T>(insnum_ori, infea_ori, insnum_pad, infea_pad, dx->data<T>(),
                                   dx_help.data<float>(), scale_factor);
        cast_and_cut_cpu_kernel<T>(infea_ori, outfea_ori, infea_pad, outfea_pad, dw->data<T>(),
                                   dw_help.data<float>(), scale_factor);
        VLOG(3) << "input_scale_factor=" << input_scale_factor
                << ", bias_scale_factor_use=" << bias_scale_factor_use
                << ", dout_grad_scale_factor=" << dout_grad_scale_factor
                << ", dx_grad dw_grad scale_factor=" << scale_factor;
    }
};

class ScaledFCOp : public framework::OperatorWithKernel {
   public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "ScaledFCOp");
        OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "ScaledFCOp");
        OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "ScaledFCOp");
        OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ScaledFCOp");

        auto input_dims = ctx->GetInputDim("Input");
        auto w_dims = ctx->GetInputDim("W");

        int feature_dim = input_dims[1];
        PADDLE_ENFORCE_EQ(
            feature_dim, w_dims[0],
            platform::errors::InvalidArgument("Input.dim[1] and W.dim[0] of ScaledFCOp "
                                              "should be same."));

        auto bias_dims = ctx->GetInputDim("Bias");
        PADDLE_ENFORCE_EQ(
            bias_dims[0], w_dims[1],
            platform::errors::InvalidArgument("Bias.dim[0] should be same as W.dim[1]."));

        ctx->SetOutputDim("Out", {input_dims[0], w_dims[1]});
        ctx->ShareLoD("Input", /*->*/ "Out");
    }

   protected:
    framework::OpKernelType GetExpectedKernelType(
        const framework::ExecutionContext& ctx) const override {
        return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
                                       ctx.device_context());
    }
};

class ScaledFCGradOp : public framework::OperatorWithKernel {
   public:
    using framework::OperatorWithKernel::OperatorWithKernel;

    void InferShape(framework::InferShapeContext* ctx) const override {
        PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                          platform::errors::InvalidArgument("Input should not be null"));
        PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                          platform::errors::InvalidArgument("Input(W) should not be null"));

        ctx->SetOutputDim(framework::GradVarName("Input"), ctx->GetInputDim("Input"));
        ctx->SetOutputDim(framework::GradVarName("W"), ctx->GetInputDim("W"));
        ctx->SetOutputDim(framework::GradVarName("Bias"), ctx->GetInputDim("Bias"));
    }

   protected:
    framework::OpKernelType GetExpectedKernelType(
        const framework::ExecutionContext& ctx) const override {
        return framework::OpKernelType(
            OperatorWithKernel::IndicateVarDataType(ctx, framework::GradVarName("Out")),
            ctx.device_context());
    }
};

class ScaledFCOpMaker : public framework::OpProtoAndCheckerMaker {
   public:
    void Make() override {
        AddInput("Input", "(Tensor) Input tensor of scaled_fc_op operator.");
        AddInput("W", "(Tensor) Input tensor of scaled_fc_op operator.");
        AddInput("Bias", "(Tensor) Input tensor of scaled_fc_op operator.");
        AddAttr<float>("input_scale_factor", "(float) the scale factor of input tensor");
        AddAttr<float>("bias_scale_factor", "(float) the scale factor of bias tensor");
        AddAttr<float>("grad_scale_factor", "(float) the scale factor of gradient tensor");
        AddOutput("Out", "Output tensor of scaled_fc_op operator.");
        AddComment(R"DOC(
ScaledFC Operator.
Notice: It currently supports GPU device.
This Op exists in contrib, which means that it is not shown to the public.
)DOC");
    }
};

template <typename T>
class ScaledFCGradOpMaker : public framework::SingleGradOpMaker<T> {
   public:
    using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

   protected:
    void Apply(GradOpPtr<T> op) const override {
        op->SetType("scaled_fc_grad");

        op->SetInput("Input", this->Input("Input"));
        op->SetInput("W", this->Input("W"));
        op->SetInput("Bias", this->Input("Bias"));
        op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

        op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
        op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
        op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
        op->SetAttrMap(this->Attrs());
    }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
// REGISTER_OPERATOR(scaled_fc, ops::ScaledFCOp, ops::ScaledFCOpMaker);
REGISTER_OPERATOR(scaled_fc,
                  ops::ScaledFCOp,
                  ops::ScaledFCOpMaker,
                  ops::ScaledFCGradOpMaker<paddle::framework::OpDesc>,
                  ops::ScaledFCGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(scaled_fc_grad, ops::ScaledFCGradOp);

REGISTER_OP_CPU_KERNEL(scaled_fc,
                       ops::ScaledFCCPUKernel<CPUCtx, float>,
                       ops::ScaledFCKernel<CPUCtx, double>);

REGISTER_OP_CPU_KERNEL(scaled_fc_grad,
                       ops::ScaledFCGradCPUKernel<CPUCtx, float>,
                       ops::ScaledFCKernel<CPUCtx, double>);
