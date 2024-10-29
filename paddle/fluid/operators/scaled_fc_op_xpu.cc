/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include "paddle/fluid/operators/scaled_fc_op.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/operators/xpu_api_wrapper.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "xpu/refactor/math.h"

using XPUCtx = phi::XPUContext;

namespace paddle {
namespace operators {
using framework::Tensor;

template <typename T>
void cast_and_padding_xpu_kernel(xpu::Context* xpu_ctx,
                                 const int rown_ori,
                                 const int coln_ori,
                                 const int rown_pad,
                                 const int coln_pad,
                                 const T* matrix,
                                 paddle::platform::float16* matrix_pad,
                                 float scale_factor) {
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    auto matrix_ptr = reinterpret_cast<const T*>(matrix);
    auto matrix_scale_ptr = RAII_GUARD.alloc_l3_or_gm<T>(rown_ori * coln_ori);
    auto matrix_cast_ptr = RAII_GUARD.alloc_l3_or_gm<float16>(rown_ori * coln_ori);
    auto matrix_pad_ptr = reinterpret_cast<float16*>(matrix_pad);

    int ret = xpu::scale<T>(xpu_ctx, matrix_ptr, matrix_scale_ptr, rown_ori * coln_ori, true,
                            scale_factor, 0.0);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU scale API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));

    ret = xpu::cast<T, float16>(xpu_ctx, matrix_scale_ptr, matrix_cast_ptr, rown_ori * coln_ori);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        phi::errors::External("XPU cast API return wrong value[%d %s].", ret, XPUAPIErrorMsg[ret]));

    ret = xpu::pad<float16>(xpu_ctx, matrix_cast_ptr, matrix_pad_ptr, {rown_ori, coln_ori}, {0, 0},
                            {rown_pad - rown_ori, coln_pad - coln_ori});
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        phi::errors::External("XPU pad API return wrong value[%d %s].", ret, XPUAPIErrorMsg[ret]));
}


template <typename T>
void vec_mat_row_add_xpu_kernel(xpu::Context* xpu_ctx,
                                const int rown,
                                const int coln,
                                T* matrix,
                                const T* bias,
                                float bias_scale_factor) {
    auto matrix_ptr = reinterpret_cast<float16*>(matrix);
    auto bias_ptr = reinterpret_cast<const float16*>(bias);
    auto matrix_ptr_const = reinterpret_cast<const float16*>(matrix);
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    float16* bias_scaled_ptr = RAII_GUARD.alloc_l3_or_gm<float16>(coln);

    int ret =
        xpu::scale<float16>(xpu_ctx, bias_ptr, bias_scaled_ptr, coln, true, bias_scale_factor, 0);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU scale API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));

    ret = xpu::broadcast_add<float16>(xpu_ctx, matrix_ptr_const, bias_scaled_ptr, matrix_ptr,
                                      {rown, coln}, {1, coln});
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU broadcast_add API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));
}

template <typename T>
void cast_and_cut_xpu_kernel(xpu::Context* xpu_ctx,
                             const int rown_ori,
                             const int coln_ori,
                             const int rown_pad,
                             const int coln_pad,
                             T* matrix,
                             paddle::platform::float16* matrix_pad,
                             float scale_factor) {
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    auto matrix_pad_ptr = reinterpret_cast<const float16*>(matrix_pad);
    auto matrix_slice_ptr = RAII_GUARD.alloc_l3_or_gm<float16>(rown_ori * coln_ori);
    auto matrix_cast_ptr = reinterpret_cast<T*>(matrix);
    auto matrix_ptr = reinterpret_cast<T*>(matrix);

    int ret = xpu::slice<float16>(xpu_ctx, matrix_pad_ptr, matrix_slice_ptr, {rown_pad, coln_pad},
                                  {0, 0}, {rown_ori, coln_ori});
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU slice API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));

    ret = xpu::cast<float16, T>(xpu_ctx, matrix_slice_ptr, matrix_cast_ptr, rown_ori * coln_ori);
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        phi::errors::External("XPU cast API return wrong value[%d %s].", ret, XPUAPIErrorMsg[ret]));

    ret = xpu::scale<T>(xpu_ctx, matrix_cast_ptr, matrix_ptr, rown_ori * coln_ori, true,
                        scale_factor, 0.0);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU scale API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));
}

template <typename T>
void col_sum_mat_xpu_kernel(xpu::Context* xpu_ctx,
                            const int rown,
                            const int coln,
                            const T* matrix,
                            T* matrix_out) {
    auto matrix_ptr = reinterpret_cast<const T*>(matrix);
    auto matrix_out_ptr = reinterpret_cast<T*>(matrix_out);

    int ret = xpu::reduce_sum<T>(xpu_ctx, matrix_ptr, matrix_out_ptr, {rown, coln}, {1});
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU reduce_sum API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));
}

//for fp32
template <typename T>
void padding_xpu_kernel(xpu::Context* xpu_ctx,
                                 const int rown_ori,
                                 const int coln_ori,
                                 const int rown_pad,
                                 const int coln_pad,
                                 const T* matrix,
                                 float* matrix_pad,
                                 float scale_factor) {
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    auto matrix_ptr = reinterpret_cast<const T*>(matrix);
    auto matrix_scale_ptr = RAII_GUARD.alloc_l3_or_gm<T>(rown_ori * coln_ori);
    auto matrix_pad_ptr = reinterpret_cast<float*>(matrix_pad);

    int ret = xpu::scale<T>(xpu_ctx, matrix_ptr, matrix_scale_ptr, rown_ori * coln_ori, true,
                            scale_factor, 0.0);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU scale API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));


    ret = xpu::pad<float>(xpu_ctx, matrix_scale_ptr, matrix_pad_ptr, {rown_ori, coln_ori}, {0, 0},
                            {rown_pad - rown_ori, coln_pad - coln_ori});
    PADDLE_ENFORCE_EQ(
        ret, XPU_SUCCESS,
        phi::errors::External("XPU pad API return wrong value[%d %s].", ret, XPUAPIErrorMsg[ret]));
}

//for fp32
template <typename T>
void vec_mat_row_add_xpu_kernel2(xpu::Context* xpu_ctx,
                                const int rown,
                                const int coln,
                                T* matrix,
                                const T* bias,
                                float bias_scale_factor) {
    auto matrix_ptr = reinterpret_cast<float*>(matrix);
    auto bias_ptr = reinterpret_cast<const float*>(bias);
    auto matrix_ptr_const = reinterpret_cast<const float*>(matrix);
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    float* bias_scaled_ptr = RAII_GUARD.alloc_l3_or_gm<float>(coln);

    int ret =
        xpu::scale<float>(xpu_ctx, bias_ptr, bias_scaled_ptr, coln, true, bias_scale_factor, 0);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU scale API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));

    ret = xpu::broadcast_add<float>(xpu_ctx, matrix_ptr_const, bias_scaled_ptr, matrix_ptr,
                                      {rown, coln}, {1, coln});
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU broadcast_add API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));
}

//for fp32
template <typename T>
void cut_xpu_kernel(xpu::Context* xpu_ctx,
                             const int rown_ori,
                             const int coln_ori,
                             const int rown_pad,
                             const int coln_pad,
                             T* matrix,
                             float* matrix_pad,
                             float scale_factor) {
    xpu::ctx_guard RAII_GUARD(xpu_ctx);
    auto matrix_pad_ptr = reinterpret_cast<const float*>(matrix_pad);
    auto matrix_slice_ptr = RAII_GUARD.alloc_l3_or_gm<float>(rown_ori * coln_ori);
    auto matrix_ptr = reinterpret_cast<T*>(matrix);

    int ret = xpu::slice<float>(xpu_ctx, matrix_pad_ptr, matrix_slice_ptr, {rown_pad, coln_pad},
                                  {0, 0}, {rown_ori, coln_ori});
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU slice API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));

    ret = xpu::scale<T>(xpu_ctx, matrix_slice_ptr, matrix_ptr, rown_ori * coln_ori, true,
                        scale_factor, 0.0);
    PADDLE_ENFORCE_EQ(ret, XPU_SUCCESS,
                      phi::errors::External("XPU scale API return wrong value[%d %s].", ret,
                                            XPUAPIErrorMsg[ret]));
}

// for not cast to fp16
#if 1
template <typename DeviceContext, typename T>
class ScaledFCXPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* input = ctx.Input<framework::LoDTensor>("Input");
        auto* w = ctx.Input<Tensor>("W");
        auto* bias = ctx.Input<Tensor>("Bias");
        auto* output = ctx.Output<framework::LoDTensor>("Out");
        auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
        auto bias_scale_factor = ctx.Attr<float>("bias_scale_factor");

        auto input_dims = input->dims();
        auto w_dims = w->dims();
        auto ins_num = input_dims[0];
        auto in_feat = input_dims[1];
        auto out_feat = w_dims[1];

        output->mutable_data<T>(ctx.GetPlace());
        output->Resize({ins_num, w_dims[1]});

        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        xpu::Context* xpu_ctx = dev_ctx.x_context();

        const int insnum_ori = ins_num;
        const int infea_ori = in_feat;
        const int outfea_ori = out_feat;
        const int insnum_pad =
            (insnum_ori % 8) == 0 ? insnum_ori : insnum_ori + (8 - insnum_ori % 8);
        const int infea_pad = (infea_ori % 8) == 0 ? infea_ori : infea_ori + (8 - infea_ori % 8);
        const int outfea_pad =
            (outfea_ori % 8) == 0 ? outfea_ori : outfea_ori + (8 - outfea_ori % 8);

        framework::Tensor input_help;
        input_help = ctx.AllocateTmpTensor<float, DeviceContext>(
            {insnum_pad, infea_pad}, dev_ctx);

        framework::Tensor w_help;
        w_help = ctx.AllocateTmpTensor<float, DeviceContext>(
            {infea_pad, outfea_pad}, dev_ctx);

        framework::Tensor bias_help;
        bias_help = ctx.AllocateTmpTensor<float, DeviceContext>({outfea_pad, 1},
                                                                                    dev_ctx);

        framework::Tensor output_help;
        output_help = ctx.AllocateTmpTensor<float, DeviceContext>(
            {insnum_pad, outfea_pad}, dev_ctx);

        // Padding.
        T scale = static_cast<T>(1.0);
        padding_xpu_kernel<T>(
            xpu_ctx, insnum_ori, infea_ori, insnum_pad, infea_pad, input->data<T>(),
            input_help.mutable_data<float>(ctx.GetPlace()), scale);
        padding_xpu_kernel<T>(
            xpu_ctx, infea_ori, outfea_ori, infea_pad, outfea_pad, w->data<T>(),
            w_help.mutable_data<float>(ctx.GetPlace()), scale);
        padding_xpu_kernel<T>(
            xpu_ctx, outfea_ori, 1, outfea_pad, 1, bias->data<T>(),
            bias_help.mutable_data<float>(ctx.GetPlace()), scale);
        VLOG(3) << "input dim0=" << input->dims()[0] << ", input dim1=" << input->dims()[1]
                << ", input_help dim0=" << input_help.dims()[0]
                << ", input_help dim1=" << input_help.dims()[1];
        VLOG(3) << "w dim0=" << w->dims()[0] << ", w dim1=" << w->dims()[1]
                << ", w_help dim0=" << w_help.dims()[0] << ", w_help dim1=" << w_help.dims()[1];
        VLOG(3) << "bias dim0=" << bias->dims()[0] << ", bias dim1=" << bias->dims()[1]
                << ", bias_help dim0=" << bias_help.dims()[0]
                << ", bias_help dim1=" << bias_help.dims()[1];

        // Fc begin.
        float alpha = static_cast<float>(input_scale_factor);
        float bias_scale_factor_use = static_cast<float>(bias_scale_factor);

        XpuFcInfo fc_info;
        GetFCInfo(input_help.dims(), w_help.dims(), false, false, &fc_info);
        const float* x_ptr =
            reinterpret_cast<const float*>(input_help.data<float>());
        const float* y_ptr =
            reinterpret_cast<const float*>(w_help.data<float>());
        float* out_ptr =
            reinterpret_cast<float*>(output_help.data<float>());
        MatMulXPUFunction<float>(xpu_ctx, x_ptr, y_ptr, out_ptr, fc_info, alpha);

        vec_mat_row_add_xpu_kernel2<T>(
            xpu_ctx, insnum_pad, outfea_pad, output_help.data<float>(),
            bias_help.data<float>(), bias_scale_factor_use);

        // Cut.
        float scale_factor = static_cast<float>(1 / input_scale_factor);
        cut_xpu_kernel<T>(xpu_ctx, insnum_ori, outfea_ori, insnum_pad, outfea_pad,
                                   output->data<T>(), output_help.data<float>(),
                                   scale_factor);
        VLOG(3) << "input_scale_factor=" << input_scale_factor
                << ", bias_scale_factor_use=" << bias_scale_factor
                << ", output scale_factor=" << scale_factor;

        VLOG(3) << "output_help dim0=" << output_help.dims()[0]
                << ", output_help dim1=" << output_help.dims()[1]
                << ", output dim0=" << output->dims()[0] << ", output dim1=" << output->dims()[1];
    }
};


template <typename DeviceContext, typename T>
class ScaledFCGradXPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* input = ctx.Input<Tensor>("Input");
        auto* w = ctx.Input<Tensor>("W");
        auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));  // insnum * outfea
        auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
        auto grad_scale_factor = ctx.Attr<float>("grad_scale_factor");

        auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
        auto* dw = ctx.Output<Tensor>(framework::GradVarName("W"));
        auto* db = ctx.Output<Tensor>(framework::GradVarName("Bias"));

        auto input_dims = input->dims();  // ins_num*in_feat
        auto dout_dims = dout->dims();    // ins_num*out_feat
        auto w_dims = w->dims();          // in_feat*out_feat

        auto dout_coln = dout_dims[1];
        auto ins_num = dout_dims[0];

        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        auto xpu_ctx = dev_ctx.x_context();

        // Initialize
        dx->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, dx, 0.0);

        dw->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, dw, 0.0);

        db->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, db, 0.0);

        const int insnum_ori = input_dims[0];
        const int infea_ori = input_dims[1];
        const int outfea_ori = w_dims[1];
        const int insnum_pad =
            (insnum_ori % 8) == 0 ? insnum_ori : insnum_ori + (8 - insnum_ori % 8);
        const int infea_pad = (infea_ori % 8) == 0 ? infea_ori : infea_ori + (8 - infea_ori % 8);
        const int outfea_pad =
            (outfea_ori % 8) == 0 ? outfea_ori : outfea_ori + (8 - outfea_ori % 8);
        VLOG(3) << "input dim0=" << input_dims[0] << ", input dim1=" << input_dims[1]
                << ", dout dim0=" << dout_dims[0] << ", dout dim1=" << dout_dims[1]
                << ", w dim0=" << w_dims[0] << ", w dim1=" << w_dims[1]
                << ", insnum_ori=" << insnum_ori << ", insnum_pad=" << insnum_pad
                << ", infea_ori=" << infea_ori << ", infea_pad=" << infea_pad
                << ", outfea_ori=" << outfea_ori << ", outfea_pad=" << outfea_pad;

        framework::Tensor dx_help;
        dx_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {insnum_pad, infea_pad}, dev_ctx);

        framework::Tensor dw_help;
        dw_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {infea_pad, outfea_pad}, dev_ctx);

        framework::Tensor dout_help;
        dout_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {insnum_pad, outfea_pad}, dev_ctx);

        framework::Tensor input_help;
        input_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {insnum_pad, infea_pad}, dev_ctx);

        framework::Tensor w_help;
        w_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {infea_pad, outfea_pad}, dev_ctx);

        // Get bias grad.
        col_sum_mat_xpu_kernel<T>(xpu_ctx, ins_num, dout_coln, dout->data<T>(), db->data<T>());

        // Cast input,w,dout to fp16 and padding.
        T scale = static_cast<T>(1.0);
        cast_and_padding_xpu_kernel<T>(
            xpu_ctx, insnum_ori, infea_ori, insnum_pad, infea_pad, input->data<T>(),
            input_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
        cast_and_padding_xpu_kernel<T>(
            xpu_ctx, infea_ori, outfea_ori, infea_pad, outfea_pad, w->data<T>(),
            w_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
        T dout_grad_scale_factor =
            static_cast<T>(grad_scale_factor) * static_cast<T>(1 / input_scale_factor);
        cast_and_padding_xpu_kernel<T>(
            xpu_ctx, insnum_ori, outfea_ori, insnum_pad, outfea_pad, dout->data<T>(),
            dout_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()),
            dout_grad_scale_factor);

        // Fc begin.
        XpuFcInfo fc_info1, fc_info2;
        float alpha = static_cast<float>(input_scale_factor);
        // dx = dy * w^T
        GetFCInfo(dout_help.dims(), w_help.dims(), false, true, &fc_info1);
        const float16* x1_ptr =
            reinterpret_cast<const float16*>(dout_help.data<paddle::platform::float16>());
        const float16* y1_ptr =
            reinterpret_cast<const float16*>(w_help.data<paddle::platform::float16>());
        float16* out1_ptr = reinterpret_cast<float16*>(dx_help.data<paddle::platform::float16>());
        MatMulXPUFunction<float16>(xpu_ctx, x1_ptr, y1_ptr, out1_ptr, fc_info1, alpha);
        // dw = x^T * dy
        GetFCInfo(input_help.dims(), dout_help.dims(), true, false, &fc_info2);
        const float16* x2_ptr =
            reinterpret_cast<const float16*>(input_help.data<paddle::platform::float16>());
        const float16* y2_ptr =
            reinterpret_cast<const float16*>(dout_help.data<paddle::platform::float16>());
        float16* out2_ptr = reinterpret_cast<float16*>(dw_help.data<paddle::platform::float16>());
        MatMulXPUFunction<float16>(xpu_ctx, x2_ptr, y2_ptr, out2_ptr, fc_info2, alpha);

        // Cast dx,dw to fp32 and cut.
        T scale_factor = static_cast<T>(1.0 / grad_scale_factor);
        cast_and_cut_xpu_kernel<T>(xpu_ctx, insnum_ori, infea_ori, insnum_pad, infea_pad,
                                   dx->data<T>(), dx_help.data<paddle::platform::float16>(),
                                   scale_factor);
        cast_and_cut_xpu_kernel<T>(xpu_ctx, infea_ori, outfea_ori, infea_pad, outfea_pad,
                                   dw->data<T>(), dw_help.data<paddle::platform::float16>(),
                                   scale_factor);
        VLOG(3) << "input_scale_factor=" << input_scale_factor
                << ", dout_grad_scale_factor=" << dout_grad_scale_factor
                << ", dx_grad dw_grad scale_factor=" << scale_factor;
    }
};

#elif
template <typename DeviceContext, typename T>
class ScaledFCXPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* input = ctx.Input<framework::LoDTensor>("Input");
        auto* w = ctx.Input<Tensor>("W");
        auto* bias = ctx.Input<Tensor>("Bias");
        auto* output = ctx.Output<framework::LoDTensor>("Out");
        auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
        auto bias_scale_factor = ctx.Attr<float>("bias_scale_factor");

        auto input_dims = input->dims();
        auto w_dims = w->dims();
        auto ins_num = input_dims[0];
        auto in_feat = input_dims[1];
        auto out_feat = w_dims[1];

        output->mutable_data<T>(ctx.GetPlace());
        output->Resize({ins_num, w_dims[1]});

        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        xpu::Context* xpu_ctx = dev_ctx.x_context();

        const int insnum_ori = ins_num;
        const int infea_ori = in_feat;
        const int outfea_ori = out_feat;
        const int insnum_pad =
            (insnum_ori % 8) == 0 ? insnum_ori : insnum_ori + (8 - insnum_ori % 8);
        const int infea_pad = (infea_ori % 8) == 0 ? infea_ori : infea_ori + (8 - infea_ori % 8);
        const int outfea_pad =
            (outfea_ori % 8) == 0 ? outfea_ori : outfea_ori + (8 - outfea_ori % 8);

        framework::Tensor input_help;
        input_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {insnum_pad, infea_pad}, dev_ctx);

        framework::Tensor w_help;
        w_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {infea_pad, outfea_pad}, dev_ctx);

        framework::Tensor bias_help;
        bias_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>({outfea_pad, 1},
                                                                                    dev_ctx);

        framework::Tensor output_help;
        output_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {insnum_pad, outfea_pad}, dev_ctx);

        // Cast input,w,bias to fp16 and padding.
        T scale = static_cast<T>(1.0);
        cast_and_padding_xpu_kernel<T>(
            xpu_ctx, insnum_ori, infea_ori, insnum_pad, infea_pad, input->data<T>(),
            input_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
        cast_and_padding_xpu_kernel<T>(
            xpu_ctx, infea_ori, outfea_ori, infea_pad, outfea_pad, w->data<T>(),
            w_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
        cast_and_padding_xpu_kernel<T>(
            xpu_ctx, outfea_ori, 1, outfea_pad, 1, bias->data<T>(),
            bias_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
        VLOG(3) << "input dim0=" << input->dims()[0] << ", input dim1=" << input->dims()[1]
                << ", input_help dim0=" << input_help.dims()[0]
                << ", input_help dim1=" << input_help.dims()[1];
        VLOG(3) << "w dim0=" << w->dims()[0] << ", w dim1=" << w->dims()[1]
                << ", w_help dim0=" << w_help.dims()[0] << ", w_help dim1=" << w_help.dims()[1];
        VLOG(3) << "bias dim0=" << bias->dims()[0] << ", bias dim1=" << bias->dims()[1]
                << ", bias_help dim0=" << bias_help.dims()[0]
                << ", bias_help dim1=" << bias_help.dims()[1];

        // Fc begin.
        float alpha = static_cast<float>(input_scale_factor);
        float bias_scale_factor_use = static_cast<float>(bias_scale_factor);

        XpuFcInfo fc_info;
        GetFCInfo(input_help.dims(), w_help.dims(), false, false, &fc_info);
        const float16* x_ptr =
            reinterpret_cast<const float16*>(input_help.data<paddle::platform::float16>());
        const float16* y_ptr =
            reinterpret_cast<const float16*>(w_help.data<paddle::platform::float16>());
        float16* out_ptr =
            reinterpret_cast<float16*>(output_help.data<paddle::platform::float16>());
        MatMulXPUFunction<float16>(xpu_ctx, x_ptr, y_ptr, out_ptr, fc_info, alpha);


        vec_mat_row_add_xpu_kernel<paddle::platform::float16>(
            xpu_ctx, insnum_pad, outfea_pad, output_help.data<paddle::platform::float16>(),
            bias_help.data<paddle::platform::float16>(), bias_scale_factor_use);

        // Cast output to fp32 and cut.
        float scale_factor = static_cast<float>(1 / input_scale_factor);
        cast_and_cut_xpu_kernel<T>(xpu_ctx, insnum_ori, outfea_ori, insnum_pad, outfea_pad,
                                   output->data<T>(), output_help.data<paddle::platform::float16>(),
                                   scale_factor);
        VLOG(3) << "input_scale_factor=" << input_scale_factor
                << ", bias_scale_factor_use=" << bias_scale_factor
                << ", output scale_factor=" << scale_factor;

        VLOG(3) << "output_help dim0=" << output_help.dims()[0]
                << ", output_help dim1=" << output_help.dims()[1]
                << ", output dim0=" << output->dims()[0] << ", output dim1=" << output->dims()[1];
    }
};
template <typename DeviceContext, typename T>
class ScaledFCGradXPUKernel : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        auto* input = ctx.Input<Tensor>("Input");
        auto* w = ctx.Input<Tensor>("W");
        auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));  // insnum * outfea
        auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
        auto grad_scale_factor = ctx.Attr<float>("grad_scale_factor");

        auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
        auto* dw = ctx.Output<Tensor>(framework::GradVarName("W"));
        auto* db = ctx.Output<Tensor>(framework::GradVarName("Bias"));

        auto input_dims = input->dims();  // ins_num*in_feat
        auto dout_dims = dout->dims();    // ins_num*out_feat
        auto w_dims = w->dims();          // in_feat*out_feat

        auto dout_coln = dout_dims[1];
        auto ins_num = dout_dims[0];

        auto& dev_ctx = ctx.template device_context<DeviceContext>();
        auto xpu_ctx = dev_ctx.x_context();

        // Initialize
        dx->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, dx, 0.0);

        dw->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, dw, 0.0);

        db->mutable_data<T>(ctx.GetPlace());
        phi::funcs::set_constant(dev_ctx, db, 0.0);

        const int insnum_ori = input_dims[0];
        const int infea_ori = input_dims[1];
        const int outfea_ori = w_dims[1];
        const int insnum_pad =
            (insnum_ori % 8) == 0 ? insnum_ori : insnum_ori + (8 - insnum_ori % 8);
        const int infea_pad = (infea_ori % 8) == 0 ? infea_ori : infea_ori + (8 - infea_ori % 8);
        const int outfea_pad =
            (outfea_ori % 8) == 0 ? outfea_ori : outfea_ori + (8 - outfea_ori % 8);
        VLOG(3) << "input dim0=" << input_dims[0] << ", input dim1=" << input_dims[1]
                << ", dout dim0=" << dout_dims[0] << ", dout dim1=" << dout_dims[1]
                << ", w dim0=" << w_dims[0] << ", w dim1=" << w_dims[1]
                << ", insnum_ori=" << insnum_ori << ", insnum_pad=" << insnum_pad
                << ", infea_ori=" << infea_ori << ", infea_pad=" << infea_pad
                << ", outfea_ori=" << outfea_ori << ", outfea_pad=" << outfea_pad;

        framework::Tensor dx_help;
        dx_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {insnum_pad, infea_pad}, dev_ctx);

        framework::Tensor dw_help;
        dw_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {infea_pad, outfea_pad}, dev_ctx);

        framework::Tensor dout_help;
        dout_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {insnum_pad, outfea_pad}, dev_ctx);

        framework::Tensor input_help;
        input_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {insnum_pad, infea_pad}, dev_ctx);

        framework::Tensor w_help;
        w_help = ctx.AllocateTmpTensor<paddle::platform::float16, DeviceContext>(
            {infea_pad, outfea_pad}, dev_ctx);

        // Get bias grad.
        col_sum_mat_xpu_kernel(xpu_ctx, ins_num, dout_coln, dout->data<T>(), db->data<T>());

        // Cast input,w,dout to fp16 and padding.
        T scale = static_cast<T>(1.0);
        cast_and_padding_xpu_kernel<T>(
            xpu_ctx, insnum_ori, infea_ori, insnum_pad, infea_pad, input->data<T>(),
            input_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
        cast_and_padding_xpu_kernel<T>(
            xpu_ctx, infea_ori, outfea_ori, infea_pad, outfea_pad, w->data<T>(),
            w_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()), scale);
        T dout_grad_scale_factor =
            static_cast<T>(grad_scale_factor) * static_cast<T>(1 / input_scale_factor);
        cast_and_padding_xpu_kernel<T>(
            xpu_ctx, insnum_ori, outfea_ori, insnum_pad, outfea_pad, dout->data<T>(),
            dout_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()),
            dout_grad_scale_factor);

        // Fc begin.
        XpuFcInfo fc_info1, fc_info2;
        float alpha = static_cast<float>(input_scale_factor);
        // dx = dy * w^T
        GetFCInfo(dout_help.dims(), w_help.dims(), false, true, &fc_info1);
        const float16* x1_ptr =
            reinterpret_cast<const float16*>(dout_help.data<paddle::platform::float16>());
        const float16* y1_ptr =
            reinterpret_cast<const float16*>(w_help.data<paddle::platform::float16>());
        float16* out1_ptr = reinterpret_cast<float16*>(dx_help.data<paddle::platform::float16>());
        MatMulXPUFunction<float16>(xpu_ctx, x1_ptr, y1_ptr, out1_ptr, fc_info1, alpha);
        // dw = x^T * dy
        GetFCInfo(input_help.dims(), dout_help.dims(), true, false, &fc_info2);
        const float16* x2_ptr =
            reinterpret_cast<const float16*>(input_help.data<paddle::platform::float16>());
        const float16* y2_ptr =
            reinterpret_cast<const float16*>(dout_help.data<paddle::platform::float16>());
        float16* out2_ptr = reinterpret_cast<float16*>(dw_help.data<paddle::platform::float16>());
        MatMulXPUFunction<float16>(xpu_ctx, x2_ptr, y2_ptr, out2_ptr, fc_info2, alpha);

        // Cast dx,dw to fp32 and cut.
        T scale_factor = static_cast<T>(1.0 / grad_scale_factor);
        cast_and_cut_xpu_kernel<T>(xpu_ctx, insnum_ori, infea_ori, insnum_pad, infea_pad,
                                   dx->data<T>(), dx_help.data<paddle::platform::float16>(),
                                   scale_factor);
        cast_and_cut_xpu_kernel<T>(xpu_ctx, infea_ori, outfea_ori, infea_pad, outfea_pad,
                                   dw->data<T>(), dw_help.data<paddle::platform::float16>(),
                                   scale_factor);
        VLOG(3) << "input_scale_factor=" << input_scale_factor
                << ", dout_grad_scale_factor=" << dout_grad_scale_factor
                << ", dx_grad dw_grad scale_factor=" << scale_factor;
    }
};
#endif

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(scaled_fc,
                       ops::ScaledFCXPUKernel<paddle::platform::XPUDeviceContext, float>);

REGISTER_OP_XPU_KERNEL(scaled_fc_grad,
                       ops::ScaledFCGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
