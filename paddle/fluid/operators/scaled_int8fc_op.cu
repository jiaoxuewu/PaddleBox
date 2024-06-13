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

#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/scaled_int8fc_op.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

using GPUCtx = phi::GPUContext;
namespace paddle {
namespace operators {
using framework::Tensor;

#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = paddle::platform::PADDLE_CUDA_NUM_THREADS;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// cast to fp16 & padding
template <typename T>
__global__ void kernel_cast_and_padding(const int N, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad,
                      const T* matrix, int8_t* matrix_pad,
                      T grad_scale_factor,
                      T expand_factor,
                      T clip,
                      float int8_range) {
  CUDA_KERNEL_LOOP(i, N) { 
      int col_idx = i % coln_pad;
      int row_idx = i / coln_pad;
      if (row_idx < rown_ori && col_idx < coln_ori) {
          int idx = row_idx * coln_ori + col_idx;
          //matrix_pad[i] = static_cast<paddle::platform::float16>(matrix[idx] * grad_scale_factor);

          // mul expand_factor, mul10.0
          T matrix_idx_expand = matrix[idx] *  expand_factor;
          // range[-2.0, 2.0]
          T matrix_idx_expand_clip = 0.0;
          if ((matrix_idx_expand - 0) >= 1e-6) {
               matrix_idx_expand_clip = (matrix_idx_expand - clip) > 1e-6 ? clip : matrix_idx_expand;
          } else {
               matrix_idx_expand_clip = (matrix_idx_expand - (-clip)) < 1e-6 ? -clip : matrix_idx_expand;
          }
          // quantitate to int8
          T interval = (2*clip) / int8_range;
          matrix_pad[i] = static_cast<int8_t>(matrix_idx_expand_clip / interval + 0.50);
          //matrix_pad[i] = static_cast<int8_t>(matrix_idx_expand_clip / interval);
          //if (i % 1000000 == 0) {
          //  printf("rown_ori=%d, coln_ori=%d, rown_pad=%d, coln_pad=%d, i=%d, idx=%d,  matrix[idx]=%f, matrix_idx_expand=%f, matrix_idx_expand_clip=%f, matrix_pad_i=%d, end.\n", rown_ori, coln_ori, rown_pad, coln_pad, i, idx, matrix[idx], matrix_idx_expand, matrix_idx_expand_clip, static_cast<int>((matrix_pad[i])));
          //}
      } else {
          //matrix_pad[i] = static_cast<paddle::platform::float16>(0.0);
          matrix_pad[i] = static_cast<int8_t>(0);
      }
  }
}

template <typename T>
void cast_and_padding(cudaStream_t stream, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad, 
                      const T* matrix, int8_t* matrix_pad,
                      T grad_scale_factor,
                      T expand_factor,
                      T clip,
                      float int8_range) {
                      //T grad_scale_factor = static_cast<T>(1.0)) {
  int N = rown_pad * coln_pad;
  kernel_cast_and_padding<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown_ori, coln_ori, rown_pad, coln_pad, matrix, matrix_pad, grad_scale_factor, expand_factor, clip, int8_range);
}
// end cast to fp16 & padding


// cast to fp32 & cut
template <typename T>
__global__ void kernel_cast_and_cut(const int N, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad,
                      T* matrix, float* matrix_pad,
                      T scale_factor,
                      T expand_factor,
                      T interval) {
  CUDA_KERNEL_LOOP(i, N) { 
      int col_idx = i % coln_ori;
      int row_idx = i / coln_ori;
      int idx = row_idx * coln_pad + col_idx;

      matrix[i] = static_cast<T>(matrix_pad[idx]);
      matrix[i] /= expand_factor;
      matrix[i] *= interval;
      //if( i % 1000000 == 0) {
      //  printf("rown_ori=%d, coln_ori=%d, rown_pad=%d, coln_pad=%d, i=%d, matrix[i]=%f, idx=%d, fcget_matrix_pad_idx=%f, expand_factor=%f, interval=%f, end.\n", rown_ori, coln_ori, rown_pad, coln_pad, i, matrix[i], idx, matrix_pad[idx], expand_factor, interval);
    //}
  }
}

//cast_and_cut<T>(ctx.cuda_device_context().stream(), insnum_ori, outfea_ori, insnum_pad, outfea_pad, output->data<T>(), output_help.data<int>(), scale_factor, expand_factor, interval);
//cast_and_cut<T>(ctx.cuda_device_context().stream(), insnum_ori, outfea_ori, insnum_pad, outfea_pad, output->data<T>(), output_help.data<float>(), scale_factor, expand_factor, interval);
template <typename T>
void cast_and_cut(cudaStream_t stream, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad, 
                      T* matrix, float* matrix_pad,
                      T scale_factor,
                      T expand_factor,
                      T weight_expand_factor,
                      T interval) {
  int N = rown_ori * coln_ori;
  //T expand_factor_double = expand_factor * expand_factor;
  T expand_factor_double = expand_factor * weight_expand_factor;
  kernel_cast_and_cut<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown_ori, coln_ori, rown_pad, coln_pad, matrix, matrix_pad, scale_factor, expand_factor_double, interval);
}
// end cast to fp32 & cut

// add the same row vector to all matrix rows
template <typename T>
__global__ void kernel_vec_mat_row_add(const int N, const unsigned int rown,
                                       const unsigned int coln, T* matrix,
                                       const T* vector, const T bias_scale_factor_use) {
  //CUDA_KERNEL_LOOP(i, N) { matrix[i] += vector[i % coln] * bias_scale_factor_use; }
  CUDA_KERNEL_LOOP(i, N) { matrix[i] += vector[i % coln]; }
}

template <typename T>
void vec_mat_row_add(cudaStream_t stream, const unsigned int rown,
                     const unsigned int coln, T* matrix, const T* vector, const T bias_scale_factor_use) {
  int N = rown * coln;
  kernel_vec_mat_row_add<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown, coln, matrix, vector, bias_scale_factor_use);
}

// calculate col sum of a mat
template <typename T>
__global__ void kernel_add_col_sum_mat(const unsigned int rown,
                                       const unsigned int coln, const T* matrix,
                                       T* vector, const T bias_scale_factor_use) {
  CUDA_KERNEL_LOOP(i, coln) {
    for (unsigned int j = 0; j < rown; j++) {
      ////vector[i] += matrix[i * rown + j];
      //vector[i] += matrix[j * coln + i] * bias_scale_factor_use;
      //printf("i=%d, j=%d, rown=%d, coln=%d, matrix[j*coln+i]=%f, vector[i]=%f\n", i, j, rown, coln, matrix[j*coln+i], vector[i]);
      vector[i] += matrix[j * coln + i];
    }
  }
}

//col_sum_mat(stream, ins_num, dout_coln, dout->data<T>(), db->data<T>(), bias_scale_factor_use);
template <typename T>
void col_sum_mat(cudaStream_t stream, const unsigned int rown,
                 const unsigned int coln, const T* matrix, T* vector,
                 const T bias_scale_factor_use) {
  kernel_add_col_sum_mat<<<GET_BLOCKS(coln), CUDA_NUM_THREADS, 0, stream>>>(
      rown, coln, matrix, vector, bias_scale_factor_use);
}
////// for grad /////
// cast to fp16 & padding
template <typename T>
__global__ void kernel_cast_and_padding_grad(const int N, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad,
                      const T* matrix, paddle::platform::float16* matrix_pad,
                      T grad_scale_factor) {
  CUDA_KERNEL_LOOP(i, N) { 
      int col_idx = i % coln_pad;
      int row_idx = i / coln_pad;
      if (row_idx < rown_ori && col_idx < coln_ori) {
          int idx = row_idx * coln_ori + col_idx;
          //matrix_pad[i] = static_cast<paddle::platform::float16>(matrix[idx]);
          matrix_pad[i] = static_cast<paddle::platform::float16>(matrix[idx] * grad_scale_factor);
      } else {
          matrix_pad[i] = static_cast<paddle::platform::float16>(0.0);
      }
  }
}

template <typename T>
void cast_and_padding_grad(cudaStream_t stream, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad, 
                      const T* matrix, paddle::platform::float16* matrix_pad,
                      T grad_scale_factor) {
                      //T grad_scale_factor = static_cast<T>(1.0)) {
  int N = rown_pad * coln_pad;
  kernel_cast_and_padding_grad<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown_ori, coln_ori, rown_pad, coln_pad, matrix, matrix_pad, grad_scale_factor);
}
// end cast to fp16 & padding


// cast to fp32 & cut
template <typename T>
__global__ void kernel_cast_and_cut_grad(const int N, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad,
                      T* matrix, paddle::platform::float16* matrix_pad,
                      T scale_factor) {
  CUDA_KERNEL_LOOP(i, N) { 
      int col_idx = i % coln_ori;
      int row_idx = i / coln_ori;
      int idx = row_idx * coln_pad + col_idx;
      matrix[i] = static_cast<T>(matrix_pad[idx]);
      matrix[i] *= scale_factor;
  }
}

template <typename T>
void cast_and_cut_grad(cudaStream_t stream, 
                      const unsigned int rown_ori, const unsigned int coln_ori,
                      const unsigned int rown_pad, const unsigned int coln_pad, 
                      T* matrix, paddle::platform::float16* matrix_pad,
                      T scale_factor) {
  int N = rown_ori * coln_ori;
  kernel_cast_and_cut_grad<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown_ori, coln_ori, rown_pad, coln_pad, matrix, matrix_pad, scale_factor);
}
// end cast to fp32 & cut

// add the same row vector to all matrix rows
template <typename T>
__global__ void kernel_vec_mat_row_add_grad(const int N, const unsigned int rown,
                                       const unsigned int coln, T* matrix,
                                       const T* vector, const T bias_scale_factor_use) {
  CUDA_KERNEL_LOOP(i, N) { matrix[i] += vector[i % coln] * bias_scale_factor_use; }
}

template <typename T>
void vec_mat_row_add_grad(cudaStream_t stream, const unsigned int rown,
                     const unsigned int coln, T* matrix, const T* vector, const T bias_scale_factor_use) {
  int N = rown * coln;
  kernel_vec_mat_row_add_grad<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown, coln, matrix, vector, bias_scale_factor_use);
}

// calculate col sum of a mat
template <typename T>
__global__ void kernel_add_col_sum_mat_grad(const unsigned int rown,
                                       const unsigned int coln, const T* matrix,
                                       T* vector, const T bias_scale_factor_use) {
  CUDA_KERNEL_LOOP(i, coln) {
    for (unsigned int j = 0; j < rown; j++) {
      ////vector[i] += matrix[i * rown + j];
      //vector[i] += matrix[j * coln + i] * bias_scale_factor_use;
      vector[i] += matrix[j * coln + i];
    }
  }
}

//col_sum_mat(stream, ins_num, dout_coln, dout->data<T>(), db->data<T>(), bias_scale_factor_use);
template <typename T>
void col_sum_mat_grad(cudaStream_t stream, const unsigned int rown,
                 const unsigned int coln, const T* matrix, T* vector,
                 const T bias_scale_factor_use) {
  kernel_add_col_sum_mat_grad<<<GET_BLOCKS(coln), CUDA_NUM_THREADS, 0, stream>>>(
      rown, coln, matrix, vector, bias_scale_factor_use);
}

template <typename DeviceContext, typename T>
class ScaledINT8FCCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    //VLOG(0) << "begin compute.";
    auto* input = ctx.Input<framework::LoDTensor>("Input"); // framework::Tensor*
    auto* w = ctx.Input<Tensor>("W");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* output = ctx.Output<framework::LoDTensor>("Out");
    auto input_scale_factor = ctx.Attr<float>("input_scale_factor");
    auto bias_scale_factor = ctx.Attr<float>("bias_scale_factor");
    
    auto clip_factor = ctx.Attr<float>("clip_factor");
    auto expand_factor = ctx.Attr<float>("expand_factor");

    auto weight_clip_factor = ctx.Attr<float>("weight_clip_factor");
    auto weight_expand_factor = ctx.Attr<float>("weight_expand_factor");
    
    auto int8_range = ctx.Attr<float>("int8_range");

    auto input_dims = input->dims();
    auto w_dims = w->dims();
    auto ins_num = input_dims[0];  // oriinput: ins_num*in_feat, oriweight: in_feat* out_fea, output: ins_num* out_feat
    auto in_feat = input_dims[1];
    auto out_feat = w_dims[1];

    // get data ptr
    const T* in_data = input->data<T>();
    const T* w_data = w->data<T>();
    const T* bias_data = bias->data<T>();

    output->mutable_data<T>(ctx.GetPlace());
    output->Resize({ins_num, w_dims[1]});

    auto& dev_ctx = ctx.template device_context<GPUCtx>();
    // cast and pad
    const unsigned int insnum_ori = ins_num;
    const unsigned int infea_ori = in_feat;
    const unsigned int outfea_ori = out_feat;
    
    const unsigned int insnum_pad = (insnum_ori % 8) == 0 ? insnum_ori : insnum_ori + (8 - insnum_ori % 8);
    const unsigned int infea_pad = (infea_ori % 8) == 0 ? infea_ori : infea_ori + (8 - infea_ori % 8);
    const unsigned int outfea_pad = (outfea_ori % 8) == 0 ? outfea_ori : outfea_ori + (8 - outfea_ori % 8);

    framework::Tensor input_help;
    input_help = ctx.AllocateTmpTensor<int8_t, DeviceContext>({insnum_pad, infea_pad}, dev_ctx);

    framework::Tensor w_help;
    w_help = ctx.AllocateTmpTensor<int8_t, DeviceContext>({infea_pad, outfea_pad}, dev_ctx);

    framework::Tensor bias_help;
    bias_help = ctx.AllocateTmpTensor<int8_t, DeviceContext>({outfea_pad, 1}, dev_ctx);

    framework::Tensor output_help;
    output_help = ctx.AllocateTmpTensor<float, DeviceContext>({insnum_pad, outfea_pad}, dev_ctx);

    T scale = static_cast<T>(1.0);
    expand_factor = static_cast<T>(expand_factor);
    clip_factor = static_cast<T>(clip_factor);

    weight_expand_factor = static_cast<T>(weight_expand_factor);
    weight_clip_factor = static_cast<T>(weight_clip_factor);

    VLOG(3) << "clip_factor=" << clip_factor << ", expand_factor=" << expand_factor;
    VLOG(3) << "weight_clip_factor=" << weight_clip_factor << ", weight_expand_factor=" << weight_expand_factor;

    cast_and_padding<T>(ctx.cuda_device_context().stream(), insnum_ori, infea_ori, insnum_pad, infea_pad, input->data<T>(), input_help.mutable_data<int8_t>(ctx.GetPlace()), scale, expand_factor, clip_factor, int8_range);
    cast_and_padding<T>(ctx.cuda_device_context().stream(), infea_ori, outfea_ori, infea_pad, outfea_pad, w->data<T>(), w_help.mutable_data<int8_t>(ctx.GetPlace()), scale, weight_expand_factor, weight_clip_factor, int8_range);
    //cast_and_padding<T>(ctx.cuda_device_context().stream(), outfea_ori, 1, outfea_pad, 1, bias->data<T>(), bias_help.mutable_data<int8_t>(ctx.GetPlace()), scale, expand_factor, clip);

    VLOG(3) << "input dim0=" << input->dims()[0] << ", input dim1=" << input->dims()[1]
            << ", input_help dim0=" << input_help.dims()[0] << ", input_help dim1=" << input_help.dims()[1];
    VLOG(3) << "w dim0=" << w->dims()[0] << ", w dim1=" << w->dims()[1]
            << ", w_help dim0=" << w_help.dims()[0] << ", w_help dim1=" << w_help.dims()[1];
    VLOG(3) << "bias dim0=" << bias->dims()[0] << ", bias dim1=" << bias->dims()[1]
            << ", bias_help dim0=" << bias_help.dims()[0] << ", bias_help dim1=" << bias_help.dims()[1];

    // end cast and pad

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    auto blas = phi::funcs::GetBlas<GPUCtx, int8_t>(dev_ctx);

    T bias_scale_factor_use = static_cast<T>(1.0);
    float alpha = static_cast<float>(1);
    float beta = static_cast<float>(0);

    //blas.GEMM(transA, transB, insnum_pad, outfea_pad, infea_pad, alpha, input_help.data<paddle::platform::float16>(), w_help.data<paddle::platform::float16>(), beta, output_help.mutable_data<paddle::platform::float16>(ctx.GetPlace()));
    int flag = 0;
    blas.GEMM(transA, transB, insnum_pad, outfea_pad, infea_pad, alpha, input_help.data<int8_t>(), w_help.data<int8_t>(), beta, output_help.mutable_data<float>(ctx.GetPlace()), flag);

    T scale_factor = static_cast<T>(1.0);
    T interval = static_cast<T>(2*clip_factor/int8_range);
    //T interval = (static_cast<T>(2*clip_factor/int8_range)) * (static_cast<T>(2*weight_clip_factor/int8_range)) ;
    VLOG(3) << "interval=" << interval;
    cast_and_cut<T>(ctx.cuda_device_context().stream(), insnum_ori, outfea_ori, insnum_pad, outfea_pad, output->data<T>(), output_help.data<float>(), scale_factor, expand_factor, weight_expand_factor, interval);

    //add bias
    vec_mat_row_add<T>(ctx.cuda_device_context().stream(), insnum_ori, outfea_ori,
                       output->data<T>(), bias->data<T>(), bias_scale_factor_use);

  }
};

template <typename DeviceContext, typename T>
class ScaledINT8FCGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto* w = ctx.Input<Tensor>("W");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out")); // insnum * outfea

    T bias_scale_factor_use = static_cast<T>(1.0);
    T alpha = static_cast<T>(1.0);
    T beta = static_cast<T>(0.0);

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto* dw = ctx.Output<Tensor>(framework::GradVarName("W"));
    auto* db = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto input_dims = input->dims(); //ins_num*in_feat
    auto dout_dims = dout->dims(); //ins_num*out_feat
    auto w_dims = w->dims(); //in_feat*out_feat

    auto& dev_ctx = ctx.template device_context<GPUCtx>();
    auto stream = ctx.cuda_device_context().stream();

    //init
    dx->mutable_data<T>(ctx.GetPlace());
    phi::funcs::set_constant(dev_ctx, dx, 0.0);

    dw->mutable_data<T>(ctx.GetPlace());
    phi::funcs::set_constant(dev_ctx, dw, 0.0);

    db->mutable_data<T>(ctx.GetPlace());
    phi::funcs::set_constant(dev_ctx, db, 0.0);

    // get bias grad
    auto dout_coln = dout_dims[1];
    auto ins_num = dout_dims[0];
    col_sum_mat(stream, ins_num, dout_coln, dout->data<T>(), db->data<T>(), bias_scale_factor_use);
    // dx dw
    const unsigned int insnum_ori = input_dims[0];
    const unsigned int infea_ori = input_dims[1];
    const unsigned int outfea_ori = w_dims[1];
    
    auto blas = phi::funcs::GetBlas<GPUCtx, T>(dev_ctx);
    //dx = dy * w^T
    blas.GEMM(CblasNoTrans, CblasTrans, insnum_ori, infea_ori, outfea_ori, alpha, dout->data<T>(), w->data<T>(), beta, dx->mutable_data<T>(ctx.GetPlace()));
    //dw = x^T * dy
    blas.GEMM(CblasTrans, CblasNoTrans, infea_ori, outfea_ori, insnum_ori, alpha, input->data<T>(), dout->data<T>(), beta, dw->mutable_data<T>(ctx.GetPlace()));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(scaled_int8fc, ops::ScaledINT8FCCUDAKernel<GPUCtx, float>,
                        ops::ScaledINT8FCCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(scaled_int8fc_grad,
                        ops::ScaledINT8FCGradOpCUDAKernel<GPUCtx, float>,
                        ops::ScaledINT8FCGradOpCUDAKernel<GPUCtx, double>);                      
