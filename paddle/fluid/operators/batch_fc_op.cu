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
#include "paddle/fluid/operators/batch_fc_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
using framework::Tensor;

const int CUDA_NUM_THREADS = 1024;
static inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename T>
__global__ void add_bias_kernel(
    T* data, int slot_pairs_num, int ins_num, int out_dim, const T* bias) {
  CUDA_KERNEL_LOOP(idx, slot_pairs_num * ins_num * out_dim) {
    int block_len = ins_num * out_dim;
    int slot_index = idx / block_len;
    int out_dim_index = (idx % block_len) % out_dim;
    T temp = data[idx] + bias[slot_index * out_dim + out_dim_index];
    data[idx] = temp;
  }
}

template <typename T>
void add_bias(gpuStream_t stream,
              T* data,
              int slot_pairs_num,
              int ins_num,
              int out_dim,
              const T* bias) {
  add_bias_kernel<<<GET_BLOCKS(slot_pairs_num * ins_num * out_dim),
                    CUDA_NUM_THREADS,
                    0,
                    stream>>>(data, slot_pairs_num, ins_num, out_dim, bias);
}

template <typename T>
__global__ void add_bias_grad_kernel(const T* dout_data,
                                     int slot_pairs_num,
                                     int ins_num,
                                     int out_dim,
                                     T* db_data) {
  CUDA_KERNEL_LOOP(idx, slot_pairs_num * out_dim) {
    int row = idx / out_dim;
    int col = idx % out_dim;
    T temp = static_cast<T>(0);
    for (int i = 0; i < ins_num; ++i) {
      int select_indx = ((row + 1) * i + 1) * col;
      temp += dout_data[select_indx];
    }
    db_data[idx] += temp;
  }
}

template <typename T>
void add_bias_grad(gpuStream_t stream,
                   const T* dout_data,
                   int slot_pairs_num,
                   int ins_num,
                   int out_dim,
                   T* db_data) {
  add_bias_grad_kernel<<<GET_BLOCKS(slot_pairs_num * out_dim),
                         CUDA_NUM_THREADS,
                         0,
                         stream>>>(
      dout_data, slot_pairs_num, ins_num, out_dim, db_data);
}

// add the same row vector to all matrix rows
template <typename T>
__global__ void kernel_vec_mat_row_add(const int N, const unsigned int rown,
                                       const unsigned int coln, T* matrix,
                                       const T* vector) {
  CUDA_KERNEL_LOOP(i, N) { matrix[i] += vector[i % coln]; }
}

template <typename T>
void vec_mat_row_add(cudaStream_t stream, const unsigned int rown,
                     const unsigned int coln, T* matrix, const T* vector) {
  int N = rown * coln;
  kernel_vec_mat_row_add<<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(
      N, rown, coln, matrix, vector);
}

// calculate col sum of a mat
template <typename T>
__global__ void kernel_add_col_sum_mat(const unsigned int rown,
                                       const unsigned int coln, const T* matrix,
                                       T* vector) {
  CUDA_KERNEL_LOOP(i, coln) {
    for (unsigned int j = 0; j < rown; j++) {
      // vector[i] += matrix[i * rown + j];
      vector[i] += matrix[j * coln + i];
    }
  }
}

template <typename T>
void col_sum_mat(cudaStream_t stream, const unsigned int rown,
                 const unsigned int coln, const T* matrix, T* vector,
                 const int alpha) {
  kernel_add_col_sum_mat<<<GET_BLOCKS(coln), CUDA_NUM_THREADS, 0, stream>>>(
      rown, coln, matrix, vector);
}

template <typename T>
__global__ void kernel_transpose_split_col(const unsigned int row,
                                           const unsigned int col,
                                           const unsigned int num_block,
                                           const T* source, T* dest) {
  CUDA_KERNEL_LOOP(i, row * col) {
    int len = col / num_block;
    int dest_row = i / len;
    int dest_col = i % len;
    int block_row = dest_row % row;
    int block_idx = dest_row / row;
    int sou_col = block_idx * len + dest_col;
    dest[i] = source[block_row * col + sou_col];
  }
}

template <typename T>
void transpose_split_col(cudaStream_t stream, const unsigned int rown,
                         const unsigned int coln, const unsigned int num_block,
                         const T* source, T* dest) {
  kernel_transpose_split_col<<<GET_BLOCKS(rown * coln), CUDA_NUM_THREADS, 0,
                               stream>>>(rown, coln, num_block, source, dest);
}

template <typename T>
__global__ void kernel_transpose_split_row(const unsigned int row,
                                           const unsigned int col,
                                           const unsigned int num_block,
                                           const T* source, T* dest) {
  CUDA_KERNEL_LOOP(i, row * col) {
    int len = row / num_block;
    int dest_row = i / (col * num_block);
    int dest_col = i % (col * num_block);
    int block_idx = dest_col / col;
    int block_col = dest_col % col;
    dest[i] = source[(block_idx * len + dest_row) * col + block_col];
  }
}

template <typename T>
void transpose_split_row(cudaStream_t stream, const unsigned int rown,
                         const unsigned int coln, const unsigned int num_block,
                         const T* source, T* dest) {
  kernel_transpose_split_row<<<GET_BLOCKS(rown * coln), CUDA_NUM_THREADS, 0,
                               stream>>>(rown, coln, num_block, source, dest);
}

template <typename T>
__global__ void transpose_weight_kernel(const T* source, T* dest,
                      const unsigned int rown, const unsigned int coln, const int64_t batch_count) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < rown && y < coln) {
    int dst_coln = coln / batch_count;
    int dst_x =  x + y / dst_coln * rown;
    int dst_y =  y % dst_coln;
    dest[dst_x * dst_coln + dst_y] = source[x * coln + y];
  }
}

template <typename T>
void transpose_weight_impl(cudaStream_t stream, const T* source, T* dest,
                      const unsigned int rown, const unsigned int coln, const int64_t batch_count) {
  dim3 grid((rown + 15) / 16, (coln + 15) / 16);
  dim3 block(16, 16);
  transpose_weight_kernel<<<grid, block, 0, stream>>>(source, dest, rown, coln, batch_count);
}

template <typename DeviceContext, typename T>
class BatchFCCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int batchcount = ctx.Attr<int>("batchcount");
    auto transpose_weight = ctx.Attr<bool>("transpose_weight");
    if (transpose_weight) {
      // Input_dim: [batch_count, ?, in_dim]
      // W_dim: [in_dim, batch_count * out_dim]
      // Bias_dim: [1, batch_count * out_dim]
      // Out_dim: [batch_count, ?, out_dim]
      auto* input = ctx.Input<framework::LoDTensor>("Input");
      auto* w = ctx.Input<Tensor>("W");
      auto* bias = ctx.Input<Tensor>("Bias");
      auto* output = ctx.Output<framework::LoDTensor>("Out");
      auto input_dims = input->dims();
      auto w_dims = w->dims();
      auto slot_pairs_num = input_dims[0];
      auto ins_num = input_dims[1];
      auto in_dim = input_dims[2];
      auto out_dim = w_dims[1] / batchcount;
  
      // get data ptr
      const T* in_data = input->data<T>();
      const T* w_data = w->data<T>();
      const T* bias_data = bias->data<T>();
  
      output->Resize({slot_pairs_num, ins_num, out_dim});
      T* out_data = output->mutable_data<T>(ctx.GetPlace());

      auto& dev_ctx = ctx.template device_context<phi::GPUContext>();

      Tensor w_help;
      w_help =
          ctx.AllocateTmpTensor<T, DeviceContext>({batchcount, w_dims[0], w_dims[1] / batchcount}, dev_ctx);
      T* w_help_data = w_help.data<T>();

      transpose_weight_impl<T>(ctx.cuda_device_context().stream(), w_data, w_help_data, w_dims[0], w_dims[1], batchcount);

      CBLAS_TRANSPOSE transA = CblasNoTrans;
      CBLAS_TRANSPOSE transB = CblasNoTrans;
  
      T alpha = 1;
      T beta = 0;
      int64_t strideA = ins_num * in_dim;
      int64_t strideB = in_dim * out_dim;

      auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
      blas.BatchedGEMM(transA,
                       transB,
                       ins_num,
                       out_dim,
                       in_dim,
                       alpha,
                       in_data,
                       w_help_data,
                       beta,
                       out_data,
                       slot_pairs_num,
                       strideA,
                       strideB);
      add_bias<T>(ctx.cuda_device_context().stream(),
                  out_data,
                  slot_pairs_num,
                  ins_num,
                  out_dim,
                  bias_data);
      return;
    }
    if (batchcount > 0) {
      auto* input = ctx.Input<framework::LoDTensor>("Input");
      auto* w = ctx.Input<Tensor>("W");
      auto* bias = ctx.Input<Tensor>("Bias");
      auto* output = ctx.Output<framework::LoDTensor>("Out");

      auto input_dims = input->dims();
      auto w_dims = w->dims();
      auto ins_num = input_dims[0];
      auto in_feat = input_dims[1] / batchcount;
      auto out_feat = w_dims[1] / batchcount;

      // get data ptr
      const T* in_data = input->data<T>();
      const T* w_data = w->data<T>();
      const T* bias_data = bias->data<T>();

      output->Resize({ins_num, w_dims[1]});
      T* out_data = output->mutable_data<T>(ctx.GetPlace());

      // initialize
      auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
      phi::funcs::Transpose<DeviceContext, T, 2> trans;

      Tensor out_help;
      out_help =
          ctx.AllocateTmpTensor<T, DeviceContext>({w_dims[1], ins_num}, dev_ctx);
      phi::funcs::set_constant(dev_ctx, &out_help, static_cast<T>(0));

      Tensor input_help;
      input_help = ctx.AllocateTmpTensor<T, DeviceContext>(
          {input_dims[1], ins_num}, dev_ctx);
      trans(dev_ctx, *input, &input_help, {1, 0});

      Tensor w_help;
      w_help = ctx.AllocateTmpTensor<T, DeviceContext>({w_dims[1], w_dims[0]},
                                                       dev_ctx);
      trans(dev_ctx, *w, &w_help, {1, 0});

      CBLAS_TRANSPOSE transA = CblasNoTrans;
      CBLAS_TRANSPOSE transB = CblasNoTrans;

      T alpha = 1;
      T beta = 0;
      int64_t strideA = out_feat * in_feat;
      int64_t strideB = in_feat * ins_num;

      auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
      blas.BatchedGEMM(transA, transB, out_feat, ins_num, in_feat, alpha,
                       w_help.data<T>(), input_help.data<T>(), beta,
                       out_help.data<T>(), batchcount, strideA, strideB);

      trans(dev_ctx, out_help, output, {1, 0});
      vec_mat_row_add<T>(ctx.cuda_device_context().stream(), ins_num, w_dims[1],
                         output->data<T>(), bias->data<T>());
    } else {
      // X.dim = slot_pairs_num * ins_num * in_dim
      // W.dim = slot_pairs_num * in_dim * out_dim
      // b.dim = slot_pairs_num * out_dim
      // output.dim = slot_pairs_num * ins_num * out_dim
      auto* input = ctx.Input<framework::LoDTensor>("Input");
      auto* w = ctx.Input<Tensor>("W");
      auto* bias = ctx.Input<Tensor>("Bias");
      auto* output = ctx.Output<framework::LoDTensor>("Out");
      auto input_dims = input->dims();
      auto w_dims = w->dims();
      auto slot_pairs_num = input_dims[0];
      auto ins_num = input_dims[1];
      auto in_dim = input_dims[2];
      auto out_dim = w_dims[2];
  
      // get data ptr
      const T* in_data = input->data<T>();
      const T* w_data = w->data<T>();
      const T* bias_data = bias->data<T>();
  
      output->Resize({slot_pairs_num, ins_num, out_dim});
      T* out_data = output->mutable_data<T>(ctx.GetPlace());
      // initialize
      auto out_eigen = framework::EigenVector<T>::Flatten(*output);
      auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
      auto& place =
          *ctx.template device_context<phi::GPUContext>().eigen_device();
      out_eigen.device(place) = out_eigen.constant(static_cast<T>(0));
  
      CBLAS_TRANSPOSE transA = CblasNoTrans;
      CBLAS_TRANSPOSE transB = CblasNoTrans;
  
      T alpha = 1;
      T beta = 0;
      int64_t strideA = ins_num * in_dim;
      int64_t strideB = in_dim * out_dim;
  
      auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
      blas.BatchedGEMM(transA,
                       transB,
                       ins_num,
                       out_dim,
                       in_dim,
                       alpha,
                       in_data,
                       w_data,
                       beta,
                       out_data,
                       slot_pairs_num,
                       strideA,
                       strideB);
      add_bias<T>(ctx.cuda_device_context().stream(),
                  out_data,
                  slot_pairs_num,
                  ins_num,
                  out_dim,
                  bias_data);
    }
  }
};

template <typename DeviceContext, typename T>
class BatchFCGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int batchcount = ctx.Attr<int>("batchcount");
    if (batchcount > 0) {
      auto* input = ctx.Input<Tensor>("Input");
      auto* w = ctx.Input<Tensor>("W");
      auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));

      auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
      auto* dw = ctx.Output<Tensor>(framework::GradVarName("W"));
      auto* db = ctx.Output<Tensor>(framework::GradVarName("Bias"));

      auto input_dims = input->dims();
      auto dout_dims = dout->dims();
      auto w_dims = w->dims();

      auto dout_coln = dout_dims[1];
      auto ins_num = dout_dims[0];

      auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
      auto stream = ctx.cuda_device_context().stream();
      // initialize
      dx->mutable_data<T>(ctx.GetPlace());
      phi::funcs::set_constant(dev_ctx, dx, static_cast<T>(0));

      dw->mutable_data<T>(ctx.GetPlace());
      phi::funcs::set_constant(dev_ctx, dw, static_cast<T>(0));

      db->mutable_data<T>(ctx.GetPlace());
      phi::funcs::set_constant(dev_ctx, db, static_cast<T>(0));

      // get bias grad
      col_sum_mat(stream, ins_num, dout_coln, dout->data<T>(), db->data<T>(), 1);

      Tensor dout_help;
      dout_help = ctx.AllocateTmpTensor<T, DeviceContext>(
          {dout_dims[0] * batchcount, dout_dims[1] / batchcount}, dev_ctx);
      dout_help.mutable_data<T>(ctx.GetPlace());

      Tensor input_help;
      input_help = ctx.AllocateTmpTensor<T, DeviceContext>(
          {input_dims[0] * batchcount, input_dims[1] / batchcount}, dev_ctx);
      input_help.mutable_data<T>(ctx.GetPlace());

      Tensor w_help;
      w_help = ctx.AllocateTmpTensor<T, DeviceContext>(
          {w_dims[0] * batchcount, w_dims[1] / batchcount}, dev_ctx);
      w_help.mutable_data<T>(ctx.GetPlace());

      Tensor dx_help;
      dx_help = ctx.AllocateTmpTensor<T, DeviceContext>(
          {input_dims[0] * batchcount, input_dims[1] / batchcount}, dev_ctx);
      phi::funcs::set_constant(dev_ctx, &dx_help, static_cast<T>(0));

      Tensor dw_help;
      dw_help = ctx.AllocateTmpTensor<T, DeviceContext>(
          {w_dims[0] * batchcount, w_dims[1] / batchcount}, dev_ctx);
      phi::funcs::set_constant(dev_ctx, &dw_help, static_cast<T>(0));

      transpose_split_col(stream, dout_dims[0], dout_dims[1], batchcount,
                          dout->data<T>(), dout_help.data<T>());
      transpose_split_col(stream, input_dims[0], input_dims[1], batchcount,
                          input->data<T>(), input_help.data<T>());
      transpose_split_col(stream, w_dims[0], w_dims[1], batchcount, w->data<T>(),
                          w_help.data<T>());

      // dx = dout_data * y^T
      auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
      T alpha = 1;
      T beta = 0;

      // dx = dout_data * y^T
      blas.BatchedGEMM(CblasNoTrans, CblasTrans, dout_dims[0], w_dims[0],
                       dout_dims[1] / batchcount, alpha, dout_help.data<T>(),
                       w_help.data<T>(), beta, dx_help.data<T>(), batchcount,
                       dout_dims[0] * dout_dims[1] / batchcount,
                       w_dims[0] * dout_dims[1] / batchcount);

      transpose_split_row(stream, dout_dims[0] * batchcount, w_dims[0],
                          batchcount, dx_help.data<T>(), dx->data<T>());

      // dy = x^T * dout_data
      blas.BatchedGEMM(CblasTrans, CblasNoTrans, input_dims[1] / batchcount,
                       dout_dims[1] / batchcount, input_dims[0], alpha,
                       input_help.data<T>(), dout_help.data<T>(), beta,
                       dw_help.data<T>(), batchcount,
                       input_dims[0] * input_dims[1] / batchcount,
                       input_dims[0] * dout_dims[1] / batchcount);

      transpose_split_row(stream, w_dims[0] * batchcount, w_dims[1] / batchcount,
                          batchcount, dw_help.data<T>(), dw->data<T>());
    } else {
      auto* input = ctx.Input<Tensor>("Input");
      auto* w = ctx.Input<Tensor>("W");
      auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
  
      auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));
      auto* dw = ctx.Output<Tensor>(framework::GradVarName("W"));
      auto* db = ctx.Output<Tensor>(framework::GradVarName("Bias"));
  
      auto input_dims = input->dims();
      auto w_dims = w->dims();
      auto slot_pairs_num = input_dims[0];
      auto ins_num = input_dims[1];
      auto in_dim = input_dims[2];
      auto out_dim = w_dims[2];
  
      auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
      auto& place =
          *ctx.template device_context<phi::GPUContext>().eigen_device();
      // initialize
      dx->mutable_data<T>(ctx.GetPlace());
      auto dx_eigen = framework::EigenVector<T>::Flatten(*dx);
      dx_eigen.device(place) = dx_eigen.constant(static_cast<T>(0));
  
      dw->mutable_data<T>(ctx.GetPlace());
      auto dw_eigen = framework::EigenVector<T>::Flatten(*dw);
      dw_eigen.device(place) = dw_eigen.constant(static_cast<T>(0));
  
      // get data ptr
      const T* x_data = input->data<T>();
      const T* w_data = w->data<T>();
      const T* dout_data = dout->data<T>();
      T* dx_data = dx->data<T>();
      T* dw_data = dw->data<T>();
  
      db->mutable_data<T>(ctx.GetPlace());
      auto db_eigen = framework::EigenVector<T>::Flatten(*db);
      db_eigen.device(place) = db_eigen.constant(static_cast<T>(0));
      T* db_data = db->data<T>();
      add_bias_grad<T>(ctx.cuda_device_context().stream(),
                       dout_data,
                       slot_pairs_num,
                       ins_num,
                       out_dim,
                       db_data);
  
      auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
      T alpha = 1;
      T beta = 0;
  
      // dx = dout_data * y^T
      blas.BatchedGEMM(CblasNoTrans,
                       CblasTrans,
                       ins_num,
                       in_dim,
                       out_dim,
                       alpha,
                       dout_data,
                       w_data,
                       beta,
                       dx_data,
                       slot_pairs_num,
                       ins_num * out_dim,
                       out_dim * in_dim);
      // dy = x^T * dout_data
      blas.BatchedGEMM(CblasTrans,
                       CblasNoTrans,
                       in_dim,
                       out_dim,
                       ins_num,
                       alpha,
                       x_data,
                       dout_data,
                       beta,
                       dw_data,
                       slot_pairs_num,
                       in_dim * ins_num,
                       ins_num * out_dim);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = phi::GPUContext;
REGISTER_OP_CUDA_KERNEL(batch_fc,
                        ops::BatchFCCUDAKernel<GPUCtx, float>,
                        ops::BatchFCCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(batch_fc_grad,
                        ops::BatchFCGradOpCUDAKernel<GPUCtx, float>,
                        ops::BatchFCGradOpCUDAKernel<GPUCtx, double>);
