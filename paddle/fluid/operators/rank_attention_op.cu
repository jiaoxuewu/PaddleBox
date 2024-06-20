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

#include <algorithm>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/rank_attention.cu.h"
#include "paddle/fluid/operators/rank_attention_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename Dtype>
__global__ void KernelMemSet(const size_t N, const Dtype alpha, Dtype *y) {
  CUDA_KERNEL_LOOP(index, N) { y[index] = alpha; }
}

template <typename DeviceContext, typename T>
class RankAttentionCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<Tensor>("X");
    auto *rank_offset = ctx.Input<Tensor>("RankOffset");
    auto *param = ctx.Input<Tensor>("RankParam");
    auto *input_help = ctx.Output<Tensor>("InputHelp");
    auto *param_help = ctx.Output<Tensor>("ParamHelp");
    auto *ins_rank = ctx.Output<Tensor>("InsRank");
    int max_rank = ctx.Attr<int>("MaxRank");
    int64_t max_size = ctx.Attr<int>("MaxSize");
    auto *Out = ctx.Output<Tensor>("Out");

    // check dims
    auto x_dims = X->dims();
    auto ins_num = x_dims[0];
    auto x_fea_dim = x_dims[1];
    auto para_dims = param->dims();
    auto para_row = para_dims[0];
    auto para_col = para_dims[1];
    auto rank_offset_dims = rank_offset->dims();
    PADDLE_ENFORCE_EQ(
        rank_offset_dims[0], ins_num,
        platform::errors::InvalidArgument("Input(RankOffset) has wrong rows."));
    PADDLE_ENFORCE_EQ((rank_offset_dims[1] - 1) / 2, max_rank,
                      platform::errors::InvalidArgument(
                          "Input(RankOffset) has wrong columns."));
    PADDLE_ENFORCE_EQ(
        max_rank * max_rank * x_fea_dim, para_row,
        platform::errors::InvalidArgument("Input(RankParam) has wrong rows."));

    int block_matrix_row = max_rank * x_fea_dim;

    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();

    int max_ins = std::max(ins_num, max_size);

    param_help->Resize({max_ins * block_matrix_row, para_col});
    param_help->mutable_data<T>(ctx.GetPlace());

    input_help->Resize({max_ins, block_matrix_row});
    ins_rank->Resize({max_ins, 1});
    input_help->mutable_data<T>(ctx.GetPlace());
    ins_rank->mutable_data<T>(ctx.GetPlace());
    Out->mutable_data<T>(ctx.GetPlace());

    auto stream = ctx.cuda_device_context().stream();
    // get data ptr
    T *input_help_data = input_help->data<T>();
    T *param_help_data = param_help->data<T>();
    T *ins_rank_data = ins_rank->data<T>();
    T *out_data = Out->data<T>();

    phi::funcs::set_constant(dev_ctx, ins_rank, -1);

    expand_rank_attention_input(stream, X->data<T>(), ins_num, x_fea_dim,
                                input_help_data, ins_num, block_matrix_row,
                                rank_offset->data<int>(), rank_offset_dims[0],
                                rank_offset_dims[1], ins_rank_data, max_rank);

    expand_rank_attention_param(stream, X->data<T>(), ins_num, x_fea_dim,
                                rank_offset->data<int>(), rank_offset_dims[0],
                                rank_offset_dims[1], param->data<T>(), para_row,
                                para_col, param_help_data,
                                ins_num * block_matrix_row, para_col, max_rank);

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    T alpha = 1;
    T beta = 0;
    int64_t strideA = block_matrix_row;
    int64_t strideB = block_matrix_row * para_col;

    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
    blas.BatchedGEMM(transA, transB, 1, para_col, block_matrix_row, alpha,
                     input_help_data, param_help_data, beta, out_data, ins_num,
                     strideA, strideB);
  }
};

template <typename DeviceContext, typename T>
class RankAttentionGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<Tensor>("X");                     // not use data
    auto *rank_offset = ctx.Input<Tensor>("RankOffset");  // not use data
    auto *param = ctx.Input<Tensor>("RankParam");         // not use data
    auto *input_help = ctx.Input<Tensor>("InputHelp");
    auto *param_help = ctx.Input<Tensor>("ParamHelp");
    auto *ins_rank = ctx.Input<Tensor>("InsRank");
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    int64_t max_size = ctx.Attr<int>("MaxSize");
    bool enable_input_bp = ctx.Attr<bool>("EnableInputBp");

    auto *drank_para = ctx.Output<Tensor>(framework::GradVarName("RankParam"));

    // get dim
    auto x_dims = X->dims();
    auto ins_num = x_dims[0];
    auto x_fea_dim = x_dims[1];
    auto para_dims = param->dims();
    auto para_row = para_dims[0];
    auto para_col = para_dims[1];
    auto rank_offset_dims = rank_offset->dims();
    auto max_rank = (rank_offset_dims[1] - 1) / 2;
    int block_matrix_row = max_rank * x_fea_dim;
    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto stream = ctx.cuda_device_context().stream();
    int max_ins = std::max(ins_num, max_size);
    // initialize out grad
    T *drank_para_ptr = drank_para->mutable_data<T>(ctx.GetPlace());
    cudaMemsetAsync(drank_para_ptr, 0, sizeof(T) * drank_para->numel(), stream);

    // copy data
    Tensor param_grad;
    param_grad = ctx.AllocateTmpTensor<T, DeviceContext>(
        {max_ins * block_matrix_row, para_col}, dev_ctx);

    // initialize
    // get data ptr
    const T *input_help_data = input_help->data<T>();
    const T *ins_rank_data = ins_rank->data<T>();
    T *param_grad_data = param_grad.data<T>();
    cudaMemsetAsync(param_grad_data, 0, sizeof(T) * param_grad.numel(), stream);

    auto blas = phi::funcs::GetBlas<phi::GPUContext, T>(dev_ctx);
    T alpha = 1;
    T beta = 0;

    // get param_grad
    CBLAS_TRANSPOSE transA = CblasTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;
    int64_t strideA = block_matrix_row;
    int64_t strideB = para_col;
    blas.BatchedGEMM(transA, transB, block_matrix_row, para_col, 1, alpha,
                     input_help_data, dout->data<T>(), beta, param_grad_data,
                     ins_num, strideA, strideB);
    // merge param_grad to get drank_para
    merge_rank_attention_param_grad(
        stream, param_grad_data, ins_num * block_matrix_row, para_col,
        drank_para->data<T>(), para_row, para_col, ins_rank_data, ins_num,
        max_rank, x_fea_dim);

    // ------input back propagation------
    if (enable_input_bp) {
        auto rank_offset_row = rank_offset_dims[0];
        auto rank_offset_col = rank_offset_dims[1];
        auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));
        // initialize x-grad
        T *dx_ptr = dx->mutable_data<T>(ctx.GetPlace());
        cudaMemsetAsync(dx_ptr, 0, sizeof(T) * dx->numel(), stream);
        // copy data
        Tensor input_grad;
        input_grad = ctx.AllocateTmpTensor<T, DeviceContext>(
            {max_ins, block_matrix_row}, dev_ctx);
        // get data ptr
        const T *param_help_data = param_help->data<T>();
        const int *rank_offset_data = rank_offset->data<int>();
        T *input_grad_data = input_grad.data<T>();
        cudaMemsetAsync(input_grad_data, 0, sizeof(T) * input_grad.numel(), stream);

        // get param_grad
        transA = CblasNoTrans;
        transB = CblasTrans;
        strideA = para_col;
        strideB = para_col * block_matrix_row;
        blas.BatchedGEMM(transA, transB, 1, block_matrix_row, para_col, alpha,
                        dout->data<T>(), param_help_data, beta, input_grad_data,
                        ins_num, strideA, strideB);
        // merge param_grad to get drank_para
        merge_rank_attention_input_grad(stream, 
            input_grad_data, ins_num, block_matrix_row,
            dx->data<T>(), ins_num, x_fea_dim, 
            rank_offset_data, rank_offset_row, rank_offset_col, 
            ins_rank_data, ins_num, max_rank);
    }
  }
};

template <typename T>
__global__ void kernel_rank_feed_forward(const int ins_num, const int ins_col,
                                         const T *input, const int max_rank,
                                         const int *rank_offset,
                                         const int para_row, const int para_col,
                                         const T *para, T *out_val) {
  int rank_cols = max_rank * 2 + 1;
  // rank offset 2:1:46:2:44:3:45
  CUDA_KERNEL_LOOP(idx, ins_num * para_col) {
    int row_id = idx / para_col;
    int col_id = idx % para_col;

    int lower = rank_offset[row_id * rank_cols] - 1;
    if (lower < 0) {
      out_val[idx] = 0.0;
      continue;
    }

    float sum = 0.0;
    assert(lower < max_rank);
    for (int k = 0; k < max_rank; ++k) {
      int faster = rank_offset[row_id * rank_cols + 2 * k + 1] - 1;
      assert(faster < max_rank);
      // note look rank_offset to know why
      if (faster < 0) {
        continue;
      }
      int index = rank_offset[row_id * rank_cols + 2 * k + 2];
      int start = (lower * max_rank + faster) * ins_col;
      assert(start + ins_col <= para_row);
      assert(index < ins_num);

      for (int j = 0; j < ins_col; ++j) {
        sum +=
            input[index * ins_col + j] * para[(start + j) * para_col + col_id];
      }
    }
    out_val[idx] = sum;
  }
}

template <typename T>
__global__ void kernel_rank_back_propagate(const int para_row,
                                           const int para_col, T *out_para_grad,
                                           const int ins_num, const int ins_col,
                                           const T *input, const int max_rank,
                                           const int *rank_offset,
                                           const T *out_grad) {
  int rank_cols = max_rank * 2 + 1;
  // rank offset 2:1:46:2:44:3:45
  CUDA_KERNEL_LOOP(idx, ins_num * ins_col * para_col * max_rank) {
    int ins_id = idx / para_col / ins_col / max_rank;
    int para_col_id = (idx / ins_col / ins_num) % para_col;
    int ins_col_id = (idx / para_col / max_rank) % ins_col;
    int k = idx % max_rank;

    int lower = rank_offset[ins_id * rank_cols] - 1;
    if (lower < 0) {
      continue;
    }
    assert(lower < max_rank);

    int faster = rank_offset[ins_id * rank_cols + 2 * k + 1] - 1;
    assert(faster < max_rank);
    // note look rank_offset to know why
    if (faster < 0) {
      continue;
    }
    int index = rank_offset[ins_id * rank_cols + 2 * k + 2];
    int start = (lower * max_rank + faster) * ins_col;
    assert(start + ins_col <= para_row);
    assert(index < ins_num);

    paddle::platform::CudaAtomicAdd(
        reinterpret_cast<float *>(
            &out_para_grad[(start + ins_col_id) * para_col + para_col_id]),
        input[index * ins_col + ins_col_id] *
            out_grad[ins_id * para_col + para_col_id]);
  }
}

template <typename DeviceContext, typename T>
class RankAttention2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<Tensor>("X");
    auto *rank_offset = ctx.Input<Tensor>("RankOffset");
    auto *param = ctx.Input<Tensor>("RankParam");
    int max_rank = ctx.Attr<int>("MaxRank");
    auto *Out = ctx.Output<Tensor>("Out");

    // check dims
    auto x_dims = X->dims();
    auto ins_num = x_dims[0];
    auto x_fea_dim = x_dims[1];
    auto para_dims = param->dims();
    auto para_row = para_dims[0];
    auto para_col = para_dims[1];
    auto rank_offset_dims = rank_offset->dims();

    PADDLE_ENFORCE_EQ(
        rank_offset_dims[0], ins_num,
        platform::errors::InvalidArgument("Input(RankOffset) has wrong rows."));
    PADDLE_ENFORCE_EQ((rank_offset_dims[1] - 1) / 2, max_rank,
                      platform::errors::InvalidArgument(
                          "Input(RankOffset) has wrong columns."));
    PADDLE_ENFORCE_EQ(
        max_rank * max_rank * x_fea_dim, para_row,
        platform::errors::InvalidArgument("Input(RankParam) has wrong rows."));

    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto stream = ctx.cuda_device_context().stream();

    // get data ptr
    T *out_data = Out->mutable_data<T>(ctx.GetPlace());

    kernel_rank_feed_forward<<<GET_BLOCKS(ins_num * para_col), CUDA_NUM_THREADS,
                               0, stream>>>(
        ins_num, x_fea_dim, X->data<T>(), max_rank, rank_offset->data<int>(),
        para_row, para_col, param->data<T>(), out_data);
  }
};

template <typename DeviceContext, typename T>
class RankAttention2GradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<Tensor>("X");                     // not use data
    auto *rank_offset = ctx.Input<Tensor>("RankOffset");  // not use data
    auto *param = ctx.Input<Tensor>("RankParam");         // not use data
    auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *drank_para = ctx.Output<Tensor>(framework::GradVarName("RankParam"));

    // get dim
    auto x_dims = X->dims();
    auto ins_num = x_dims[0];
    auto x_fea_dim = x_dims[1];
    auto para_dims = param->dims();
    auto para_row = para_dims[0];
    auto para_col = para_dims[1];
    auto rank_offset_dims = rank_offset->dims();
    auto max_rank = (rank_offset_dims[1] - 1) / 2;

    auto &dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto stream = ctx.cuda_device_context().stream();

    // initialize out grad
    T *drank_para_ptr = drank_para->mutable_data<T>(ctx.GetPlace());
    phi::funcs::set_constant(dev_ctx, drank_para, 0.0);

    kernel_rank_back_propagate<<<GET_BLOCKS(ins_num * x_fea_dim * para_col *
                                            max_rank),
                                 CUDA_NUM_THREADS, 0, stream>>>(
        para_row, para_col, drank_para_ptr, ins_num, x_fea_dim, X->data<T>(),
        max_rank, rank_offset->data<int>(), dout->data<T>());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = phi::GPUContext;
REGISTER_OP_CUDA_KERNEL(rank_attention,
                        ops::RankAttentionCUDAKernel<GPUCtx, float>,
                        ops::RankAttentionCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(rank_attention_grad,
                        ops::RankAttentionGradOpCUDAKernel<GPUCtx, float>,
                        ops::RankAttentionGradOpCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(rank_attention2,
                        ops::RankAttention2CUDAKernel<GPUCtx, float>,
                        ops::RankAttention2CUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(rank_attention2_grad,
                        ops::RankAttention2GradOpCUDAKernel<GPUCtx, float>,
                        ops::RankAttention2GradOpCUDAKernel<GPUCtx, double>);
