#include <fstream>
#include <string>
#include "paddle/fluid/operators/fused/fused_seq_tensor_op.h" // don't remove this
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void cal_ad_slot_session_kernel(const T* input,
                                  const T* ad_input,
                                  T* din_output,
                                  T* ad_slot_session_output,
                                  const size_t batch_num, 
                                  const size_t ins_num, 
                                  const size_t slot_num,
                                  const size_t max_length,
                                  const size_t fea_emb_dim,
                                  const size_t ad_slot_num,
                                  const size_t ad_slot_offset) {
                              
  size_t batch_idx = blockIdx.x;
  size_t ins_idx = blockIdx.y;
  size_t fea_idx = blockIdx.z;

  const size_t one_slot_dim = max_length * fea_emb_dim;
  const size_t one_seq_dim = slot_num * one_slot_dim;
  const size_t ad_seq_dim = ad_slot_num * one_slot_dim;

  const size_t piece_of_ad_seq_dim = ad_slot_num * fea_emb_dim;
  for (size_t idx = threadIdx.x; idx < piece_of_ad_seq_dim; idx += blockDim.x) {
    size_t slot_idx = idx / fea_emb_dim + ad_slot_offset;
    size_t out_slot_idx = idx / fea_emb_dim;
    size_t fea_dim_idx = idx % fea_emb_dim;
    
    size_t input_fea_begin_idx = ins_idx * (batch_num * one_seq_dim) + batch_idx * one_seq_dim +
                                slot_idx * one_slot_dim + fea_idx * fea_emb_dim;

    size_t ad_fea_begin_idx =
      ins_idx * (1 * batch_num * piece_of_ad_seq_dim) +  batch_idx * piece_of_ad_seq_dim +
      out_slot_idx * fea_emb_dim;
    
    const T input_val = input[input_fea_begin_idx + fea_dim_idx];
    const T ad_val = ad_input[ad_fea_begin_idx + fea_dim_idx];

    size_t fea_concat_start_idx =
      batch_idx * (ins_num * ad_seq_dim * 4) + ins_idx * (ad_seq_dim * 4) +
      fea_idx * (piece_of_ad_seq_dim * 4) + out_slot_idx * fea_emb_dim;
    
    din_output[fea_concat_start_idx + fea_dim_idx] = input_val;
    din_output[fea_concat_start_idx + fea_dim_idx + piece_of_ad_seq_dim] = ad_val;
    din_output[fea_concat_start_idx + fea_dim_idx + piece_of_ad_seq_dim * 2] = input_val - ad_val;
    din_output[fea_concat_start_idx + fea_dim_idx + piece_of_ad_seq_dim * 3] = input_val * ad_val;
    
    size_t ad_slot_session_out_start_idx =
      batch_idx * (ins_num * ad_seq_dim) + ins_idx * ad_seq_dim +
      fea_idx * piece_of_ad_seq_dim + out_slot_idx * fea_emb_dim;
    ad_slot_session_output[ad_slot_session_out_start_idx + fea_dim_idx] = input_val;
  }
}

template <typename T>
__global__ void cal_sideinfo_kernel(const T* input,
                                  T* side_info_output,
                                  const size_t batch_num,
                                  const size_t ins_num, 
                                  const size_t slot_num,
                                  const size_t max_length,
                                  const size_t fea_emb_dim,
                                  const size_t sideinfo_slot_num,
                                  const size_t sideinfo_slot_offset) {
  
  size_t batch_idx = blockIdx.x;
  size_t ins_idx = blockIdx.y;
  size_t fea_idx = blockIdx.z;
  
  const size_t one_slot_dim = max_length * fea_emb_dim;
  const size_t input_one_seq_dim = slot_num * one_slot_dim;
  const size_t sideinfo_seq_dim = sideinfo_slot_num * one_slot_dim;

  const size_t piece_of_sideinfo_seq_dim = sideinfo_slot_num * fea_emb_dim;
  for (size_t idx = threadIdx.x; idx < piece_of_sideinfo_seq_dim; idx += blockDim.x) {
    size_t out_slot_idx = idx / fea_emb_dim;
    size_t slot_idx = out_slot_idx + sideinfo_slot_offset;
    size_t fea_dim_idx = idx % fea_emb_dim;
    
    size_t input_fea_begin_idx = ins_idx * (batch_num * input_one_seq_dim) + batch_idx * input_one_seq_dim +
                                slot_idx * one_slot_dim + fea_idx * fea_emb_dim;
    
    size_t fea_transpose_start_idx =
      batch_idx * (ins_num * sideinfo_seq_dim) + ins_idx * sideinfo_seq_dim +
      fea_idx * (sideinfo_slot_num * fea_emb_dim) + out_slot_idx * fea_emb_dim;

    side_info_output[fea_transpose_start_idx + fea_dim_idx] = input[input_fea_begin_idx + fea_dim_idx];
  }
}

template <typename T>
__global__ void cal_sideinfo_kernel_without_loop(const T* input,
                                  T* side_info_output,
                                  const size_t batch_num,
                                  const size_t ins_num, 
                                  const size_t slot_num,
                                  const size_t max_length,
                                  const size_t fea_emb_dim,
                                  const size_t sideinfo_slot_num,
                                  const size_t sideinfo_slot_offset) {
  
  size_t batch_idx = blockIdx.x;
  size_t ins_idx = blockIdx.y;
  size_t fea_idx = blockIdx.z;

  size_t slot_idx = threadIdx.y + sideinfo_slot_offset;
  size_t out_slot_idx = threadIdx.y;
  size_t fea_dim_idx = threadIdx.x;
  
  const size_t one_slot_dim = max_length * fea_emb_dim;
  size_t input_one_seq_dim = slot_num * one_slot_dim;
  size_t out_one_seq_dim = sideinfo_slot_num * one_slot_dim;

  size_t input_fea_begin_idx = ins_idx * (batch_num * input_one_seq_dim) + batch_idx * (input_one_seq_dim) +
                              slot_idx * one_slot_dim + fea_idx * fea_emb_dim;
  
  size_t fea_transpose_start_idx =
    batch_idx * (ins_num * out_one_seq_dim) + ins_idx * out_one_seq_dim +
    fea_idx * (sideinfo_slot_num * fea_emb_dim) + out_slot_idx * fea_emb_dim;

  side_info_output[fea_transpose_start_idx + fea_dim_idx] = input[input_fea_begin_idx + fea_dim_idx];
}

template <typename T>
__device__ void warpReduce(volatile T* cache, int tid) {
    cache[tid] += cache[tid+32];
    cache[tid] += cache[tid+16];
    cache[tid] += cache[tid+8];
    cache[tid] += cache[tid+4];
    cache[tid] += cache[tid+2];
    cache[tid] += cache[tid+1];
}

#define THREAD_PER_BLOCK 128
template <typename T>
__global__ void reduce_sum_max_length(const T* input,
                                      T* mask_output,
                                      const size_t batch_count,
                                      const size_t ins_num,
                                      const size_t slot_num,
                                      const size_t max_length,
                                      const size_t fea_emb_dim) {
    size_t batch_idx = blockIdx.x;
    size_t ins_idx = blockIdx.y; 
    size_t fea_idx = blockIdx.z;

    size_t data_len_per_block = slot_num * fea_emb_dim;
    
    __shared__ T sdata[THREAD_PER_BLOCK];
    //each thread loads one element from global memory to shared mem
    size_t input_start_idx = ins_idx * (batch_count * slot_num * max_length * fea_emb_dim) + 
                              batch_idx * (slot_num * max_length * fea_emb_dim);

    size_t tid = threadIdx.x;
    // memset shared mem
    sdata[tid] = 0;  
    for (size_t idx = tid; idx < data_len_per_block; idx += blockDim.x) {
      size_t slot_idx = idx / fea_emb_dim;
      size_t fea_dim_idx = idx % fea_emb_dim;
      size_t offset = slot_idx * (max_length * fea_emb_dim) + fea_idx * fea_emb_dim + fea_dim_idx;
      sdata[tid] += input[input_start_idx + offset];
    }
    __syncthreads();

    for(size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // When s < 32, we have only one warp left, no need to sync threads, no need to if (tid < s)
    if(tid < 32) {
      warpReduce<T>(sdata, tid);
    }

    if(tid == 0) {
        // [batch_count, ins_num, max_length]
        size_t out_idx = batch_idx * (ins_num * max_length)
                        + ins_idx * (max_length) 
                        + fea_idx;
        if (fabs(sdata[tid]) > 1e-8) {
            mask_output[out_idx] = 1;
        } else {
            mask_output[out_idx] = 0;
        }
    }
}

template <typename T>
class FusedSeqTensorCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto input = ctx.Input<framework::Tensor>("Input");
    PADDLE_ENFORCE_NOT_NULL(input, platform::errors::NotFound("Input not found"));
    auto ad_input = ctx.Input<framework::Tensor>("ADInput");
    PADDLE_ENFORCE_NOT_NULL(ad_input, platform::errors::NotFound("Input not found"));

    auto din_output = ctx.Output<framework::Tensor>("DINOut");
    PADDLE_ENFORCE_NOT_NULL(din_output,
                            platform::errors::NotFound("DINOut not found"));
    T* din_output_data = din_output->mutable_data<T>(ctx.GetPlace());
    auto mask_output = ctx.Output<framework::Tensor>("MaskOut");
    PADDLE_ENFORCE_NOT_NULL(mask_output,
                            platform::errors::NotFound("MaskOut not found"));
    T* mask_output_output_data = mask_output->mutable_data<T>(ctx.GetPlace());
    auto side_info_output = ctx.Output<framework::Tensor>("SideInfoOut");
    PADDLE_ENFORCE_NOT_NULL(side_info_output,
                            platform::errors::NotFound("Output not found"));
    T* side_info_output_data =
        side_info_output->mutable_data<T>(ctx.GetPlace());
    auto ad_slot_session_output =
        ctx.Output<framework::Tensor>("ADSlotSessionOut");
    PADDLE_ENFORCE_NOT_NULL(ad_slot_session_output,
                            platform::errors::NotFound("Output not found"));
    T* ad_slot_session_output_data =
        ad_slot_session_output->mutable_data<T>(ctx.GetPlace());

    auto batch_count = ctx.Attr<int64_t>("batch_count");
    auto max_length = ctx.Attr<int64_t>("max_length");
    auto slot_num = ctx.Attr<int64_t>("slot_num");
    auto fea_emb_dim = ctx.Attr<int64_t>("fea_emb_dim");
    auto ad_slot_num = ctx.Attr<int64_t>("ad_slot_num");
    auto ad_slot_offset = ctx.Attr<int64_t>("ad_slot_offset");

    auto& dev_ctx = ctx.template device_context<phi::GPUContext>();
    auto stream = ctx.cuda_device_context().stream();

    auto input_dims = input->dims();
    size_t ins_num = input_dims[0];

    dim3 ad_grid(batch_count, ins_num, max_length);
    dim3 ad_block(std::min(static_cast<size_t>(1024), static_cast<size_t>(ad_slot_num * fea_emb_dim)));

    cal_ad_slot_session_kernel<<<ad_grid, ad_block, 0, stream>>>(
        input->data<T>(), ad_input->data<T>(), din_output_data,
        ad_slot_session_output_data,
        batch_count, ins_num, slot_num, max_length, fea_emb_dim,
        ad_slot_num, ad_slot_offset);

    size_t sideinfo_slot_offset = 0;
    if (ad_slot_offset == 0) {
      sideinfo_slot_offset = ad_slot_num;
    }
    size_t fea_padding_dim = ((fea_emb_dim + 31) / 32) * 32;
    size_t sideinfo_slot_num = slot_num - ad_slot_num;
    
    if (sideinfo_slot_num * fea_emb_dim < 1024) {
      dim3 sideinfo_grid(batch_count, ins_num, max_length);
      dim3 sideinfo_block(fea_emb_dim, sideinfo_slot_num);
      cal_sideinfo_kernel_without_loop<<<sideinfo_grid, sideinfo_block, 0, stream>>>(
        input->data<T>(), side_info_output_data, batch_count, ins_num, 
        slot_num, max_length, fea_emb_dim,
        sideinfo_slot_num, sideinfo_slot_offset);
    } else {
      dim3 sideinfo_grid(batch_count, ins_num, max_length);
      dim3 sideinfo_block(sideinfo_slot_num * fea_emb_dim);
      cal_sideinfo_kernel<<<sideinfo_grid, sideinfo_block, 0, stream>>>(
          input->data<T>(), side_info_output_data, batch_count, ins_num, 
          slot_num, max_length, fea_emb_dim,
          sideinfo_slot_num, sideinfo_slot_offset);
    }

    dim3 reduce_grid(batch_count, ins_num, max_length);
    dim3 reduce_block(THREAD_PER_BLOCK);
    reduce_sum_max_length<<<reduce_grid, reduce_block, 0, stream>>>(
        input->data<T>(), mask_output_output_data, batch_count,
        ins_num, slot_num, max_length, fea_emb_dim);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
  fused_seq_tensor,
  ops::FusedSeqTensorCUDAKernel<float>);