/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_causal_mask_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

template <typename T>
__global__ void FusedCausalMaskKernel(const T* seq_info,
                                      int64_t* mask,
                                      int total_length,
                                      int num_sequences) {
  auto blockid = blockIdx.x;
  auto start = seq_info[blockid];
  auto end = seq_info[blockid + 1];
  for (int i = threadIdx.x + start; i < end; i += blockDim.x) {
    for (int j = threadIdx.y + start; j <= i; j += blockDim.y) {
      mask[i * total_length + j] = 1;
    }
  }
}

template <typename T>
class FusedCausalMaskCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto seq_info = ctx.Input<framework::Tensor>("SeqInfoInput");
    PADDLE_ENFORCE_NOT_NULL(
        seq_info, platform::errors::NotFound("SeqInfoInput not found"));
    const T* seq_info_data = seq_info->data<T>();
    auto num_sequences = seq_info->dims()[0] - 1;
    T total_length;
    cudaMemcpy(&total_length,
               &seq_info_data[num_sequences],
               sizeof(T),
               cudaMemcpyDeviceToHost);

    auto causal_mask = ctx.Output<framework::Tensor>("CausalMaskOutput");
    PADDLE_ENFORCE_NOT_NULL(causal_mask,
                            platform::errors::NotFound("MaskOut not found"));
    int64_t* causal_mask_data = causal_mask->mutable_data<int64_t>(
        {total_length, total_length}, ctx.GetPlace());
    cudaMemset(
        causal_mask_data, 0, total_length * total_length * sizeof(int64_t));

    dim3 blockSize(16, 16);
    dim3 gridSize(num_sequences);
    // Launch the kernel
    FusedCausalMaskKernel<<<gridSize, blockSize>>>(
        seq_info_data, causal_mask_data, total_length, num_sequences);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_causal_mask,
                        ops::FusedCausalMaskCUDAKernel<int>,
                        ops::FusedCausalMaskCUDAKernel<int64_t>);