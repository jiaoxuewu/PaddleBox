// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "paddle/fluid/framework/fleet/metrics.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

#if defined(PADDLE_WITH_CUDA)
#include <cuda_runtime.h>
#endif

namespace paddle {
namespace framework {
const int CUDA_NUM_THREADS = platform::PADDLE_CUDA_NUM_THREADS;
#define GET_BLOCK(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)
__global__ void ComputeThreadValueKernel(const float *label,
                                         const float *pred,
                                         int batch_size,
                                         const int64_t **mask,
                                         int *mask_value,
                                         int mask_size,
                                         double *results) {
  extern __shared__ double shared_data[];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;

  double local_results[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  bool flag = true;

  if (idx < batch_size) {
    for (size_t mask_index = 0; mask_index < mask_size; ++mask_index) {
      if (mask[mask_index][idx] != mask_value[mask_index]) {
        flag = false;
        break;
      }
    }
    if (flag) {
      double diff = pred[idx] - label[idx];
      local_results[0] = fabs(diff);
      local_results[1] = diff * diff;
      local_results[2] = pred[idx];
      local_results[3] = label[idx];
      local_results[4] = 1.0;
    }
  }

  for (int i = 0; i < 5; i++) {
    shared_data[tid + i * blockDim.x] = local_results[i];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (tid < stride) {
      for (int i = 0; i < 5; i++) {
        shared_data[tid + i * blockDim.x] +=
            shared_data[tid + i * blockDim.x + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int i = 0; i < 5; i++) {
      platform::CudaAtomicAdd(&results[i], shared_data[i * blockDim.x]);
    }
  }
}

void BasicAucCalculator::computeThreadValue(
    const float *d_label,
    const float *d_predict,
    int batch_size,
    const std::vector<const int64_t *> &h_mask,
    int *h_mask_value,
    int mask_size,
    std::vector<double> &h_value,
    const paddle::platform::Place &place) {
  auto stream = dynamic_cast<phi::GPUContext *>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  double *d_value = nullptr;
  const int64_t **d_mask = nullptr;
  int *d_mask_value = nullptr;
  cudaMalloc(&d_value, h_value.size() * sizeof(double));
  cudaMalloc((void **)&d_mask, h_mask.size() * sizeof(const int64_t *));
  cudaMalloc(&d_mask_value, mask_size * sizeof(int));

  cudaMemcpy(d_value,
             h_value.data(),
             h_value.size() * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask,
             h_mask.data(),
             h_mask.size() * sizeof(const int64_t *),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_mask_value,
             h_mask_value,
             mask_size * sizeof(int),
             cudaMemcpyHostToDevice);
  int sharedMemSize = h_value.size() * CUDA_NUM_THREADS * sizeof(double);
  ComputeThreadValueKernel<<<GET_BLOCK(batch_size),
                             platform::PADDLE_CUDA_NUM_THREADS,
                             sharedMemSize,
                             stream>>>(
      d_label, d_predict, batch_size, d_mask, d_mask_value, mask_size, d_value);
  cudaStreamSynchronize(stream);
  cudaMemcpy(h_value.data(),
             d_value,
             h_value.size() * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(d_value);
  cudaFree(d_mask);
  cudaFree(d_mask_value);
}
}  // namespace framework
}  // namespace paddle