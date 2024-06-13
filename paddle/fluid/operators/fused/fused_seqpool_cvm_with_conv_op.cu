//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include "paddle/fluid/operators/fused/fused_seqpool_cvm_with_conv_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

#define GET_BLOCK(N) \
  ((N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS)

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// normal
template <typename T>
__global__ void FusedSeqpoolWithConvKernelNormal(const size_t N, T **input_values,
                                         T **seqpool_output_values,
                                         size_t **lods_values,
                                         const int batch_size,
                                         const int embedding_size,
                                         const float pad_value) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    double val = pad_value;
    for (auto k = start; k < end; ++k) {
      val += *(input_values[x] + k * embedding_size + offset);
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}

// Filter
template <typename T>
__global__ void FusedSeqpoolWithConvKernelFilter(const size_t N, T **input_values,
                                         T **seqpool_output_values,
                                         size_t **lods_values,
                                         const int batch_size,
                                         const int embedding_size,
                                         const float pad_value,
                                         const float show_coeff,
                                         const float clk_coeff,
                                         const float threshold) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    double val = pad_value;
    for (auto k = start; k < end; ++k) {
      T &show = *(input_values[x] + k * embedding_size);
      T &click = *(input_values[x] + k * embedding_size + 1);
      if ((show - click) * show_coeff + click * clk_coeff < threshold) {
        continue;
      }
      val += *(input_values[x] + k * embedding_size + offset);
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}

// normal & expand slot's feasign
template <typename T>
__global__ void FusedSeqpoolWithConvKernelNormalEmbedxConcate(const size_t N, T **input_values,
                                         T **seqpool_output_values,
                                         size_t **lods_values,
                                         const int batch_size,
                                         const int embedding_size,
                                         const float pad_value,
                                         const int embedx_concate_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    int concate_index = 0;

    double val = pad_value;
    for (auto k = start; k < end; ++k) {
      val = *(input_values[x] + k * embedding_size + offset);
      if (concate_index == embedx_concate_size) {
        *(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + (embedx_concate_size-1) * embedding_size + offset) += val;
      } else {
        *(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + concate_index * embedding_size + offset) = val;
        concate_index += 1;
      }
    }
    while (concate_index < embedx_concate_size) {
      *(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + concate_index * embedding_size + offset) = pad_value;
      concate_index += 1;
    }
  }
}

// join only need show input
template <typename T>
__global__ void FusedCVMWithConvKernelNormal(const size_t N, T **output_values,
                                       T **seqpool_output_values,
                                       const int batch_size,
                                       const int embedding_size,
                                       const int noclk_embedding_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / noclk_embedding_size;
    int offset = i % noclk_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    if (offset == 0) {         // show
        *(output_values[x] + y * embedding_size) =
            log(*(seqpool_output_values[x] + y * embedding_size) + 1);
    } else if (offset == 1) {  // click
      *(output_values[x] + y * embedding_size + 1) =
          log(*(seqpool_output_values[x] + y * embedding_size + 1) + 1);
    } else if (offset == 2) {  // conv
      *(output_values[x] + y * embedding_size + 2) =
          log(*(seqpool_output_values[x] + y * embedding_size + 2) + 1) -
          log(*(seqpool_output_values[x] + y * embedding_size + 1) + 1);
    } else {  // filter show, offset - 1
      *(output_values[x] + y * noclk_embedding_size + offset) =
          *(seqpool_output_values[x] + y * embedding_size + offset);
    }
  }
}

// join only need show input, and expand slot's feasign
template <typename T>
__global__ void FusedCVMWithConvKernelNormalConcate(const size_t N, T **output_values,
                                       T **seqpool_output_values,
                                       const int batch_size,
                                       const int embedding_size,
                                       const int noclk_embedding_size,
                                       const int embedx_concate_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / noclk_embedding_size;
    int offset = i % noclk_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    for (int k=0; k < embedx_concate_size; k++) {
      if (offset == 0) {         // show
        *(output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size) =
            log(*(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size) + 1);
      } else if (offset == 1) {  // click
        *(output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size + 1) =
            log(*(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size + 1) + 1);
      } else if (offset == 2) {  // conv
        *(output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size + 2) =
            log(*(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size + 2) + 1) -
            log(*(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size + 1) + 1);
      } else {  // filter show, offset - 1
        *(output_values[x] + y * noclk_embedding_size * embedx_concate_size + k * embedding_size + offset) =
            *(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size + offset);
      }
    }
  }
}

// join only need show input
template <typename T>
__global__ void FusedCVMWithConvKernelWithOutShow(const size_t N, T **output_values,
                                      T **seqpool_output_values,
                                      const int batch_size,
                                      const int embedding_size,
                                      const int noclk_embedding_size) {
 CUDA_KERNEL_LOOP(i, N) {
   int key = i / noclk_embedding_size;
   int offset = i % noclk_embedding_size;
   int x = key / batch_size;  // slot id
   int y = key % batch_size;  // ins id
   if (offset == 0) {         // show
     // do nothing
   } else if (offset == 1) {  // click
     *(output_values[x] + y * embedding_size) =
         log(*(seqpool_output_values[x] + y * embedding_size + 1) + 1);
   } else if (offset == 2) {  // conv
     *(output_values[x] + y * embedding_size + 1) =
         log(*(seqpool_output_values[x] + y * embedding_size + 2) + 1) -
         log(*(seqpool_output_values[x] + y * embedding_size + 1) + 1);
   } else {  // filter show, offset - 1
     *(output_values[x] + y * noclk_embedding_size + offset - 1) =
         *(seqpool_output_values[x] + y * embedding_size + offset);
   }
 }
}

// join only need show input, and expand slot's feasign
template <typename T>
__global__ void FusedCVMWithConvKernelWithOutShowConcate(const size_t N, T **output_values,
                                      T **seqpool_output_values,
                                      const int batch_size,
                                      const int embedding_size,
                                      const int noclk_embedding_size,
                                      const int embedx_concate_size) {
 CUDA_KERNEL_LOOP(i, N) {
   int key = i / noclk_embedding_size;
   int offset = i % noclk_embedding_size;
   int x = key / batch_size;  // slot id
   int y = key % batch_size;  // ins id
   for (int k=0; k < embedx_concate_size; k++) {
     if (offset == 0) {         // show
       // do nothing
     } else if (offset == 1) {  // click
       *(output_values[x] + y * noclk_embedding_size * embedx_concate_size + k * noclk_embedding_size) =
           log(*(seqpool_output_values[x] + y * embedding_size + 1) + 1);
     } else if (offset == 2) {  // conv
       *(output_values[x] + y * noclk_embedding_size * embedx_concate_size + k * noclk_embedding_size + 1) =
           log(*(seqpool_output_values[x] + y * embedding_size + 2) + 1) -
           log(*(seqpool_output_values[x] + y * embedding_size + 1) + 1);
     } else {  // filter show, offset - 1
       *(output_values[x] + y * noclk_embedding_size * embedx_concate_size + k * noclk_embedding_size + offset) =
           *(seqpool_output_values[x] + y * embedding_size + offset);
     }
   }
 }
}

// update not need show click input
template <typename T>
__global__ void FusedCVMWithConvKernelNoCVM(const size_t N, T **output_values,
                                    T **seqpool_output_values,
                                    const int batch_size,
                                    const int no_cvm_embedding_size,
                                    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / no_cvm_embedding_size;
    int offset = i % no_cvm_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    // no cvm
    *(output_values[x] + y * no_cvm_embedding_size + offset) =
        *(seqpool_output_values[x] + y * (no_cvm_embedding_size + cvm_offset) +
          offset + cvm_offset);
  }
}

// update not need show click input, expand slot's feasign
template <typename T>
__global__ void FusedCVMWithConvKernelNoCVMConcate(const size_t N, T **output_values,
                                    T **seqpool_output_values,
                                    const int batch_size,
                                    const int no_cvm_embedding_size,
                                    const int cvm_offset,
                                    const int embedx_concate_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / no_cvm_embedding_size;
    int offset = i % no_cvm_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    // no cvm
    for (int k=0; k < embedx_concate_size; k++) {
      *(output_values[x] + y * no_cvm_embedding_size * embedx_concate_size + k * no_cvm_embedding_size + offset) =
          *(seqpool_output_values[x] + y * (no_cvm_embedding_size + cvm_offset) * embedx_concate_size +
            k * (no_cvm_embedding_size + cvm_offset) + offset + cvm_offset);
    }
  }
}

template <typename T>
void FusedSeqpoolCVMWithConv(const paddle::platform::Place &place,
                     const std::vector<const T *> &input_data,
                     const std::vector<T *> &output_data,
                     const std::vector<T *> &seqpool_output_data,
                     std::vector<const size_t *> lods, const int batch_size,
                     const int slot_num, const int embedding_size,
                     const float padding_value, const bool use_cvm,
                     float need_filter, float show_coeff,
                     float clk_coeff, float threshold,
                     const int cvm_offset, bool show_filter,
                     const int embedx_concate_size) {
  auto stream = dynamic_cast<phi::GPUContext*>(
                 platform::DeviceContextPool::Instance().Get(place))
                 ->stream();

  size_t total_ptr_len = input_data.size() + output_data.size() +
                         seqpool_output_data.size() + lods.size();
  auto temp_ptr = memory::AllocShared(place, total_ptr_len * sizeof(void *));
  void *ptr = temp_ptr->ptr();

  T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
  cudaMemcpyAsync(gpu_input_values, input_data.data(),
                  input_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);
  T **gpu_output_values =
      reinterpret_cast<T **>(&gpu_input_values[input_data.size()]);
  cudaMemcpyAsync(gpu_output_values, output_data.data(),
                  output_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);
  T **gpu_seqpool_output_values =
      reinterpret_cast<T **>(&gpu_output_values[output_data.size()]);
  cudaMemcpyAsync(gpu_seqpool_output_values, seqpool_output_data.data(),
                  seqpool_output_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice, stream);
  size_t **lods_values = reinterpret_cast<size_t **>(
      &gpu_seqpool_output_values[seqpool_output_data.size()]);
  cudaMemcpyAsync(lods_values, lods.data(), lods.size() * sizeof(size_t *),
                  cudaMemcpyHostToDevice, stream);

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  // first sum pool
  if (embedx_concate_size == 1){
    if (need_filter) {  //filter
      FusedSeqpoolWithConvKernelFilter<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                              stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size, padding_value, show_coeff, clk_coeff, threshold);
    } else {  //normal
      FusedSeqpoolWithConvKernelNormal<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size, padding_value);
    }
  } else {
      FusedSeqpoolWithConvKernelNormalEmbedxConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                 stream>>>(
          N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
          embedding_size, padding_value, embedx_concate_size);
  }
  // second log
  if (use_cvm) {
    if (show_filter) {
        N = static_cast<size_t>(batch_size * slot_num * (embedding_size - 1));
        if (embedx_concate_size == 1) {
          FusedCVMWithConvKernelWithOutShow<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                   stream>>>(N, gpu_output_values,
                                             gpu_seqpool_output_values, batch_size,
                                             embedding_size, embedding_size - 1);
        } else {
          FusedCVMWithConvKernelWithOutShowConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                   stream>>>(N, gpu_output_values,
                                             gpu_seqpool_output_values, batch_size,
                                             embedding_size, embedding_size - 1, embedx_concate_size);
        }
    } else {
        if (embedx_concate_size == 1) {
          FusedCVMWithConvKernelNormal<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                   stream>>>(N, gpu_output_values,
                                             gpu_seqpool_output_values, batch_size,
                                             embedding_size, embedding_size);
        } else {
          FusedCVMWithConvKernelNormalConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                   stream>>>(N, gpu_output_values,
                                             gpu_seqpool_output_values, batch_size,
                                             embedding_size, embedding_size, embedx_concate_size);
        }
    }
  } else {
    // not need show click input
    N = static_cast<size_t>(batch_size * slot_num *
                            (embedding_size - cvm_offset));
    if (embedx_concate_size == 1) { 
      FusedCVMWithConvKernelNoCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          N, gpu_output_values, gpu_seqpool_output_values, batch_size,
          (embedding_size - cvm_offset), cvm_offset);
    } else {
      FusedCVMWithConvKernelNoCVMConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          N, gpu_output_values, gpu_seqpool_output_values, batch_size,
          (embedding_size - cvm_offset), cvm_offset, embedx_concate_size);
    }
  }
}

// join grad
template <typename T>
__global__ void FusedSeqpoolCVMWithConvGradKernelWithCVM(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T &val = (offset < cvm_offset)
                 ? *(cvm_values[x] + y * cvm_offset + offset)
                 : *(out_grads_values[x] + y * embedding_size + offset);

    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}

// join grad, expand slot's feasign
template <typename T>
__global__ void FusedSeqpoolCVMWithConvGradKernelWithCVMConcate(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset, const int embedx_concate_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    int concate_index = 0;
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      T &val = (offset < cvm_offset)
                 ? *(cvm_values[x] + y * cvm_offset + offset)
                 : *(out_grads_values[x] + y * embedding_size * embedx_concate_size + 
                   embedding_size * concate_index + offset);
      *(in_grads_values[x] + k * embedding_size + offset) = val;
      concate_index = concate_index == (embedx_concate_size - 1) ? concate_index : concate_index + 1;
    }
  }
}
 
// join only show not has click
template <typename T>
__global__ void FusedSeqpoolCVMWithConvGradKernelWithShow(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T &val =
        (offset < cvm_offset)
            ? *(cvm_values[x] + y * cvm_offset + offset)
            : *(out_grads_values[x] + y * (embedding_size - 1) + offset - 1);

    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}

// join only show not has click, expand slot's feasign
template <typename T>
__global__ void FusedSeqpoolCVMWithConvGradKernelWithShowConcate(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset, const int embedx_concate_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    int concate_index = 0;
    for (auto k = start; k < end; ++k) {
      T &val =
        (offset < cvm_offset)
            ? *(cvm_values[x] + y * cvm_offset + offset)
            : *(out_grads_values[x] + y * (embedding_size - 1) * embedx_concate_size + 
              (embedding_size - 1) * concate_index + offset - 1);
      *(in_grads_values[x] + k * embedding_size + offset) = val;
      concate_index = concate_index == (embedx_concate_size - 1) ? concate_index : concate_index + 1;
    }
  }
}

// update grad
template <typename T>
__global__ void FusedSeqpoolCVMWithConvGradKernelNoCVM(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T &val = (offset < cvm_offset)
                 ? *(cvm_values[x] + y * cvm_offset + offset)
                 : *(out_grads_values[x] + y * (embedding_size - cvm_offset) +
                     offset - cvm_offset);

    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}

// update grad, expand slot's feasign
template <typename T>
__global__ void FusedSeqpoolCVMWithConvGradKernelNoCVMConcate(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset, const int embedx_concate_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    int concate_index = 0;
    for (auto k = start; k < end; ++k) {
      T &val = (offset < cvm_offset)
                 ? *(cvm_values[x] + y * cvm_offset + offset)
                 : *(out_grads_values[x] + y * (embedding_size - cvm_offset) * embedx_concate_size +
                     (embedding_size - cvm_offset) * concate_index + offset - cvm_offset);
      *(in_grads_values[x] + k * embedding_size + offset) = val;
      concate_index = concate_index == (embedx_concate_size - 1) ? concate_index : concate_index + 1;
    }
  }
}

template <typename T>
void FusedSeqpoolCVMGradWithConv(const paddle::platform::Place &place,
                         const std::vector<const T *> &out_grads_data,
                         const std::vector<T *> &in_grads_data,
                         const std::vector<const T *> &cvm_data,
                         const std::vector<const size_t *> &lods,
                         const int batch_size, const int slot_num,
                         const int embedding_size, const bool use_cvm,
                         const int cvm_offset, bool show_filter,
                         const int embedx_concate_size) {
  auto stream = dynamic_cast<phi::GPUContext*>(
                 platform::DeviceContextPool::Instance().Get(place))
                 ->stream();
  size_t total_ptr_len = out_grads_data.size() + in_grads_data.size() +
                         cvm_data.size() + lods.size();
  auto temp_ptr = memory::AllocShared(place, total_ptr_len * sizeof(void *));
  T **gpu_out_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
  cudaMemcpyAsync(gpu_out_grads_values, out_grads_data.data(),
                  out_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  T **gpu_in_grads_values =
      reinterpret_cast<T **>(&gpu_out_grads_values[out_grads_data.size()]);
  cudaMemcpyAsync(gpu_in_grads_values, in_grads_data.data(),
                  in_grads_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  T **gpu_cvm_values =
      reinterpret_cast<T **>(&gpu_in_grads_values[in_grads_data.size()]);
  cudaMemcpyAsync(gpu_cvm_values, cvm_data.data(),
                  cvm_data.size() * sizeof(T *), cudaMemcpyHostToDevice,
                  stream);

  size_t **lods_values =
      reinterpret_cast<size_t **>(&gpu_cvm_values[cvm_data.size()]);
  cudaMemcpyAsync(lods_values, lods.data(), lods.size() * sizeof(size_t *),
                  cudaMemcpyHostToDevice, stream);

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  if (use_cvm) {
    if (show_filter) {
      if (embedx_concate_size == 1) {
        FusedSeqpoolCVMWithConvGradKernelWithShow<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                            0, stream>>>(
            N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
            lods_values, batch_size, embedding_size, cvm_offset);
      } else {
        FusedSeqpoolCVMWithConvGradKernelWithShowConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                            0, stream>>>(
            N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
            lods_values, batch_size, embedding_size, cvm_offset, embedx_concate_size); 
      }
    } else {
      if (embedx_concate_size == 1) { 
        FusedSeqpoolCVMWithConvGradKernelWithCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                            0, stream>>>(
            N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
            lods_values, batch_size, embedding_size, cvm_offset);
      } else {
        FusedSeqpoolCVMWithConvGradKernelWithCVMConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                            0, stream>>>(
            N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
            lods_values, batch_size, embedding_size, cvm_offset, embedx_concate_size);
      }
    }
  } else {
    // update grad
    if (embedx_concate_size == 1) {
      FusedSeqpoolCVMWithConvGradKernelNoCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                       stream>>>(
          N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
          lods_values, batch_size, embedding_size, cvm_offset);
    } else {
     FusedSeqpoolCVMWithConvGradKernelNoCVMConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                       stream>>>(
          N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
          lods_values, batch_size, embedding_size, cvm_offset, embedx_concate_size);
    }
  }
}

template <typename T>
class FusedSeqpoolCVMWithConvCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<LoDTensor>("X");
    auto outputs = ctx.MultiOutput<framework::Tensor>("Out");

    const auto slot_size = inputs.size();
    std::vector<const float *> input_data(slot_size);
    std::vector<const size_t *> lods_data(slot_size);
    std::vector<T *> output_data(slot_size);

    std::vector<LoDTensor> seqpool_outputs(slot_size);
    std::vector<T *> seqpool_output_data(slot_size);

    auto padding_value = ctx.Attr<float>("pad_value");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    bool need_filter = ctx.Attr<bool>("need_filter");
    float show_coeff = ctx.Attr<float>("show_coeff");
    float clk_coeff = ctx.Attr<float>("clk_coeff");
    float threshold = ctx.Attr<float>("threshold");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    bool show_filter = ctx.Attr<bool>("show_filter");
    const int embedx_concate_size = ctx.Attr<int>("embedx_concate_size");

    framework::GPULodVector gpu_lods[slot_size];
    auto place = ctx.GetPlace();
        
    int embedding_size = inputs[0]->numel() / inputs[0]->dims()[0];
    int batch_size = -1;
    for (size_t i = 0; i < slot_size; ++i) {
      const auto *input = inputs[i];

      CHECK(input->lod().size() == 1);
      auto lod_data = input->lod()[0];
      int cur_batch = lod_data.size() - 1;
      if (batch_size == -1) {
        batch_size = cur_batch;
      } else {
        CHECK(batch_size == cur_batch) << "batch: " << batch_size << ", current: " << cur_batch;
      }
      input_data[i] = reinterpret_cast<const T *>(input->data<T>());
      auto *output = outputs[i];
      if (use_cvm) {
        if (show_filter) {
            // show will filtered
            output->Resize({batch_size, (embedding_size - 1) * embedx_concate_size});
        } else {
            // show will filtered
            output->Resize({batch_size, embedding_size * embedx_concate_size});
        }
      } else {
        output->Resize({batch_size, (embedding_size - cvm_offset) * embedx_concate_size});
      }
      output_data[i] =
          reinterpret_cast<T *>(output->mutable_data<T>(ctx.GetPlace()));
      lods_data[i] = gpu_lods[i].mutable_data<size_t>(place, lod_data);

      seqpool_output_data[i] =
          reinterpret_cast<T *>(seqpool_outputs[i].mutable_data<T>(
              {batch_size, embedding_size * embedx_concate_size}, ctx.GetPlace()));
    }
    FusedSeqpoolCVMWithConv(ctx.GetPlace(), input_data, output_data,
                    seqpool_output_data, lods_data, batch_size, slot_size,
                    embedding_size, padding_value, use_cvm,
                    need_filter, show_coeff, clk_coeff, threshold,
                    cvm_offset, show_filter, embedx_concate_size);
  }
};

template <typename T>
class FusedSeqpoolCVMWithConvGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_grads = ctx.MultiInput<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
    auto *cvm = ctx.Input<LoDTensor>("CVM");

    std::string pooltype = ctx.Attr<std::string>("pooltype");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    bool show_filter = ctx.Attr<bool>("show_filter");
    const int embedx_concate_size = ctx.Attr<int>("embedx_concate_size");

    const auto slot_size = in_grads.size();
    std::vector<const T *> out_grads_data(slot_size);
    std::vector<T *> in_grads_data(slot_size);
    std::vector<const T *> cvm_data(slot_size);
    std::vector<const size_t *> lods_data(slot_size);

    framework::GPULodVector gpu_lods[slot_size];
    auto place = ctx.GetPlace();
    
    int embedding_size = in_grads[0]->numel() / in_grads[0]->dims()[0];
    int batch_size = -1;
    for (size_t i = 0; i < slot_size; ++i) {
      auto *in_grad = in_grads[i];

      auto lod_data = in_grad->lod()[0];
      int cur_batch = lod_data.size() - 1;
      if (batch_size == -1) {
        batch_size = cur_batch;
      } else {
        CHECK(batch_size == cur_batch) << "batch: " << batch_size
                                       << ", current: " << cur_batch;
      }

      auto *out_grad = out_grads[i];
      out_grads_data[i] = reinterpret_cast<const T *>(out_grad->data<T>());

      in_grads_data[i] =
          reinterpret_cast<T *>(in_grad->mutable_data<T>(ctx.GetPlace()));
      lods_data[i] = gpu_lods[i].mutable_data<size_t>(place, lod_data);
      cvm_data[i] = reinterpret_cast<const T *>(cvm->data<T>());
    }
    FusedSeqpoolCVMGradWithConv(ctx.GetPlace(), out_grads_data, in_grads_data, cvm_data,
                        lods_data, batch_size, slot_size, embedding_size,
                        use_cvm, cvm_offset, show_filter, embedx_concate_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_with_conv,
                        ops::FusedSeqpoolCVMWithConvCUDAKernel<float>);

REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_with_conv_grad,
                        ops::FusedSeqpoolCVMWithConvGradCUDAKernel<float>);
