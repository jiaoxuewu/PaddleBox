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
#include "paddle/fluid/operators/fused/fused_seqpool_cvm_op.h"
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
__global__ void FusedSeqpoolKernelNormal(const size_t N, T **input_values,
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
// not need filter quant
template <typename T>
__global__ void FusedSeqpoolKernelQuant(
    const size_t N, T **input_values, T **seqpool_output_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const float pad_value, const int cvm_offset, const int quant_ratio,
    const int embed_thres_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    double val = pad_value;
    // quant
    for (auto k = start; k < end; ++k) {
      if (offset < cvm_offset) {  // show click
        val += *(input_values[x] + k * embedding_size + offset);
      } else {
        val += ((static_cast<int>(
                    *(input_values[x] + k * embedding_size + offset) *
                        quant_ratio +
                    0.5)) /
                static_cast<float>(quant_ratio));
      }
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}
// quant filter
template <typename T>
__global__ void FusedSeqpoolKernelQuantFilter(
    const size_t N, T **input_values, T **seqpool_output_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const float pad_value, const int cvm_offset, const float show_coeff,
    const float clk_coeff, const float threshold, const int quant_ratio,
    const int embed_thres_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx id
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    double val = pad_value;
    for (auto k = start; k < end; ++k) {
      T &show = *(input_values[x] + k * embedding_size);
      T &click = *(input_values[x] + k * embedding_size + 1);
      if ((show - click) * show_coeff + click * clk_coeff < threshold) {
        continue;
      }
      if (offset < cvm_offset) {  // show & click
        val += *(input_values[x] + k * embedding_size + offset);
      } else {
        val += ((static_cast<int>(
                    *(input_values[x] + k * embedding_size + offset) *
                        quant_ratio +
                    0.5)) /
                static_cast<float>(quant_ratio));
      }
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}
// embed quant filter
template <typename T>
__global__ void FusedSeqpoolKernelEmbedQuantFilter(
    const size_t N, T **input_values, T **seqpool_output_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const float pad_value, const int cvm_offset, const float show_coeff,
    const float clk_coeff, const float threshold, const int quant_ratio,
    const float embed_threshold, const int embed_thres_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx id
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    double val = pad_value;
    for (auto k = start; k < end; ++k) {
      T &show = *(input_values[x] + k * embedding_size);
      T &click = *(input_values[x] + k * embedding_size + 1);
      if ((show - click) * show_coeff + click * clk_coeff < threshold) {
        continue;
      }
      T &embedw = *(input_values[x] + k * embedding_size + cvm_offset);
      T embedx_weight_score = 0.0;
      int embed_thres_size_ = embed_thres_size;
      if (embed_thres_size == 0) {
        embed_thres_size_ = embedding_size - cvm_offset;
      }
      for (int i = cvm_offset + 1; i < cvm_offset + embed_thres_size_; i++) {
        embedx_weight_score +=
            pow(*(input_values[x] + k * embedding_size + i), 2);
      }
      embedx_weight_score = std::sqrt(embedx_weight_score) + std::abs(embedw);
      if (embedx_weight_score < embed_threshold) {
        continue;
      }
      if (offset < cvm_offset) {  // show & click
        val += *(input_values[x] + k * embedding_size + offset);
      } else {
        val += ((static_cast<int>(
                    *(input_values[x] + k * embedding_size + offset) *
                        quant_ratio +
                    0.5)) /
                static_cast<float>(quant_ratio));
      }
    }
    *(seqpool_output_values[x] + y * embedding_size + offset) = val;
  }
}

// embed quant filter & expand slot's feasign
template <typename T>
__global__ void FusedSeqpoolKernelEmbedQuantFilterEmbedxConcate(
    const size_t N, T **input_values, T **seqpool_output_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const float pad_value, const int cvm_offset, const float show_coeff,
    const float clk_coeff, const float threshold, const int quant_ratio,
    const float embed_threshold, const int embedx_concate_size, bool embedx_concate_filter, bool fill_zero) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx id
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id
    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);

    double val = pad_value;
    int concate_index = 0;
    bool val_use_zero = false;
    for (auto k = start; k < end; ++k) {
      val_use_zero = false;
      T &show = *(input_values[x] + k * embedding_size);
      T &click = *(input_values[x] + k * embedding_size + 1);
      if (embedx_concate_filter && (show - click) * show_coeff + click * clk_coeff < threshold) {
        if (fill_zero) {
          val_use_zero = true;
        } else {
          continue;
        }
      }
      T &embedw = *(input_values[x] + k * embedding_size + cvm_offset);
      T embedx_weight_score = 0.0;
      for (int i = cvm_offset + 1; i < embedding_size; i++) {
        embedx_weight_score +=
            pow(*(input_values[x] + k * embedding_size + i), 2);
      }
      embedx_weight_score = std::sqrt(embedx_weight_score) + std::abs(embedw);
      if (embedx_concate_filter && embedx_weight_score < embed_threshold) {
        if (fill_zero) {
          val_use_zero = true;
        } else {
          continue;
        }
      }
      if (offset < cvm_offset) {  // show & click
        if (val_use_zero) {
          val = pad_value;
        } else {
          val = *(input_values[x] + k * embedding_size + offset);
        }
      } else {
        if (val_use_zero) {
          val = pad_value;
        } else {
          val = ((static_cast<int>(
                    *(input_values[x] + k * embedding_size + offset) *
                        quant_ratio +
                    0.5)) /
                static_cast<float>(quant_ratio));
        }
      }
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

// join need show click input
template <typename T>
__global__ void FusedCVMKernelWithCVM(const size_t N, T **output_values,
                                      T **seqpool_output_values,
                                      const int batch_size,
                                      const int embedding_size,
                                      const int cvm_offset, bool fix_ctr_to_click) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    // set ptr
    T *in = (seqpool_output_values[x] + y * embedding_size);
    T *out = (output_values[x] + y * embedding_size + offset);
    if (offset == 0) { // log(show + 1)
      *out = log(in[0] + 1);
    } else if (offset == 1) { // ctr = log(click + 1) - log(show + 1)
      *out = log(in[1] + 1) - log(in[0] + 1);
      // fix_ctr_to_click: click += show
      if (fix_ctr_to_click) {
        *out = log(in[1] + 1);
      }
    } else {
      *out = in[offset];
    }
  }
}
// join only need show input
template <typename T>
__global__ void FusedCVMKernelWithShow(const size_t N, T **output_values,
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
      *(output_values[x] + y * noclk_embedding_size) =
          log(*(seqpool_output_values[x] + y * embedding_size) + 1);
    } else {  // skip click offset + 1
      *(output_values[x] + y * noclk_embedding_size + offset) =
          *(seqpool_output_values[x] + y * embedding_size + offset + 1);
    }
  }
}

// join only need show input, and expand slot's feasign
template <typename T>
__global__ void FusedCVMKernelWithShowConcate(const size_t N, T **output_values,
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
          *(output_values[x] + y * noclk_embedding_size * embedx_concate_size + k * noclk_embedding_size) =
              log(*(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size) + 1);
        } else {  // skip click offset + 1
          *(output_values[x] + y * noclk_embedding_size * embedx_concate_size + k * noclk_embedding_size + offset) =
              *(seqpool_output_values[x] + y * embedding_size * embedx_concate_size + k * embedding_size + offset + 1);
        }
    }
  }
}

// update not need show click input
template <typename T>
__global__ void FusedCVMKernelNoCVM(const size_t N, T **output_values,
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

// not need cvm input, expand slot's feasign
template <typename T>
__global__ void FusedCVMKernelNoCVMEmbedxConcate(const size_t N, T **output_values,
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
void FusedSeqpoolCVM(const paddle::platform::Place &place,
                     const std::vector<const T *> &input_data,
                     const std::vector<T *> &output_data,
                     const std::vector<T *> &seqpool_output_data,
                     std::vector<const size_t *> lods, const int batch_size,
                     const int slot_num, const int embedding_size,
                     const float padding_value, const bool use_cvm,
                     const int cvm_offset, float need_filter,
                     const bool embed_threshold_filter, float show_coeff,
                     float clk_coeff, float threshold, float embed_threshold,
                     const int quant_ratio, const bool clk_filter,
                     const int embed_thres_size, const int embedx_concate_size,
                     bool embedx_concate_filter,
                     bool fill_zero, 
                     bool fix_ctr_to_click) {
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
  if (need_filter && embed_threshold_filter) {  // embed quant filter
    if (embedx_concate_size == 1) {
        FusedSeqpoolKernelEmbedQuantFilter<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                             0, stream>>>(
            N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
            embedding_size, padding_value, cvm_offset, show_coeff, clk_coeff,
            threshold, quant_ratio, embed_threshold, embed_thres_size);
    } else {
        FusedSeqpoolKernelEmbedQuantFilterEmbedxConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                         0, stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size, padding_value, cvm_offset, show_coeff, clk_coeff,
        threshold, quant_ratio, embed_threshold, embedx_concate_size, embedx_concate_filter, fill_zero);
    }
  } else if (need_filter) {  // quant need filter
    FusedSeqpoolKernelQuantFilter<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                    stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size, padding_value, cvm_offset, show_coeff, clk_coeff,
        threshold, quant_ratio, embed_thres_size);
  } else if (quant_ratio > 0) {  // quant not filter
    FusedSeqpoolKernelQuant<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                              stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size, padding_value, cvm_offset, quant_ratio, embed_thres_size);
  } else {  // normal
    FusedSeqpoolKernelNormal<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                               stream>>>(
        N, gpu_input_values, gpu_seqpool_output_values, lods_values, batch_size,
        embedding_size, padding_value);
  }
  // second log
  if (use_cvm) {
    if (clk_filter) {  // skip click
      N = static_cast<size_t>(batch_size * slot_num * (embedding_size - 1));
      if (embedx_concate_size == 1) {
        FusedCVMKernelWithShow<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                               stream>>>(N, gpu_output_values,
                                         gpu_seqpool_output_values, batch_size,
                                         embedding_size, embedding_size - 1);
      } else {
        FusedCVMKernelWithShowConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                               stream>>>(N, gpu_output_values,
                                         gpu_seqpool_output_values, batch_size,
                                         embedding_size, embedding_size - 1, embedx_concate_size);
      }
    } else {
      FusedCVMKernelWithCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                              stream>>>(N, gpu_output_values,
                                        gpu_seqpool_output_values, batch_size,
                                        embedding_size, cvm_offset, fix_ctr_to_click);
    }
  } else {
    // not need show click input
    N = static_cast<size_t>(batch_size * slot_num *
                            (embedding_size - cvm_offset - embed_thres_size));
    if (embedx_concate_size == 1) {
      FusedCVMKernelNoCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
            N, gpu_output_values, gpu_seqpool_output_values, batch_size,
            (embedding_size - cvm_offset - embed_thres_size), 
            (cvm_offset + embed_thres_size));
    } else {
      FusedCVMKernelNoCVMEmbedxConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        N, gpu_output_values, gpu_seqpool_output_values, batch_size,
        (embedding_size - cvm_offset), cvm_offset, embedx_concate_size);
    }
  }
}
// join grad
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithCVM(
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
// join only show not has click
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithShow(
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

// only show, expand slot's feasign
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithShowConcate(
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
                 : *(out_grads_values[x] + y * (embedding_size - 1) * embedx_concate_size +
                    (embedding_size - 1) * concate_index + offset - 1);
      *(in_grads_values[x] + k * embedding_size + offset) = val;
      concate_index = concate_index == (embedx_concate_size - 1) ? concate_index : concate_index + 1;
    }
  }
}

// update grad
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelNoCVM(
    const size_t N, T **out_grads_values, T **in_grads_values, T **cvm_values,
    size_t **lods_values, const int batch_size, const int embedding_size,
    const int cvm_offset, const int embed_thres_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    T val = 0;
    if (embed_thres_size == 0) {
        val = (offset < cvm_offset)
                    ? *(cvm_values[x] + y * cvm_offset + offset)
                    : *(out_grads_values[x] + y * (embedding_size
                        - cvm_offset) + offset - cvm_offset);
    } else {
        val = (offset < cvm_offset + embed_thres_size)
                    ? 0
                    : *(out_grads_values[x] + y * (embedding_size
                        - cvm_offset - embed_thres_size) + offset
                        - cvm_offset - embed_thres_size);
    }

    auto &start = *(lods_values[x] + y);
    auto &end = *(lods_values[x] + y + 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}

//@grad, no cvm input, expand slot's feasign
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelNoCVMConcate(
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
void FusedSeqpoolCVMGrad(const paddle::platform::Place &place,
                         const std::vector<const T *> &out_grads_data,
                         const std::vector<T *> &in_grads_data,
                         const std::vector<const T *> &cvm_data,
                         const std::vector<const size_t *> &lods,
                         const int batch_size, const int slot_num,
                         const int embedding_size, const bool use_cvm,
                         const int cvm_offset, const bool clk_filter,
                         const int embed_thres_size, const int embedx_concate_size) {
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
    if (clk_filter) {
      if (embedx_concate_size == 1) {
        FusedSeqpoolCVMGradKernelWithShow<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                          0, stream>>>(
          N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
          lods_values, batch_size, embedding_size, cvm_offset);
      } else {
        FusedSeqpoolCVMGradKernelWithShowConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                          0, stream>>>(
          N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
          lods_values, batch_size, embedding_size, cvm_offset, embedx_concate_size);
      }
    } else {
      // join grad
      FusedSeqpoolCVMGradKernelWithCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS,
                                         0, stream>>>(
          N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
          lods_values, batch_size, embedding_size, cvm_offset);
    }
  } else {
    // update grad
    if (embedx_concate_size == 1) {
      FusedSeqpoolCVMGradKernelNoCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                     stream>>>(
        N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
        lods_values, batch_size, embedding_size, cvm_offset, 
        embed_thres_size);
    } else {
       FusedSeqpoolCVMGradKernelNoCVMConcate<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0,
                                     stream>>>(
        N, gpu_out_grads_values, gpu_in_grads_values, gpu_cvm_values,
        lods_values, batch_size, embedding_size, cvm_offset, embedx_concate_size);
    }
  }
}

template <typename T>
class FusedSeqpoolCVMCUDAKernel : public framework::OpKernel<T> {
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
    bool embed_threshold_filter = ctx.Attr<bool>("embed_threshold_filter");
    float show_coeff = ctx.Attr<float>("show_coeff");
    float clk_coeff = ctx.Attr<float>("clk_coeff");
    float threshold = ctx.Attr<float>("threshold");
    float embed_threshold = ctx.Attr<float>("embed_threshold");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    const int quant_ratio = ctx.Attr<int>("quant_ratio");
    bool clk_filter = ctx.Attr<bool>("clk_filter");
    const int embed_thres_size = ctx.Attr<int>("embed_thres_size");
    const int embedx_concate_size = ctx.Attr<int>("embedx_concate_size");
    bool embedx_concate_filter = ctx.Attr<bool>("embedx_concate_filter");
    bool fill_zero = ctx.Attr<bool>("fill_zero");
    bool fix_ctr_to_click = ctx.Attr<bool>("fix_ctr_to_click");

    framework::GPULodVector gpu_lods[slot_size];
    auto place = ctx.GetPlace();

    if (embedx_concate_size != 1 && embed_thres_size != 0) {
      CHECK(1 == 0) << "embedx_concate_size: " << embedx_concate_size
                    << "embed_thres_size: " << embed_thres_size;
    }
    
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
        CHECK(batch_size == cur_batch) << "batch: " << batch_size
                                       << ", current: " << cur_batch;
      }
      input_data[i] = reinterpret_cast<const T *>(input->data<T>());
      auto *output = outputs[i];
      if (use_cvm) {
        if (clk_filter) {
          output->Resize({batch_size, (embedding_size - 1) * embedx_concate_size});
        } else {
          output->Resize({batch_size, embedding_size});
        }
      } else {
        output->Resize({batch_size, (embedding_size - cvm_offset - embed_thres_size) * embedx_concate_size});
      }
      output_data[i] =
          reinterpret_cast<T *>(output->mutable_data<T>(place));
      lods_data[i] = gpu_lods[i].mutable_data<size_t>(place, lod_data);

      seqpool_output_data[i] =
          reinterpret_cast<T *>(seqpool_outputs[i].mutable_data<T>(
              {batch_size, embedding_size * embedx_concate_size}, place));
    }
    FusedSeqpoolCVM(place, input_data, output_data,
                    seqpool_output_data, lods_data, batch_size, slot_size,
                    embedding_size, padding_value, use_cvm, cvm_offset,
                    need_filter, embed_threshold_filter, show_coeff, clk_coeff,
                    threshold, embed_threshold, quant_ratio, clk_filter,
                    embed_thres_size, embedx_concate_size, embedx_concate_filter, 
                    fill_zero, fix_ctr_to_click);
  }
};

template <typename T>
class FusedSeqpoolCVMGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_grads = ctx.MultiInput<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
    auto *cvm = ctx.Input<LoDTensor>("CVM");

    std::string pooltype = ctx.Attr<std::string>("pooltype");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    bool clk_filter = ctx.Attr<bool>("clk_filter");
    const int embed_thres_size = ctx.Attr<int>("embed_thres_size");
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
          reinterpret_cast<T *>(in_grad->mutable_data<T>(place));
      lods_data[i] = gpu_lods[i].mutable_data<size_t>(place, lod_data);
      
      cvm_data[i] = reinterpret_cast<const T *>(cvm->data<T>());
    }
    FusedSeqpoolCVMGrad(place, out_grads_data, in_grads_data, cvm_data,
                        lods_data, batch_size, slot_size, embedding_size,
                        use_cvm, cvm_offset, clk_filter, embed_thres_size, embedx_concate_size);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm,
                        ops::FusedSeqpoolCVMCUDAKernel<float>);

REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_grad,
                        ops::FusedSeqpoolCVMGradCUDAKernel<float>);
