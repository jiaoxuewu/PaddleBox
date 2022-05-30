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

#ifdef PADDLE_WITH_BOX_PS
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace framework {
const int CUDA_NUM_THREADS = platform::PADDLE_CUDA_NUM_THREADS;
#define GET_BLOCK(N) ((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS)

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)
#define CUDA_BLOCK(N) GET_BLOCK(N), CUDA_NUM_THREADS, 0

template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopy(float** dest, const FEATURE_VALUE_GPU_TYPE* src,
                         const int hidden, const int expand_dim,
                         const int total_len, uint64_t** keys, int* total_dims,
                         const int64_t* slot_lens, const int slot_num,
                         const int* key2slot, const float scale,
                         const int cvm_offset, const int skip_offset) {
  assert(expand_dim == 0);
  // only process no expand data
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[i];
    //    *(dest[x] + y * hidden) = (src + i)->show;
    //    *(dest[x] + y * hidden + 1) = (src + i)->clk;
    //    *(dest[x] + y * hidden + 2) = (src + i)->embed_w;
    float* dest_ptr = dest[x] + y * hidden;
    const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
    for (int k = 0; k < cvm_offset; ++k) {
      dest_ptr[k] = src_ptr[k + skip_offset];
    }
    // embedx
    int embedx_size = hidden - cvm_offset - skip_offset;
    if (src_val.embedding_size > 0) {
      total_dims[i] = 0x01;
      for (int j = 0; j < embedx_size; ++j) {
        dest_ptr[cvm_offset + j] = src_val.embedx[j] * scale;
      }
    } else {
      total_dims[i] = 0x00;
      for (int j = 0; j < embedx_size; ++j) {
        dest_ptr[cvm_offset + j] = 0;
      }
    }
  }  // end kernel loop
}

template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyBase(float** dest, const FEATURE_VALUE_GPU_TYPE* src,
                             const int hidden, const int total_len,
                             uint64_t** keys, int* total_dims,
                             const int64_t* slot_lens, const int slot_num,
                             const int* key2slot, const int cvm_offset,
                             const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[i];
    float* dest_ptr = dest[x] + y * hidden;
    const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
    for (int k = 0; k < cvm_offset; ++k) {
      dest_ptr[k] = src_ptr[k + skip_offset];
    }
    total_dims[i] = static_cast<int>(src_val.embedding_size > 0);
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyExpand(float** dest, const FEATURE_VALUE_GPU_TYPE* src,
                               const int embedx_dim, const int total_len,
                               const int* total_dims, const int64_t* slot_lens,
                               const int slot_num, const int* key2slot,
                               const float scale, const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / embedx_dim;
    int col = i % embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& src_val = src[idx];
    int offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
    if (total_dims[idx] & 0x01) {
      *(dest[x] + offset) = src_val.embedx[col] * scale;
    } else {
      *(dest[x] + offset) = 0;
    }
  }  // end kernel loop
}

template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullDedupCopyBase(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int hidden,
    const int total_len, uint64_t** keys, int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const uint32_t* restore_idx, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[restore_idx[i]];
    float* dest_ptr = dest[x] + y * hidden;
    const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
    for (int k = 0; k < cvm_offset; ++k) {
      dest_ptr[k] = src_ptr[k + skip_offset];
    }
    total_dims[i] = static_cast<int>(src_val.embedding_size > 0);
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullDedupCopyExpand(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int embedx_dim,
    const int total_len, const int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const float scale,
    const int cvm_offset, const uint32_t* restore_idx) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / embedx_dim;
    int col = i % embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& src_val = src[restore_idx[idx]];
    int offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
    if (total_dims[idx] & 0x01) {
      *(dest[x] + offset) = src_val.embedx[col] * scale;
    } else {
      *(dest[x] + offset) = 0;
    }
  }  // end kernel loop
}

//================================== support nncross
//================================
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyBaseNNCross(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int hidden,
    const int expand_dim, const int total_len, uint64_t** keys, int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[i];
    if (dest[x] != 0) {
      float* dest_ptr = dest[x] + y * hidden;
      const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
      for (int k = 0; k < cvm_offset; ++k) {
        dest_ptr[k] = src_ptr[k + skip_offset];
      }
    }

    if (dest[x + slot_num] != 0) {
      float* dest_ptr = dest[x + slot_num] + y * (hidden + expand_dim);
      const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
      for (int k = 0; k < cvm_offset; ++k) {
        dest_ptr[k] = src_ptr[k];
      }
    }

    // embedx flags + expand flags   && *(keys[x] + y) != 0  && *(keys[x] + y)
    // != 0
    total_dims[i] = static_cast<int>(src_val.embedding_size > 0) +
                    (static_cast<int>(src_val.embed_expand_size[0] > 0) << 1);
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyExpandNNCross(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const float scale, const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& src_val = src[idx];
    if (col < embedx_dim) {  // embedx
      if (dest[x] == 0) {
        return;
      }
      int offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
      if (total_dims[idx] & 0x01) {
        *(dest[x] + offset) = src_val.embedx[col] * scale;
      } else {
        *(dest[x] + offset) = 0;
      }

      if (dest[x + slot_num] == 0) {
        return;
      }
      int offset_2 = y * (embedx_dim + cvm_offset + expand_dim)
                    + cvm_offset + col;
      if (total_dims[idx] & 0x02) {
        *(dest[x + slot_num] + offset_2) = src_val.embedx[col] * scale;
      } else {
        *(dest[x + slot_num] + offset_2) = 0;
      }
    } else {  // expand
      if (dest[x + slot_num] == 0) {
        return;
      }
      int offset = y * (embedx_dim + cvm_offset + expand_dim)
                    + cvm_offset + col;
      if (total_dims[idx] & 0x02) {
        *(dest[x + slot_num] + offset) =
            src_val.embed_expand[col - embedx_dim] * scale;
      } else {
        *(dest[x + slot_num] + offset) = 0;
      }
    }
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullDedupCopyBaseNNCross(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int hidden,
    const int expand_dim, const int total_len, uint64_t** keys, int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const uint32_t* restore_idx, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[restore_idx[i]];
    if (dest[x] != 0) {
      float* dest_ptr = dest[x] + y * hidden;
      const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
      for (int k = 0; k < cvm_offset; ++k) {
        dest_ptr[k] = src_ptr[k + skip_offset];
      }
    }

    if (dest[x + slot_num] != 0) {
      float* dest_ptr = dest[x + slot_num] + y * (hidden + expand_dim);
      const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
      for (int k = 0; k < cvm_offset; ++k) {
        dest_ptr[k] = src_ptr[k];
      }
    }

    // embedx flags + expand flags   && *(keys[x] + y) != 0  && *(keys[x] + y)
    // != 0
    total_dims[i] = static_cast<int>(src_val.embedding_size > 0) +
                    (static_cast<int>(src_val.embed_expand_size[0] > 0) << 1);
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullDedupCopyExpandNNCross(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const float scale, const int cvm_offset,
    const uint32_t* restore_idx) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& src_val = src[restore_idx[idx]];
    if (col < embedx_dim) {  // embedx
      if (dest[x] == 0) {
        return;
      }
      int offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
      if (total_dims[idx] & 0x01) {
        *(dest[x] + offset) = src_val.embedx[col] * scale;
      } else {
        *(dest[x] + offset) = 0;
      }

      if (dest[x + slot_num] == 0) {
        return;
      }
      int offset_2 = y * (embedx_dim + cvm_offset + expand_dim)
                    + cvm_offset + col;
      if (total_dims[idx] & 0x02) {
        *(dest[x + slot_num] + offset_2) = src_val.embedx[col] * scale;
      } else {
        *(dest[x + slot_num] + offset_2) = 0;
      }
    } else {  // expand
      if (dest[x + slot_num] == 0) {
        return;
      }
      int offset = y * (embedx_dim + cvm_offset + expand_dim)
                    + cvm_offset + col;
      if (total_dims[idx] & 0x02) {
        *(dest[x + slot_num] + offset) =
            src_val.embed_expand[col - embedx_dim] * scale;
      } else {
        *(dest[x + slot_num] + offset) = 0;
      }
    }
  }  // end kernel loop
}
//========================== feature var pull ========================
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyBaseVariable(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int hidden,
    const int expand_hidden, const int total_len, uint64_t** keys,
    int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[i];
    float* dest_ptr = 0;
    if (dest[x + slot_num] != 0) {
      dest_ptr = dest[x + slot_num] + y * expand_hidden;
      total_dims[i] =
          (static_cast<int>(src_val.embedding_size ==
                            static_cast<uint32_t>(expand_hidden - cvm_offset))
           << 1);
    } else {
      dest_ptr = dest[x] + y * hidden;
      total_dims[i] = static_cast<int>(
          src_val.embedding_size == static_cast<uint32_t>(hidden - cvm_offset));
    }
    assert(dest_ptr != 0);
    if (total_dims[i] == 0 && src_val.embedding_size > 0) {
      total_dims[i] = 0x04;
      for (int k = 0; k < cvm_offset; ++k) {
        dest_ptr[k] = 0;
      }
    } else {
      const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
      for (int k = 0; k < cvm_offset; ++k) {
        dest_ptr[k] = src_ptr[k + skip_offset];
      }
    }
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullCopyExpandVariable(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const float scale, const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& src_val = src[idx];
    if (dest[x + slot_num] != 0) {  // expand
      int offset = y * (expand_dim + cvm_offset) + cvm_offset + col;
      if (total_dims[idx] & 0x02) {
        *(dest[x + slot_num] + offset) = src_val.embedx[col] * scale;
      } else {
        *(dest[x + slot_num] + offset) = 0;
      }
    } else if (dest[x] != 0 && col < embedx_dim) {  // embedx
      int offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
      if (total_dims[idx] & 0x01) {
        *(dest[x] + offset) = src_val.embedx[col] * scale;
      } else {
        *(dest[x] + offset) = 0;
      }
    }
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullDedupCopyBaseVariable(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int hidden,
    const int expand_hidden, const int total_len, uint64_t** keys,
    int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const uint32_t* restore_idx,
    const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& src_val = src[restore_idx[i]];
    float* dest_ptr = 0;
    if (dest[x + slot_num] != 0) {
      dest_ptr = dest[x + slot_num] + y * expand_hidden;
      total_dims[i] =
          (static_cast<int>(src_val.embedding_size ==
                            static_cast<uint32_t>(expand_hidden - cvm_offset))
           << 1);
    } else {
      dest_ptr = dest[x] + y * hidden;
      total_dims[i] = static_cast<int>(
          src_val.embedding_size == static_cast<uint32_t>(hidden - cvm_offset));
    }
    assert(dest_ptr != 0);
    if (total_dims[i] == 0 && src_val.embedding_size > 0) {
      total_dims[i] = 0x04;
      for (int k = 0; k < cvm_offset; ++k) {
        dest_ptr[k] = 0;
      }
    } else {
      const float* src_ptr = reinterpret_cast<const float*>(&src_val.show);
      for (int k = 0; k < cvm_offset; ++k) {
        dest_ptr[k] = src_ptr[k + skip_offset];
      }
    }
    //    assert(src_val.embed_w >= -10.0 && src_val.embed_w <= 10.0);
  }  // end kernel loop
}
template <typename FEATURE_VALUE_GPU_TYPE>
__global__ void PullDedupCopyExpandVariable(
    float** dest, const FEATURE_VALUE_GPU_TYPE* src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const float scale, const int cvm_offset,
    const uint32_t* restore_idx) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& src_val = src[restore_idx[idx]];
    if (dest[x + slot_num] != 0) {  // expand
      int offset = y * (expand_dim + cvm_offset) + cvm_offset + col;
      if (total_dims[idx] & 0x02) {
        *(dest[x + slot_num] + offset) = src_val.embedx[col] * scale;
      } else {
        *(dest[x + slot_num] + offset) = 0;
      }
      //      assert(*(dest[x + slot_num] + offset) >= -10.0 && *(dest[x +
      //      slot_num] + offset) <= 10.0);
    } else if (dest[x] != 0 && col < embedx_dim) {  // embedx
      int offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
      if (total_dims[idx] & 0x01) {
        *(dest[x] + offset) = src_val.embedx[col] * scale;
      } else {
        *(dest[x] + offset) = 0;
      }
      //      assert(*(dest[x] + offset) >= -10.0 && *(dest[x] + offset) <=
      //      10.0);
    }
  }  // end kernel loop
}
//==========================  end ==================================
__global__ void FillKey2Slot(const int total_len, const int64_t* slot_lens,
                             const int slot_num, int* key2slots) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int low = 0;
    int high = slot_num - 1;
    while (low < high) {
      int mid = (low + high) / 2;
      if (i < slot_lens[mid + 1]) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    key2slots[i] = low;
  }
}

__global__ void CopyKeysKernel(const int total_len, uint64_t** src_keys,
                               uint64_t* dest_total_keys,
                               const int64_t* slot_lens, const int* key2slot) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];
    dest_total_keys[i] = src_keys[x][y];
  }
}

template <typename FeaturePushValueGpuType>
__global__ void PushCopy(FeaturePushValueGpuType* dest, float** src,
                         const int hidden, const int expand_dim,
                         const int total_len, int bs, const int* slot_vector,
                         const int* total_dims, const int64_t* slot_lens,
                         const int slot_num, const int* key2slot,
                         const int cvm_offset, const int skip_offset) {
  assert(expand_dim == 0);
  // only process no expand
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];

    float* optr = reinterpret_cast<float*>(&dest_val.show);
    float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = 1.0;
    }
    for (int k = 0; k < cvm_offset; ++k) {
      optr[k + skip_offset] = src_val[k];  // support variable length
    }
    dest_val.embed_g *= -1. * bs;

    if (total_dims[i] & 0x01) {
      for (int j = 0; j < hidden - cvm_offset; ++j) {
        dest_val.embedx_g[j] = src_val[cvm_offset + j] * -1. * bs;
      }
    } else {
      for (int j = 0; j < hidden - cvm_offset; ++j) {
        dest_val.embedx_g[j] = 0;
      }
    }
  }
}
//================================== base push ================================
template <typename FeaturePushValueGpuType>
__global__ void PushCopyBase(FeaturePushValueGpuType* dest, float** src,
                             const int hidden, const int total_len,
                             const int bs, const int* slot_vector,
                             const int* total_dims, const int64_t* slot_lens,
                             const int slot_num, const int* key2slot,
                             const int cvm_offset, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = 1.0;
    }
    for (int k = 0; k < cvm_offset; ++k) {
      optr[k + skip_offset] = src_val[k];  // support variable length
    }
    dest_val.embed_g *= -1. * bs;
  }
}
template <typename FeaturePushValueGpuType>
__global__ void PushCopyExpand(FeaturePushValueGpuType* dest, float** src,
                               const int embedx_dim, const int total_len,
                               const int bs, const int* slot_vector,
                               const int* total_dims, const int64_t* slot_lens,
                               const int slot_num, const int* key2slot,
                               const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / embedx_dim;
    int col = i % embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& dest_val = dest[idx];
    // embedx
    if ((total_dims[idx] & 0x01)) {
      dest_val.embedx_g[col] =
          *(src[x] + y * (embedx_dim + cvm_offset) + cvm_offset + col) * -1. *
          bs;
    } else {
      dest_val.embedx_g[col] = 0;
    }
  }
}

template <typename FeaturePushValueGpuType>
__global__ void PushCopyBaseShareEmbedding(
    FeaturePushValueGpuType* dest, float** src, const int hidden,
    const int total_len, const int bs, const int* slot_vector,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = 1.0;
    }
    for (int k = 0; k < cvm_offset; ++k) {
      optr[k + skip_offset] = src_val[k];  // support variable length
    }

    // for embed_g[SHARE_EMBEDDING_NUM]
    for (int e_index = 0; e_index < (cvm_offset - 2); ++e_index) {
      dest_val.embed_g[e_index] *= -1. * bs;
    }
  }
}

template <typename FeaturePushValueGpuType>
__global__ void PushMergeCopyBase(
    FeaturePushValueGpuType* dest, float** src, const int hidden,
    const int total_len, const int bs, const int* slot_vector,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_cnt,
    const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    const uint32_t& start = d_sort_offset[i];
    const uint32_t& count = d_sort_cnt[i];
    const uint32_t& pos = d_sort_idx[start];

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = count;
    }
    for (int k = 0; k < cvm_offset; ++k) {
      optr[k + skip_offset] = src_val[k];
    }
    // merge same key in diffent slot id
    for (uint32_t j = 1; j < count; ++j) {
      const uint32_t& pos = d_sort_idx[start + j];
      const int& x = key2slot[pos];
      y = pos - slot_lens[x];
      src_val = reinterpret_cast<float*>(src[x] + y * hidden);
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] += src_val[k];
      }
    }
    dest_val.embed_g *= -1. * bs;
  }
}
template <typename FeaturePushValueGpuType>
__global__ void PushMergeCopyExpand(
    FeaturePushValueGpuType* dest, float** src, const int embedx_dim,
    const int total_len, const int bs, const int* slot_vector,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_cnt) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int id = i / embedx_dim;
    int col = i % embedx_dim;

    const uint32_t& start = d_sort_offset[id];
    const uint32_t& count = d_sort_cnt[id];
    const uint32_t& pos = d_sort_idx[start];

    auto& dest_val = dest[id];
    // embedx
    double val = 0;
    int y = 0;
    // merge same key in diffent slot id
    for (uint32_t j = 0; j < count; ++j) {
      const uint32_t& pos = d_sort_idx[start + j];
      const int& x = key2slot[pos];
      if ((total_dims[pos] & 0x01)) {
        y = pos - slot_lens[x];
        val += *(src[x] + y * (embedx_dim + cvm_offset) + cvm_offset + col);
      }
    }
    dest_val.embedx_g[col] = val * -1. * bs;
  }
}

template <typename FeaturePushValueGpuType>
__global__ void PushMergeCopyBaseShareEmbedding(
    FeaturePushValueGpuType* dest, float** src, const int hidden,
    const int total_len, const int bs, const int* slot_vector,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_cnt,
    const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    const uint32_t& start = d_sort_offset[i];
    const uint32_t& count = d_sort_cnt[i];
    const uint32_t& pos = d_sort_idx[start];

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];

    float* optr = reinterpret_cast<float*>(&dest_val.show);
    float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = count;
    }
    for (int k = 0; k < cvm_offset; ++k) {
      optr[k + skip_offset] = src_val[k];
    }
    // merge same key in diffent slot id
    for (uint32_t j = 1; j < count; ++j) {
      const uint32_t& pos = d_sort_idx[start + j];
      const int& x = key2slot[pos];
      y = pos - slot_lens[x];
      src_val = reinterpret_cast<float*>(src[x] + y * hidden);
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] += src_val[k];
      }
    }
    // for embed_g[SHARE_EMBEDDING_NUM]
    for (int e_index = 0; e_index < (cvm_offset - 2); ++e_index) {
      dest_val.embed_g[e_index] *= -1. * bs;
    }
  }
}
//============================== expand nncross ===============================
template <typename FeaturePushValueGpuType>
__global__ void PushCopyBaseNNCross(
    FeaturePushValueGpuType* dest, float** src, const int hidden,
    const int total_len, const int bs, const int* slot_vector,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = 1.0;
    }
    if (src[x] != 0) {
      float* src_val = reinterpret_cast<float*>(src[x] + y * hidden);
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] = src_val[k];  // support variable length
      }
      dest_val.embed_g *= -1. * bs;
    } else {
      for (int k = 1; k < cvm_offset; ++k) {
        optr[k + skip_offset] = 0;  // support variable length
      }
      dest_val.show = 1;
    }
  }
}
template <typename FeaturePushValueGpuType>
__global__ void PushCopyExpandNNCross(
    FeaturePushValueGpuType* dest, float** src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int bs, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& dest_val = dest[idx];
    if (col < embedx_dim) {  // embedx
      if ((total_dims[idx] & 0x01) && src[x] != 0) {
        dest_val.embedx_g[col] =
            *(src[x] + y * (embedx_dim + cvm_offset) + cvm_offset + col) * -1. *
            bs;
      } else {
        dest_val.embedx_g[col] = 0;
      }
    } else {  // expand
      int offset = y * (embedx_dim + cvm_offset + expand_dim)
                    + cvm_offset + col;
      if ((total_dims[idx] & 0x02) && src[x + slot_num] != 0) {
        dest_val.embed_expand_g[col - embedx_dim] =
            *(src[x + slot_num] + offset) * -1. * bs;
      } else {
        dest_val.embed_expand_g[col - embedx_dim] = 0;
      }
    }
  }
}

template <typename FeaturePushValueGpuType>
__global__ void PushMergeCopyBaseNNCross(
    FeaturePushValueGpuType* dest, float** src, const int hidden,
    const int total_len, const int bs, const int* slot_vector,
    const int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_cnt,
    const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    const uint32_t& start = d_sort_offset[i];
    const uint32_t& count = d_sort_cnt[i];
    const uint32_t& pos = d_sort_idx[start];

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = count;
    }
    float* src_val = NULL;
    if (src[x] != 0) {
      src_val = reinterpret_cast<float*>(src[x] + y * hidden);
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] = src_val[k];
      }
    } else {
      for (int k = 1; k < cvm_offset; ++k) {
        optr[k + skip_offset] = 0;
      }
      dest_val.show = 1;
    }
    // merge same key in diffent slot id
    for (uint32_t j = 1; j < count; ++j) {
      const uint32_t& pos = d_sort_idx[start + j];
      const int& x = key2slot[pos];
      if (src[x] == 0) {
        continue;
      }
      y = pos - slot_lens[x];
      src_val = reinterpret_cast<float*>(src[x] + y * hidden);
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] += src_val[k];
      }
    }
    dest_val.embed_g *= -1. * bs;
  }
}
template <typename FeaturePushValueGpuType>
__global__ void PushMergeCopyExpandNNCross(
    FeaturePushValueGpuType* dest, float** src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int bs, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_cnt) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int id = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    const uint32_t& start = d_sort_offset[id];
    const uint32_t& count = d_sort_cnt[id];
    const uint32_t& pos = d_sort_idx[start];

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];

    auto& dest_val = dest[id];
    if (col < embedx_dim) {  // embedx
      double val = 0.0;
      for (uint32_t j = 0; j < count; ++j) {
        const uint32_t& pos = d_sort_idx[start + j];
        const int& x = key2slot[pos];
        if ((total_dims[pos] & 0x01) && src[x] != 0) {
          y = pos - slot_lens[x];
          val += *(src[x] + y * (embedx_dim + cvm_offset) + cvm_offset + col);
        }
      }
      dest_val.embedx_g[col] = val * -1. * bs;
    } else {                   // expand
      int offset = y * (embedx_dim + cvm_offset + expand_dim)
                    + cvm_offset + col;
      double val = 0.0;
      for (uint32_t j = 0; j < count; ++j) {
        const uint32_t& pos = d_sort_idx[start + j];
        const int& x = key2slot[pos];
        if ((total_dims[pos] & 0x02) && src[x + slot_num] != 0) {
          y = pos - slot_lens[x];
          val += *(src[x + slot_num] + offset);
        }
      }
      dest_val.embed_expand_g[col - embedx_dim] = val * -1 * bs;
    }
  }
}
//========================== feature variable push ============================
template <typename FeaturePushValueGpuType>
__global__ void PushCopyBaseVariable(
    FeaturePushValueGpuType* dest, float** src, const int hidden,
    const int expand_hidden, const int total_len, const int bs,
    const int* slot_vector, const int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int cvm_offset,
    const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int x = key2slot[i];
    int y = i - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = 1.0;
    }
    float* src_val = 0;
    if (src[x + slot_num] != 0) {
      src_val = reinterpret_cast<float*>(src[x + slot_num] + y * expand_hidden);
    } else {
      src_val = reinterpret_cast<float*>(src[x] + y * hidden);
    }
    assert(src_val != 0);
    if (total_dims[i] & 0x04) {
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] = 0;  // support variable length
      }
    } else {
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] = src_val[k];  // support variable length
      }
      dest_val.embed_g *= -1. * bs;
    }
  }
}
template <typename FeaturePushValueGpuType>
__global__ void PushCopyExpandVariable(
    FeaturePushValueGpuType* dest, float** src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int bs, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int idx = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    int x = key2slot[idx];
    int y = idx - slot_lens[x];

    auto& dest_val = dest[idx];
    if ((total_dims[idx] & 0x02) && src[x + slot_num] != 0) {  // expand
      int offset = y * (expand_dim + cvm_offset) + cvm_offset + col;
      dest_val.embedx_g[col] = *(src[x + slot_num] + offset) * -1. * bs;
    } else if ((total_dims[idx] & 0x01) && src[x] != 0 &&
               col < embedx_dim) {  // embedx
      int offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
      dest_val.embedx_g[col] = *(src[x] + offset) * -1. * bs;
    } else {
      dest_val.embedx_g[col] = 0;
    }
  }
}

template <typename FeaturePushValueGpuType>
__global__ void PushMergeCopyBaseVariable(
    FeaturePushValueGpuType* dest, float** src, const int hidden,
    const int expand_hidden, const int total_len, const int bs,
    const int* slot_vector, const int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int cvm_offset,
    const uint32_t* d_sort_idx, const uint32_t* d_sort_offset,
    const uint32_t* d_sort_cnt, const int skip_offset) {
  CUDA_KERNEL_LOOP(i, total_len) {
    const uint32_t& start = d_sort_offset[i];
    const uint32_t& count = d_sort_cnt[i];
    const uint32_t& pos = d_sort_idx[start];

    const int& x = key2slot[pos];
    int y = pos - slot_lens[x];

    auto& dest_val = dest[i];
    dest_val.slot = slot_vector[x];

    float* src_val = NULL;
    float* optr = reinterpret_cast<float*>(&dest_val.show);
    for (int k = 0; k < skip_offset; ++k) {
      optr[k] = count;
    }
    if (total_dims[pos] & 0x04) {
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] = 0;
      }
    } else {
      if (src[x + slot_num] != 0) {
        src_val =
            reinterpret_cast<float*>(src[x + slot_num] + y * expand_hidden);
      } else {
        src_val = reinterpret_cast<float*>(src[x] + y * hidden);
      }
      assert(src_val != 0);
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] = src_val[k];
      }
    }
    // merge same key in diffent slot id
    for (uint32_t j = 1; j < count; ++j) {
      const uint32_t& pos = d_sort_idx[start + j];
      const int& x = key2slot[pos];
      y = pos - slot_lens[x];
      if (total_dims[pos] & 0x04) {
        continue;
      }
      if (src[x + slot_num] != 0) {
        src_val =
            reinterpret_cast<float*>(src[x + slot_num] + y * expand_hidden);
      } else {
        src_val = reinterpret_cast<float*>(src[x] + y * hidden);
      }
      assert(src_val != 0);
      for (int k = 0; k < cvm_offset; ++k) {
        optr[k + skip_offset] += src_val[k];
      }
    }
    dest_val.embed_g *= -1. * bs;
  }
}
template <typename FeaturePushValueGpuType>
__global__ void PushMergeCopyExpandVariable(
    FeaturePushValueGpuType* dest, float** src, const int total_embedx_dim,
    const int embedx_dim, const int expand_dim, const int total_len,
    const int bs, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_cnt) {
  CUDA_KERNEL_LOOP(i, total_len) {
    int id = i / total_embedx_dim;
    int col = i % total_embedx_dim;

    const uint32_t& start = d_sort_offset[id];
    const uint32_t& count = d_sort_cnt[id];

    auto& dest_val = dest[id];
    double val = 0.0;
    int y = 0;
    int offset = 0;
    // merge same key in diffent slot id
    for (uint32_t j = 0; j < count; ++j) {
      const uint32_t& pos = d_sort_idx[start + j];
      const int& x = key2slot[pos];
      y = pos - slot_lens[x];
      if ((total_dims[pos] & 0x02) && src[x + slot_num] != 0) {  // expand
        offset = y * (expand_dim + cvm_offset) + cvm_offset + col;
        val += *(src[x + slot_num] + offset);
      } else if ((total_dims[pos] & 0x01) && src[x] != 0 &&
                 col < embedx_dim) {  // embedx
        offset = y * (embedx_dim + cvm_offset) + cvm_offset + col;
        val += *(src[x] + offset);
      }
    }
    dest_val.embedx_g[col] = -1. * bs * val;
  }
}

__device__ void add_calculator_value(const int table_size, const float pred,
                                     const int64_t label, const int idx,
                                     double* positive, double* negative,
                                     double* abs_error, double* sqr_error,
                                     double* local_pred) {
  int pos = static_cast<int>(pred * table_size);
  if (pos >= table_size) {
    pos = table_size - 1;
  }
  if (label == 0) {
    // atomicAdd(negative + pos, 1.0);
    paddle::platform::CudaAtomicAdd(negative + pos, 1.0);
  } else {
    // atomicAdd(positive + pos, 1.0);
    paddle::platform::CudaAtomicAdd(positive + pos, 1.0);
  }
  double err = pred - label;
  abs_error[idx] += fabs(err);
  sqr_error[idx] += err * err;
  local_pred[idx] += pred;
}

__global__ void AddBasicCalculator(const float* pred, const int64_t* label,
                                   double* positive, double* negative,
                                   double* abs_error, double* sqr_error,
                                   double* local_pred, int len,
                                   int table_size) {
  CUDA_KERNEL_LOOP(ins_idx, len) {
    add_calculator_value(table_size, pred[ins_idx], label[ins_idx], ins_idx,
                         positive, negative, abs_error, sqr_error, local_pred);
  }
}

__global__ void AddMaskCalculator(const float* pred, const int64_t* label,
                                  const int64_t* mask, double* positive,
                                  double* negative, double* abs_error,
                                  double* sqr_error, double* local_pred,
                                  int len, int table_size) {
  CUDA_KERNEL_LOOP(ins_idx, len) {
    if (mask[ins_idx] != 1) {
      continue;
    }
    add_calculator_value(table_size, pred[ins_idx], label[ins_idx], ins_idx,
                         positive, negative, abs_error, sqr_error, local_pred);
  }
}
template <typename FeaturePullValueType>
void FeaturePullCopy(cudaStream_t stream, uint64_t** gpu_keys,
                     float** gpu_values, void* src, const int hidden_size,
                     const size_t embedx_dim, const int total_length,
                     int* total_dims, const int64_t* slot_lens,
                     const int slot_num, const int* key2slot, const float scale,
                     const int cvm_offset, const uint32_t* gpu_restore_idx,
                     const int skip_offset) {
  FeaturePullValueType* pull_values_gpu =
      reinterpret_cast<FeaturePullValueType*>(src);
  if (gpu_restore_idx != nullptr) {
    // normal
    PullDedupCopyBase<
        FeaturePullValueType><<<CUDA_BLOCK(total_length), stream>>>(
        gpu_values, pull_values_gpu, hidden_size, total_length, gpu_keys,
        total_dims, slot_lens, slot_num, key2slot, cvm_offset, gpu_restore_idx,
        skip_offset);
    if (embedx_dim == 0) {
      return;
    }
    // embedx
    int embedx_total_length = total_length * embedx_dim;
    PullDedupCopyExpand<
        FeaturePullValueType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        gpu_values, pull_values_gpu, embedx_dim, embedx_total_length,
        total_dims, slot_lens, slot_num, key2slot, scale, cvm_offset,
        gpu_restore_idx);
  } else {
    // normal
    PullCopyBase<FeaturePullValueType><<<CUDA_BLOCK(total_length), stream>>>(
        gpu_values, pull_values_gpu, hidden_size, total_length, gpu_keys,
        total_dims, slot_lens, slot_num, key2slot, cvm_offset, skip_offset);
    if (embedx_dim == 0) {
      return;
    }
    // embedx
    int embedx_total_length = total_length * embedx_dim;
    PullCopyExpand<
        FeaturePullValueType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        gpu_values, pull_values_gpu, embedx_dim, embedx_total_length,
        total_dims, slot_lens, slot_num, key2slot, scale, cvm_offset);
  }
}

template <typename FeaturePullValueType>
void FeaturePullCopyNNCross(cudaStream_t stream, uint64_t** gpu_keys,
                            float** gpu_values, void* src,
                            const int hidden_size, const size_t embedx_dim,
                            const size_t expand_dim, const int total_length,
                            int* total_dims, const int64_t* slot_lens,
                            const int slot_num, const int* key2slot,
                            const float scale, const int cvm_offset,
                            const uint32_t* gpu_restore_idx,
                            const int skip_offset) {
  FeaturePullValueType* pull_values_gpu =
      reinterpret_cast<FeaturePullValueType*>(src);
  if (gpu_restore_idx != nullptr) {
    // nncross
    PullDedupCopyBaseNNCross<
        FeaturePullValueType><<<CUDA_BLOCK(total_length), stream>>>(
        gpu_values, pull_values_gpu, hidden_size, expand_dim, total_length,
        gpu_keys, total_dims, slot_lens, slot_num, key2slot, cvm_offset,
        gpu_restore_idx, skip_offset);
    // embedx + expand_embedx
    int embedx_total_length = total_length * (embedx_dim + expand_dim);
    PullDedupCopyExpandNNCross<
        FeaturePullValueType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        gpu_values, pull_values_gpu, (embedx_dim + expand_dim), embedx_dim,
        expand_dim, embedx_total_length, total_dims, slot_lens, slot_num,
        key2slot, scale, cvm_offset, gpu_restore_idx);
  } else {
    // nncross
    PullCopyBaseNNCross<
        FeaturePullValueType><<<CUDA_BLOCK(total_length), stream>>>(
        gpu_values, pull_values_gpu, hidden_size, expand_dim, total_length,
        gpu_keys, total_dims, slot_lens, slot_num, key2slot, cvm_offset,
        skip_offset);
    // embedx + expand_embedx
    int embedx_total_length = total_length * (embedx_dim + expand_dim);
    PullCopyExpandNNCross<
        FeaturePullValueType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        gpu_values, pull_values_gpu, (embedx_dim + expand_dim), embedx_dim,
        expand_dim, embedx_total_length, total_dims, slot_lens, slot_num,
        key2slot, scale, cvm_offset);
  }
}

template <typename FeaturePullValueType>
void FeaturePullCopyVariable(cudaStream_t stream, uint64_t** gpu_keys,
                             float** gpu_values, void* src,
                             const int hidden_size, const size_t embedx_dim,
                             const size_t expand_dim, const int total_length,
                             int* total_dims, const int64_t* slot_lens,
                             const int slot_num, const int* key2slot,
                             const float scale, const int cvm_offset,
                             const uint32_t* gpu_restore_idx,
                             const int skip_offset) {
  FeaturePullValueType* pull_values_gpu =
      reinterpret_cast<FeaturePullValueType*>(src);
  if (gpu_restore_idx != nullptr) {
    PullDedupCopyBaseVariable<
        FeaturePullValueType><<<CUDA_BLOCK(total_length), stream>>>(
        gpu_values, pull_values_gpu, hidden_size, (expand_dim + cvm_offset),
        total_length, gpu_keys, total_dims, slot_lens, slot_num, key2slot,
        cvm_offset, gpu_restore_idx, skip_offset);
    // embedx or expand_embedx
    int max_embedx_dim = (embedx_dim > expand_dim) ? embedx_dim : expand_dim;
    int embedx_total_length = total_length * max_embedx_dim;
    PullDedupCopyExpandVariable<
        FeaturePullValueType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        gpu_values, pull_values_gpu, max_embedx_dim, embedx_dim, expand_dim,
        embedx_total_length, total_dims, slot_lens, slot_num, key2slot, scale,
        cvm_offset, gpu_restore_idx);
  } else {
    PullCopyBaseVariable<
        FeaturePullValueType><<<CUDA_BLOCK(total_length), stream>>>(
        gpu_values, pull_values_gpu, hidden_size, (expand_dim + cvm_offset),
        total_length, gpu_keys, total_dims, slot_lens, slot_num, key2slot,
        cvm_offset, skip_offset);
    // embedx or expand_embedx
    int max_embedx_dim = (embedx_dim > expand_dim) ? embedx_dim : expand_dim;
    int embedx_total_length = total_length * max_embedx_dim;
    PullCopyExpandVariable<
        FeaturePullValueType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        gpu_values, pull_values_gpu, max_embedx_dim, embedx_dim, expand_dim,
        embedx_total_length, total_dims, slot_lens, slot_num, key2slot, scale,
        cvm_offset);
  }
}

void BoxWrapper::CopyForPull(const paddle::platform::Place& place,
                             uint64_t** gpu_keys, float** gpu_values,
                             void* total_values_gpu, const int64_t* slot_lens,
                             const int slot_num, const int* key2slot,
                             const int hidden_size, const int expand_embed_dim,
                             const int64_t total_length, int* total_dims,
                             const int skip_offset,
                             const uint32_t* gpu_restore_idx) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  const int cvm_offset = cvm_offset_ - skip_offset;
#define EMBEDX_CASE(i, ...)                                                  \
  case i: {                                                                  \
    constexpr size_t EmbedxDim = i;                                          \
    switch (expand_embed_dim_) {                                             \
      __VA_ARGS__                                                            \
      default:                                                               \
        PADDLE_THROW(platform::errors::InvalidArgument(                      \
            "Unsupport this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                        \
  } break

#define EXPAND_EMBED_PULL_CASE(i, ...)                                        \
  case i: {                                                                   \
    constexpr size_t ExpandDim = i;                                           \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {             \
      FeaturePullCopy<boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>>(  \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,        \
          EmbedxDim, total_length, total_dims, slot_lens, slot_num, key2slot, \
          pull_embedx_scale_, cvm_offset, gpu_restore_idx, skip_offset);      \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_QUANT) ||     \
               feature_type_ == static_cast<int>(boxps::FEATURE_SHOWCLK)) {   \
      FeaturePullCopy<boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>>( \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,        \
          EmbedxDim, total_length, total_dims, slot_lens, slot_num, key2slot, \
          pull_embedx_scale_, cvm_offset, gpu_restore_idx, skip_offset);      \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_CONV)) {      \
      FeaturePullCopy<boxps::FeaturePullValueGpuConv<EmbedxDim, ExpandDim>>(  \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,        \
          EmbedxDim, total_length, total_dims, slot_lens, slot_num, key2slot, \
          pull_embedx_scale_, cvm_offset, gpu_restore_idx, skip_offset);      \
    } else {                                                                  \
      FeaturePullCopy<boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>>(      \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,        \
          EmbedxDim, total_length, total_dims, slot_lens, slot_num, key2slot, \
          1.0, cvm_offset, gpu_restore_idx, skip_offset);                     \
    }                                                                         \
  } break

#define EXPAND_EMBED_PULL_NNCROSS(i, ...)                                      \
  case i: {                                                                    \
    constexpr size_t ExpandDim = i;                                            \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {              \
      FeaturePullCopyNNCross<                                                  \
          boxps::FeaturePullValueGpuPCOC<EmbedxDim, ExpandDim>>(               \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,         \
          EmbedxDim, ExpandDim, total_length, total_dims, slot_lens, slot_num, \
          key2slot, pull_embedx_scale_, cvm_offset, gpu_restore_idx,           \
          skip_offset);                                                        \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_QUANT) ||      \
               feature_type_ == static_cast<int>(boxps::FEATURE_SHOWCLK)) {    \
      FeaturePullCopyNNCross<                                                  \
          boxps::FeaturePullValueGpuQuant<EmbedxDim, ExpandDim>>(              \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,         \
          EmbedxDim, ExpandDim, total_length, total_dims, slot_lens, slot_num, \
          key2slot, pull_embedx_scale_, cvm_offset, gpu_restore_idx,           \
          skip_offset);                                                        \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_VARIABLE)) {   \
      FeaturePullCopyVariable<                                                 \
          boxps::FeatureVarPullValueGpu<EmbedxDim, ExpandDim>>(                \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,         \
          EmbedxDim, ExpandDim, total_length, total_dims, slot_lens, slot_num, \
          key2slot, 1.0, cvm_offset, gpu_restore_idx, skip_offset);            \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_CONV)) {       \
      FeaturePullCopyNNCross<                                                  \
          boxps::FeaturePullValueGpuConv<EmbedxDim, ExpandDim>>(               \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,         \
          EmbedxDim, ExpandDim, total_length, total_dims, slot_lens, slot_num, \
          key2slot, 1.0, cvm_offset, gpu_restore_idx, skip_offset);            \
    } else {                                                                   \
      FeaturePullCopyNNCross<                                                  \
          boxps::FeaturePullValueGpu<EmbedxDim, ExpandDim>>(                   \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,         \
          EmbedxDim, ExpandDim, total_length, total_dims, slot_lens, slot_num, \
          key2slot, 1.0, cvm_offset, gpu_restore_idx, skip_offset);            \
    }                                                                          \
  } break

#define EXPAND_EMBED_PULL_SHARE(i, ...)                                        \
  case i: {                                                                    \
    constexpr size_t ExpandDim = i;                                            \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_SHARE_EMBEDDING)) {   \
      FeaturePullCopy<                                                         \
          boxps::FeaturePullValueGpuShareEmbedding<EmbedxDim, ExpandDim>>(     \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,         \
          (hidden_size - cvm_offset), total_length, total_dims, slot_lens,     \
          slot_num, key2slot, pull_embedx_scale_, cvm_offset, gpu_restore_idx, \
          skip_offset);                                                        \
    }                                                                          \
  } break

// adam 767 embedx=0, expand=767 for PGL ERNIE
#define EXPAND_EMBED_PULL_ADAMEX(i, ...)                                      \
  case i: {                                                                   \
    constexpr size_t ExpandDim = i;                                           \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_ADAM)) {             \
      FeaturePullCopy<boxps::FeatureVarPullValueGpu<EmbedxDim, ExpandDim>>(   \
          stream, gpu_keys, gpu_values, total_values_gpu, hidden_size,        \
          ExpandDim, total_length, total_dims, slot_lens, slot_num, key2slot, \
          1.0, cvm_offset, gpu_restore_idx, skip_offset);                     \
    }                                                                         \
  } break

  switch (embedx_dim_) {
    EMBEDX_CASE(0, EXPAND_EMBED_PULL_CASE(0); EXPAND_EMBED_PULL_ADAMEX(255);
                EXPAND_EMBED_PULL_ADAMEX(767););
    EMBEDX_CASE(2, EXPAND_EMBED_PULL_CASE(0););
    EMBEDX_CASE(4, EXPAND_EMBED_PULL_CASE(0););
    EMBEDX_CASE(8, EXPAND_EMBED_PULL_CASE(0); EXPAND_EMBED_PULL_SHARE(1);
                EXPAND_EMBED_PULL_SHARE(2); EXPAND_EMBED_PULL_SHARE(3);
                EXPAND_EMBED_PULL_SHARE(4); EXPAND_EMBED_PULL_SHARE(5);
                EXPAND_EMBED_PULL_SHARE(6); EXPAND_EMBED_PULL_SHARE(7);
                EXPAND_EMBED_PULL_SHARE(8); EXPAND_EMBED_PULL_NNCROSS(64););
    EMBEDX_CASE(16, EXPAND_EMBED_PULL_CASE(0); EXPAND_EMBED_PULL_SHARE(1);
                EXPAND_EMBED_PULL_SHARE(2); EXPAND_EMBED_PULL_SHARE(3);
                EXPAND_EMBED_PULL_SHARE(4); EXPAND_EMBED_PULL_SHARE(5);
                EXPAND_EMBED_PULL_SHARE(6); EXPAND_EMBED_PULL_SHARE(7);
                EXPAND_EMBED_PULL_SHARE(8); EXPAND_EMBED_PULL_NNCROSS(64););
    EMBEDX_CASE(32, EXPAND_EMBED_PULL_CASE(0););
    EMBEDX_CASE(64, EXPAND_EMBED_PULL_CASE(0););
    EMBEDX_CASE(256, EXPAND_EMBED_PULL_CASE(0););
    EMBEDX_CASE(128, EXPAND_EMBED_PULL_CASE(0););
    EMBEDX_CASE(280, EXPAND_EMBED_PULL_CASE(0););
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupport this embedding size [%d]", hidden_size - cvm_offset));
  }
  cudaStreamSynchronize(stream);
#undef EXPAND_EMBED_PULL_CASE
#undef EMBEDX_CASE
}

void BoxWrapper::CopyKeys(const paddle::platform::Place& place,
                          uint64_t** origin_keys, uint64_t* total_keys,
                          const int64_t* slot_lens, int slot_num, int total_len,
                          int* key2slot) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  FillKey2Slot<<<CUDA_BLOCK(total_len), stream>>>(total_len, slot_lens,
                                                  slot_num, key2slot);
  CopyKeysKernel<<<CUDA_BLOCK(total_len), stream>>>(
      total_len, origin_keys, total_keys, slot_lens, key2slot);
  cudaStreamSynchronize(stream);
}

template <typename FeaturePushValueGpuType>
void FeaturePushCopy(cudaStream_t stream, void* dest, float** grad_values,
                     const int hidden_size, const int embedx_dim,
                     const int total_length, const int batch_size,
                     const int* slot_vector, const int* total_dims,
                     const int64_t* slot_lens, const int slot_num,
                     const int* key2slot, const int cvm_offset,
                     const uint32_t* d_sort_idx, const uint32_t* d_sort_offset,
                     const uint32_t* d_sort_lens, const int skip_offset) {
  FeaturePushValueGpuType* push_grad_values =
      reinterpret_cast<FeaturePushValueGpuType*>(dest);
  if (d_sort_idx != nullptr) {
    // normal
    PushMergeCopyBase<
        FeaturePushValueGpuType><<<CUDA_BLOCK(total_length), stream>>>(
        push_grad_values, grad_values, hidden_size, total_length, batch_size,
        slot_vector, total_dims, slot_lens, slot_num, key2slot, cvm_offset,
        d_sort_idx, d_sort_offset, d_sort_lens, skip_offset);
    if (embedx_dim == 0) {
      return;
    }
    // normal
    int embedx_total_length = total_length * embedx_dim;
    PushMergeCopyExpand<
        FeaturePushValueGpuType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        push_grad_values, grad_values, embedx_dim, embedx_total_length,
        batch_size, slot_vector, total_dims, slot_lens, slot_num, key2slot,
        cvm_offset, d_sort_idx, d_sort_offset, d_sort_lens);
  } else {
    // normal
    PushCopyBase<FeaturePushValueGpuType><<<CUDA_BLOCK(total_length), stream>>>(
        push_grad_values, grad_values, hidden_size, total_length, batch_size,
        slot_vector, total_dims, slot_lens, slot_num, key2slot, cvm_offset,
        skip_offset);
    if (embedx_dim == 0) {
      return;
    }
    // normal
    int embedx_total_length = total_length * embedx_dim;
    PushCopyExpand<
        FeaturePushValueGpuType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        push_grad_values, grad_values, embedx_dim, embedx_total_length,
        batch_size, slot_vector, total_dims, slot_lens, slot_num, key2slot,
        cvm_offset);
  }
}

template <typename FeaturePushValueGpuType>
void FeaturePushCopyNNCross(
    cudaStream_t stream, void* dest, float** grad_values, const int hidden_size,
    const int embedx_dim, const int expand_dim, const int total_length,
    const int batch_size, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_lens,
    const int skip_offset) {
  FeaturePushValueGpuType* push_grad_values =
      reinterpret_cast<FeaturePushValueGpuType*>(dest);
  if (d_sort_idx != nullptr) {
    // nncross
    PushMergeCopyBaseNNCross<
        FeaturePushValueGpuType><<<CUDA_BLOCK(total_length), stream>>>(
        push_grad_values, grad_values, hidden_size, total_length, batch_size,
        slot_vector, total_dims, slot_lens, slot_num, key2slot, cvm_offset,
        d_sort_idx, d_sort_offset, d_sort_lens, skip_offset);
    int embedx_total_length = total_length * (embedx_dim + expand_dim);
    PushMergeCopyExpandNNCross<
        FeaturePushValueGpuType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        push_grad_values, grad_values, (embedx_dim + expand_dim), embedx_dim,
        expand_dim, embedx_total_length, batch_size, slot_vector, total_dims,
        slot_lens, slot_num, key2slot, cvm_offset, d_sort_idx, d_sort_offset,
        d_sort_lens);
  } else {
    // nncross
    PushCopyBaseNNCross<
        FeaturePushValueGpuType><<<CUDA_BLOCK(total_length), stream>>>(
        push_grad_values, grad_values, hidden_size, total_length, batch_size,
        slot_vector, total_dims, slot_lens, slot_num, key2slot, cvm_offset,
        skip_offset);
    int embedx_total_length = total_length * (embedx_dim + expand_dim);
    PushCopyExpandNNCross<
        FeaturePushValueGpuType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        push_grad_values, grad_values, (embedx_dim + expand_dim), embedx_dim,
        expand_dim, embedx_total_length, batch_size, slot_vector, total_dims,
        slot_lens, slot_num, key2slot, cvm_offset);
  }
}

template <typename FeaturePushValueGpuType>
void FeaturePushCopyShareEmbedding(
    cudaStream_t stream, void* dest, float** grad_values, const int hidden_size,
    const size_t embedx_dim, const int total_length, const int batch_size,
    const int* slot_vector, const int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int cvm_offset,
    const uint32_t* d_sort_idx, const uint32_t* d_sort_offset,
    const uint32_t* d_sort_lens, const int skip_offset) {
  FeaturePushValueGpuType* push_grad_values =
      reinterpret_cast<FeaturePushValueGpuType*>(dest);
  if (d_sort_idx != nullptr) {
    // share embedding
    PushMergeCopyBaseShareEmbedding<
        FeaturePushValueGpuType><<<CUDA_BLOCK(total_length), stream>>>(
        push_grad_values, grad_values, hidden_size, total_length, batch_size,
        slot_vector, total_dims, slot_lens, slot_num, key2slot, cvm_offset,
        d_sort_idx, d_sort_offset, d_sort_lens, skip_offset);
    int embedx_total_length = total_length * embedx_dim;
    PushMergeCopyExpand<
        FeaturePushValueGpuType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        push_grad_values, grad_values, embedx_dim, embedx_total_length,
        batch_size, slot_vector, total_dims, slot_lens, slot_num, key2slot,
        cvm_offset, d_sort_idx, d_sort_offset, d_sort_lens);
  } else {
    // share embedding
    PushCopyBaseShareEmbedding<
        FeaturePushValueGpuType><<<CUDA_BLOCK(total_length), stream>>>(
        push_grad_values, grad_values, hidden_size, total_length, batch_size,
        slot_vector, total_dims, slot_lens, slot_num, key2slot, cvm_offset,
        skip_offset);
    int embedx_total_length = total_length * embedx_dim;
    PushCopyExpand<
        FeaturePushValueGpuType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        push_grad_values, grad_values, embedx_dim, embedx_total_length,
        batch_size, slot_vector, total_dims, slot_lens, slot_num, key2slot,
        cvm_offset);
  }
}
template <typename FeaturePushValueGpuType>
void FeaturePushCopyVariable(
    cudaStream_t stream, void* dest, float** grad_values, const int hidden_size,
    const int embedx_dim, const int expand_dim, const int total_length,
    const int batch_size, const int* slot_vector, const int* total_dims,
    const int64_t* slot_lens, const int slot_num, const int* key2slot,
    const int cvm_offset, const uint32_t* d_sort_idx,
    const uint32_t* d_sort_offset, const uint32_t* d_sort_lens,
    const int skip_offset) {
  FeaturePushValueGpuType* push_grad_values =
      reinterpret_cast<FeaturePushValueGpuType*>(dest);
  if (d_sort_idx != nullptr) {
    PushMergeCopyBaseVariable<
        FeaturePushValueGpuType><<<CUDA_BLOCK(total_length), stream>>>(
        push_grad_values, grad_values, hidden_size, expand_dim + cvm_offset,
        total_length, batch_size, slot_vector, total_dims, slot_lens, slot_num,
        key2slot, cvm_offset, d_sort_idx, d_sort_offset, d_sort_lens,
        skip_offset);

    int max_embedx_dim = (embedx_dim > expand_dim) ? embedx_dim : expand_dim;
    int embedx_total_length = total_length * max_embedx_dim;
    PushMergeCopyExpandVariable<
        FeaturePushValueGpuType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        push_grad_values, grad_values, max_embedx_dim, embedx_dim, expand_dim,
        embedx_total_length, batch_size, slot_vector, total_dims, slot_lens,
        slot_num, key2slot, cvm_offset, d_sort_idx, d_sort_offset, d_sort_lens);
  } else {
    PushCopyBaseVariable<
        FeaturePushValueGpuType><<<CUDA_BLOCK(total_length), stream>>>(
        push_grad_values, grad_values, hidden_size, expand_dim + cvm_offset,
        total_length, batch_size, slot_vector, total_dims, slot_lens, slot_num,
        key2slot, cvm_offset, skip_offset);

    int max_embedx_dim = (embedx_dim > expand_dim) ? embedx_dim : expand_dim;
    int embedx_total_length = total_length * max_embedx_dim;
    PushCopyExpandVariable<
        FeaturePushValueGpuType><<<CUDA_BLOCK(embedx_total_length), stream>>>(
        push_grad_values, grad_values, max_embedx_dim, embedx_dim, expand_dim,
        embedx_total_length, batch_size, slot_vector, total_dims, slot_lens,
        slot_num, key2slot, cvm_offset);
  }
}
void BoxWrapper::CopyForPush(
    const paddle::platform::Place& place, float** grad_values,
    void* total_grad_values_gpu, const int* d_slot_vector,
    const int64_t* slot_lens, const int slot_num, const int hidden_size,
    const int expand_embed_dim, const int64_t total_length,
    const int batch_size, const int* total_dims, const int* key2slot,
    const int skip_offset, const uint32_t* gpu_sort_idx,
    const uint32_t* gpu_sort_offset, const uint32_t* gpu_sort_lens) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  const int cvm_offset = cvm_offset_ - skip_offset;
#define EMBEDX_CASE(i, ...)                                                  \
  case i: {                                                                  \
    constexpr size_t EmbedxDim = i;                                          \
    switch (expand_embed_dim_) {                                             \
      __VA_ARGS__                                                            \
      default:                                                               \
        PADDLE_THROW(platform::errors::InvalidArgument(                      \
            "Unsupport this expand embedding size [%d]", expand_embed_dim)); \
    }                                                                        \
  } break

#define EXPAND_EMBED_PUSH_CASE(i, ...)                                        \
  case i: {                                                                   \
    constexpr size_t ExpandDim = i;                                           \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {             \
      FeaturePushCopy<boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>>(  \
          stream, total_grad_values_gpu, grad_values, hidden_size, EmbedxDim, \
          total_length, batch_size, d_slot_vector, total_dims, slot_lens,     \
          slot_num, key2slot, cvm_offset, gpu_sort_idx, gpu_sort_offset,      \
          gpu_sort_lens, skip_offset);                                        \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_CONV)) {      \
      FeaturePushCopy<boxps::FeaturePushValueGpuConv<EmbedxDim, ExpandDim>>(  \
          stream, total_grad_values_gpu, grad_values, hidden_size, EmbedxDim, \
          total_length, batch_size, d_slot_vector, total_dims, slot_lens,     \
          slot_num, key2slot, cvm_offset, gpu_sort_idx, gpu_sort_offset,      \
          gpu_sort_lens, skip_offset);                                        \
    } else {                                                                  \
      FeaturePushCopy<boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>>(      \
          stream, total_grad_values_gpu, grad_values, hidden_size, EmbedxDim, \
          total_length, batch_size, d_slot_vector, total_dims, slot_lens,     \
          slot_num, key2slot, cvm_offset, gpu_sort_idx, gpu_sort_offset,      \
          gpu_sort_lens, skip_offset);                                        \
    }                                                                         \
  } break

#define EXPAND_EMBED_PUSH_NNCROSS(i, ...)                                     \
  case i: {                                                                   \
    constexpr size_t ExpandDim = i;                                           \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_PCOC)) {             \
      FeaturePushCopyNNCross<                                                 \
          boxps::FeaturePushValueGpuPCOC<EmbedxDim, ExpandDim>>(              \
          stream, total_grad_values_gpu, grad_values, hidden_size, EmbedxDim, \
          ExpandDim, total_length, batch_size, d_slot_vector, total_dims,     \
          slot_lens, slot_num, key2slot, cvm_offset, gpu_sort_idx,            \
          gpu_sort_offset, gpu_sort_lens, skip_offset);                       \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_VARIABLE)) {  \
      FeaturePushCopyVariable<                                                \
          boxps::FeatureVarPushValueGpu<EmbedxDim, ExpandDim>>(               \
          stream, total_grad_values_gpu, grad_values, hidden_size, EmbedxDim, \
          ExpandDim, total_length, batch_size, d_slot_vector, total_dims,     \
          slot_lens, slot_num, key2slot, cvm_offset, gpu_sort_idx,            \
          gpu_sort_offset, gpu_sort_lens, skip_offset);                       \
    } else if (feature_type_ == static_cast<int>(boxps::FEATURE_CONV)) {      \
      FeaturePushCopyVariable<                                                \
          boxps::FeaturePushValueGpuConv<EmbedxDim, ExpandDim>>(              \
          stream, total_grad_values_gpu, grad_values, hidden_size, EmbedxDim, \
          ExpandDim, total_length, batch_size, d_slot_vector, total_dims,     \
          slot_lens, slot_num, key2slot, cvm_offset, gpu_sort_idx,            \
          gpu_sort_offset, gpu_sort_lens, skip_offset);                       \
    } else {                                                                  \
      FeaturePushCopyNNCross<                                                 \
          boxps::FeaturePushValueGpu<EmbedxDim, ExpandDim>>(                  \
          stream, total_grad_values_gpu, grad_values, hidden_size, EmbedxDim, \
          ExpandDim, total_length, batch_size, d_slot_vector, total_dims,     \
          slot_lens, slot_num, key2slot, cvm_offset, gpu_sort_idx,            \
          gpu_sort_offset, gpu_sort_lens, skip_offset);                       \
    }                                                                         \
  } break

#define EXPAND_EMBED_PUSH_SHARE(i, ...)                                        \
  case i: {                                                                    \
    constexpr size_t ExpandDim = i;                                            \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_SHARE_EMBEDDING)) {   \
      FeaturePushCopyShareEmbedding<                                           \
          boxps::FeaturePushValueGpuShareEmbedding<EmbedxDim, ExpandDim>>(     \
          stream, total_grad_values_gpu, grad_values, hidden_size,             \
          (hidden_size - cvm_offset), total_length, batch_size, d_slot_vector, \
          total_dims, slot_lens, slot_num, key2slot, cvm_offset, gpu_sort_idx, \
          gpu_sort_offset, gpu_sort_lens, skip_offset);                        \
    }                                                                          \
  } break

#define EXPAND_EMBED_PUSH_ADAMEX(i, ...)                                      \
  case i: {                                                                   \
    constexpr size_t ExpandDim = i;                                           \
    if (feature_type_ == static_cast<int>(boxps::FEATURE_ADAM)) {             \
      FeaturePushCopy<boxps::FeatureVarPushValueGpu<EmbedxDim, ExpandDim>>(   \
          stream, total_grad_values_gpu, grad_values, hidden_size, ExpandDim, \
          total_length, batch_size, d_slot_vector, total_dims, slot_lens,     \
          slot_num, key2slot, cvm_offset, gpu_sort_idx, gpu_sort_offset,      \
          gpu_sort_lens, skip_offset);                                        \
    }                                                                         \
  } break

  switch (embedx_dim_) {
    EMBEDX_CASE(0, EXPAND_EMBED_PUSH_CASE(0); EXPAND_EMBED_PUSH_ADAMEX(255);
                EXPAND_EMBED_PUSH_ADAMEX(767););
    EMBEDX_CASE(2, EXPAND_EMBED_PUSH_CASE(0););
    EMBEDX_CASE(4, EXPAND_EMBED_PUSH_CASE(0););
    EMBEDX_CASE(8, EXPAND_EMBED_PUSH_CASE(0); EXPAND_EMBED_PUSH_SHARE(1);
                EXPAND_EMBED_PUSH_SHARE(2); EXPAND_EMBED_PUSH_SHARE(3);
                EXPAND_EMBED_PUSH_SHARE(4); EXPAND_EMBED_PUSH_SHARE(5);
                EXPAND_EMBED_PUSH_SHARE(6); EXPAND_EMBED_PUSH_SHARE(7);
                EXPAND_EMBED_PUSH_SHARE(8); EXPAND_EMBED_PUSH_NNCROSS(64););
    EMBEDX_CASE(16, EXPAND_EMBED_PUSH_CASE(0); EXPAND_EMBED_PUSH_SHARE(1);
                EXPAND_EMBED_PUSH_SHARE(2); EXPAND_EMBED_PUSH_SHARE(3);
                EXPAND_EMBED_PUSH_SHARE(4); EXPAND_EMBED_PUSH_SHARE(5);
                EXPAND_EMBED_PUSH_SHARE(6); EXPAND_EMBED_PUSH_SHARE(7);
                EXPAND_EMBED_PUSH_SHARE(8); EXPAND_EMBED_PUSH_NNCROSS(64););
    EMBEDX_CASE(32, EXPAND_EMBED_PUSH_CASE(0););
    EMBEDX_CASE(64, EXPAND_EMBED_PUSH_CASE(0););
    EMBEDX_CASE(256, EXPAND_EMBED_PUSH_CASE(0););
    EMBEDX_CASE(128, EXPAND_EMBED_PUSH_CASE(0););
    EMBEDX_CASE(280, EXPAND_EMBED_PUSH_CASE(0););
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupport this embedding size [%d]", hidden_size - cvm_offset));
  }

  cudaStreamSynchronize(stream);
#undef EXPAND_EMBED_PUSH_CASE
#undef EMBEDX_CASE
}

void BasicAucCalculator::cuda_add_data(const paddle::platform::Place& place,
                                       const int64_t* label, const float* pred,
                                       int len) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();

  int i = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();

  cudaSetDevice(i);

  AddBasicCalculator<<<CUDA_BLOCK(len), stream>>>(
      pred, label, reinterpret_cast<double*>(_d_positive[i]->ptr()),
      reinterpret_cast<double*>(_d_negative[i]->ptr()),
      reinterpret_cast<double*>(_d_abserr[i]->ptr()),
      reinterpret_cast<double*>(_d_sqrerr[i]->ptr()),
      reinterpret_cast<double*>(_d_pred[i]->ptr()), len, _table_size);
}

void BasicAucCalculator::cuda_add_mask_data(
    const paddle::platform::Place& place, const int64_t* label,
    const float* pred, const int64_t* mask, int len) {
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(
                        BOOST_GET_CONST(platform::CUDAPlace, place)))
                    ->stream();
  int i = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();

  cudaSetDevice(i);

  AddMaskCalculator<<<CUDA_BLOCK(len), stream>>>(
      pred, label, mask, reinterpret_cast<double*>(_d_positive[i]->ptr()),
      reinterpret_cast<double*>(_d_negative[i]->ptr()),
      reinterpret_cast<double*>(_d_abserr[i]->ptr()),
      reinterpret_cast<double*>(_d_sqrerr[i]->ptr()),
      reinterpret_cast<double*>(_d_pred[i]->ptr()), len, _table_size);
}

__global__ void pull_cache_value_kernel(int len, int dim, uint64_t* key,
                                        float* val, float* table) {
  CUDA_KERNEL_LOOP(i, len) { val[i] = table[key[i / dim] * dim + i % dim]; }
}

void GpuReplicaCache::PullCacheValue(uint64_t* d_keys, float* d_vals, int num,
                                     int gpu_id) {
  auto place = platform::CUDAPlace(gpu_id);
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  int len = emb_dim_ * num;
  pull_cache_value_kernel<<<CUDA_BLOCK(len), stream>>>(len, emb_dim_, d_keys,
                                                       d_vals, d_embs_[gpu_id]);
}

}  // end namespace framework
}  // end namespace paddle
#endif
