/*
 Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 *
 *  Created on: 2022年6月9日
 *      Author: humingqing
 */
#ifdef PADDLE_WITH_BOX_PS
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include <memory>
#include "paddle/fluid/framework/fleet/box_wrapper_impl.h"
#include "paddle/fluid/framework/threadpool.h"

namespace paddle {
namespace framework {
// copy cpu keys
void BoxWrapper::CopyCPUKeys(const paddle::platform::Place& place,
                             const std::vector<const uint64_t*>& origin_keys,
                             uint64_t* total_keys,
                             const int64_t* slot_lengths_lod, int slot_num,
                             int total_len, int* key2slot) {
  this->ExecuteFunc(place, slot_num, [&](const size_t& slot_id) {
    const uint64_t* slot_keys = origin_keys[slot_id];
    const int64_t& start = slot_lengths_lod[slot_id];
    const int64_t& end = slot_lengths_lod[slot_id + 1];
    for (int64_t i = start; i < end; ++i) {
      total_keys[i] = slot_keys[i - start];
      key2slot[i] = slot_id;
    }
  });
}
inline void FeaturePullCopyData(
    const paddle::platform::Place& place, const boxps::FeaturePullOffset& info,
    const size_t& pull_size_float,
    const std::vector<const uint64_t*>& slot_keys,
    const std::vector<float*>& dest, const float* src, const int hidden_size,
    const uint32_t& embedx_dim, const int total_length, const int dedup_len,
    int* total_dims, const int64_t* slot_lens, const int slot_num,
    const int* key2slot, const float scale, const int cvm_offset,
    const uint32_t* restore_idx, const int skip_offset) {
  BoxWrapper::GetInstance()->ExecRangeFunc(
      place, total_length, [&](const size_t& start, const size_t& end) {
        for (size_t i = start; i < end; ++i) {
          int x = key2slot[i];
          int y = i - slot_lens[x];
          const uint32_t& src_id = restore_idx[i];
          CHECK(src_id < static_cast<uint32_t>(dedup_len))
              << "i=" << i << ", dedup_len=" << dedup_len
              << ", src_id=" << src_id;
          const float* src_val = &src[src_id * pull_size_float];
          float* dest_ptr = dest[x] + y * hidden_size;
          const float* src_ptr =
              reinterpret_cast<const float*>(&src_val[info.show]);
          for (int k = 0; k < cvm_offset; ++k) {
            dest_ptr[k] = src_ptr[k + skip_offset];
          }
          const uint32_t embedx_size =
              *reinterpret_cast<const uint32_t*>(&src_val[info.embedx_size]);
          total_dims[i] = static_cast<int>(embedx_size > 0);
          dest_ptr = dest_ptr + cvm_offset;
          if (embedx_size == 0 || embedx_size > embedx_dim) {
            for (uint32_t col = 0; col < embedx_dim; ++col) {
              dest_ptr[col] = 0;
            }
          } else {
            src_ptr = &src_val[info.embedx];
            if (info.is_quant) {
              const int16_t* embedx_ptr =
                  reinterpret_cast<const int16_t*>(src_ptr);
              // copy embedx
              for (uint32_t col = 0; col < embedx_size; ++col) {
                dest_ptr[col] = embedx_ptr[col] * scale;
              }
            } else {
              // copy embedx
              for (uint32_t col = 0; col < embedx_size; ++col) {
                dest_ptr[col] = src_ptr[col];
              }
            }
            if (embedx_size < embedx_dim) {
              for (uint32_t col = embedx_size; col < embedx_dim; ++col) {
                dest_ptr[col] = 0;
              }
            }
          }
        }
      });
}
// cpu copy values
void BoxWrapper::CopyForPullCPU(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& slot_keys,
                                const std::vector<float*>& slot_values,
                                void* total_values_gpu,
                                const int64_t* slot_lens, const int slot_num,
                                const int* key2slot, const int hidden_size,
                                const int expand_embed_dim,
                                const int64_t total_length, int* total_dims,
                                const int skip_offset, bool expand_only,
                                const uint32_t* restore_idx) {
  int device_id = GetPlaceDeviceId(place);
  int dedup_len = device_caches_[device_id].dedup_key_length;
  const int cvm_offset = cvm_offset_ - skip_offset;
  float* pull_values_gpu = reinterpret_cast<float*>(total_values_gpu);
  FeaturePullCopyData(place, pull_info_, pull_float_num_, slot_keys,
                      slot_values, pull_values_gpu, hidden_size, embedx_dim_,
                      total_length, dedup_len, total_dims, slot_lens, slot_num,
                      key2slot, pull_embedx_scale_, cvm_offset, restore_idx,
                      skip_offset);
}
inline void PushMergeCopyData(
    const paddle::platform::Place& place, const boxps::FeaturePushOffset& info,
    const size_t& push_float_num, float* dest,
    const std::vector<const float*>& src, const int hidden,
    const int embedx_dim, const int total_len, const int bs,
    const int* slot_vector, const int* total_dims, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int cvm_offset,
    const uint32_t* d_sort_idx, const uint32_t* d_sort_offset,
    const uint32_t* d_sort_cnt, const int skip_offset) {
  BoxWrapper::GetInstance()->ExecRangeFunc(
      place, total_len, [&](const size_t& start, const size_t& end) {
        for (size_t i = start; i < end; ++i) {
          const uint32_t& start = d_sort_offset[i];
          const uint32_t& count = d_sort_cnt[i];
          const uint32_t& pos = d_sort_idx[start];

          const int& x = key2slot[pos];
          int y = pos - slot_lens[x];

          float* dest_val = &dest[i * push_float_num];
          (*reinterpret_cast<int*>(&dest_val[info.slot])) = slot_vector[x];
          float* optr = reinterpret_cast<float*>(&dest_val[info.show]);
          const float* src_val =
              reinterpret_cast<const float*>(src[x] + y * hidden);
          for (int k = 0; k < skip_offset; ++k) {
            optr[k] = count;
          }
          for (int k = 0; k < cvm_offset; ++k) {
            optr[k + skip_offset] = src_val[k];
          }
          for (int col = 0; col < embedx_dim; ++col) {
            dest_val[info.embedx_g + col] = src_val[cvm_offset + col];
          }
          // merge same key in diffent slot id
          for (uint32_t j = 1; j < count; ++j) {
            const uint32_t& pos = d_sort_idx[start + j];
            const int& x = key2slot[pos];
            y = pos - slot_lens[x];
            src_val = reinterpret_cast<const float*>(src[x] + y * hidden);
            for (int k = 0; k < cvm_offset; ++k) {
              optr[k + skip_offset] += src_val[k];
            }
            // add embedx
            for (int col = 0; col < embedx_dim; ++col) {
              dest_val[info.embedx_g + col] += src_val[cvm_offset + col];
            }
          }
          for (int k = 0; k < info.embed_num; ++k) {
            dest_val[info.embed_g + k] *= -1. * bs;
          }
          for (int col = 0; col < embedx_dim; ++col) {
            dest_val[info.embedx_g + col] *= -1. * bs;
          }
        }
      });
}
void BoxWrapper::CopyForPushCPU(
    const paddle::platform::Place& place,
    const std::vector<const float*>& slot_grad_values,
    void* total_grad_values_gpu, const int* slots, const int64_t* slot_lens,
    const int slot_num, const int hidden_size, const int expand_embed_dim,
    const int64_t total_length, const int batch_size, const int* total_dims,
    const int* key2slot, const int skip_offset, bool expand_only,
    const uint32_t* sort_idx, const uint32_t* sort_offset,
    const uint32_t* sort_lens) {
  const int cvm_offset = cvm_offset_ - skip_offset;
  float* push_grad_values = reinterpret_cast<float*>(total_grad_values_gpu);
  PushMergeCopyData(place, push_info_, push_float_num_, push_grad_values,
                    slot_grad_values, hidden_size, embedx_dim_, total_length,
                    batch_size, slot_vector_.data(), total_dims, slot_lens,
                    slot_num, key2slot, cvm_offset, sort_idx, sort_offset,
                    sort_lens, skip_offset);
}

}  // end namespace framework
}  // end namespace paddle
#endif
