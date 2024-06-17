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
#pragma once

// #ifdef PADDLE_WITH_XPU_KP
#if (defined PADDLE_WITH_XPU_KP) && (defined PADDLE_WITH_BOX_PS)
#include <boxps_public.h>
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

class BoxWrapperKernel {
public:
void GetFeatureInfo(boxps::FeaturePullOffset &pull_info,
    size_t feature_pull_size, boxps::FeaturePushOffset &push_info,
    size_t feature_push_size, int embedx_dim, int expand_embed_dim,
    float pull_embedx_scale);

void CopyKeys(const paddle::platform::Place& place,
                            uint64_t** origin_keys, uint32_t* total_keys,
                            const int64_t* xpu_len, int slot_num,
                            int total_len, int* key2slot);
void CopyKeys(const paddle::platform::Place& place,
                            uint64_t** origin_keys, unsigned long long* total_keys,
                            const int64_t* xpu_len, int slot_num,
                            int total_len, int* key2slot);

void CopyForPull(
    const paddle::platform::Place& place, uint64_t** xpu_keys,
    float** xpu_values, void* total_values_xpu,
    boxps::FeaturePullOffset* pull_offset, const int64_t* slot_lens,
    const int slot_num, const int* key2slot, const int* d_res_idx, const int hidden_size,
    const int expand_embed_dim, const int64_t total_length, int* total_dims,
    const int skip_offset, bool expand_only,
    const int* xpu_merged_idx = nullptr, 
    const int* xpu_merged_offsets = nullptr, 
    const int merged_len = 0);

void CopyForPush(
    const paddle::platform::Place& place,
    float** gm_src_ptr,
    void* total_grad_values_xpu,
    boxps::FeaturePushOffset* push_offset,
    const int64_t total_length,
    const int* slots,
    const int* slot_inner_offset,
    const int64_t* slot_len_lod,
    const int slot_num,
    const int hidden_size,
    const int batch_size,
    const int* total_dims,
    const int skip_offset,
    const int* key2slot,
    const int expand_embed_dim,
    const int push_float_num,
    bool expand_only,
    bool is_continuous);

public:
  const static int MAX_SLOT_SIZE = 10240;

private:
  size_t feature_pull_size_ = 0;
  size_t feature_push_size_ = 0;
  boxps::FeaturePullOffset pull_info_;
  boxps::FeaturePushOffset push_info_;
  size_t pull_float_num_ = 0;
  size_t push_float_num_ = 0;

  int embedx_dim_ = 8;
  int expand_embed_dim_ = 0;
  int feature_type_ = 0;
  float pull_embedx_scale_ = 1.0;
  int cvm_offset_ = 3;
};
}
}

#endif
