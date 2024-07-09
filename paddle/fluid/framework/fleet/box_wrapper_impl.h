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
#ifdef PADDLE_WITH_BOX_PS
#include <glog/logging.h>

#include <vector>
#include <thread>

#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
// The producer side.
#include <scalopus_tracing/tracing.h>
#include <scalopus_transport/transport_loopback.h>
// The catapult recorder side.
#include <scalopus_catapult/catapult_recorder.h>
#include <scalopus_general/endpoint_manager_poll.h>
#include <scalopus_general/general_provider.h>
#include <scalopus_tracing/native_trace_provider.h>
#endif

DECLARE_bool(enable_pullpush_dedup_keys);

namespace paddle {
namespace framework {

void BoxWrapper::PullSparseCaseGPU(const paddle::platform::Place& place,
                                   const std::vector<const uint64_t*>& keys,
                                   const std::vector<float*>& values,
                                   const std::vector<int64_t>& slot_lengths,
                                   const int hidden_size,
                                   const int expand_embed_dim,
                                   const int skip_offset,
                                   bool expand_only) {
#if defined(PADDLE_WITH_CUDA)
  //  VLOG(3) << "Begin PullSparse";
  int device_id = place.GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_pull_timer;
  platform::Timer& pull_boxps_timer = dev.boxps_pull_timer;
  platform::Timer& pull_dedup_timer = dev.pull_dedup_timer;
  all_timer.Resume();

  // construct slot_level lod info
  std::vector<int64_t> slot_lengths_lod;
  slot_lengths_lod.push_back(0);

  int64_t total_length = 0;
  int slot_num = static_cast<int>(slot_lengths.size());
  for (int i = 0; i < slot_num; i++) {
    total_length += slot_lengths[i];
    slot_lengths_lod.push_back(total_length);
  }
  dev.total_key_length = total_length;

  auto stream = dynamic_cast<phi::GPUContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();

  boxps::FeaturePullOffset* pull_offset = nullptr;
  if (dev.pull_offset.memory_size() == 0) {
    pull_offset = dev.pull_offset.mutable_data<boxps::FeaturePullOffset>(
        sizeof(boxps::FeaturePullOffset), place);
    cudaMemcpyAsync(pull_offset,
                    &pull_info_,
                    sizeof(boxps::FeaturePullOffset),
                    cudaMemcpyHostToDevice,
                    stream);
  } else {
    pull_offset = dev.pull_offset.data<boxps::FeaturePullOffset>();
  }

  uint64_t* total_keys = nullptr;
  int* key2slot = nullptr;
  if (FLAGS_enable_pullpush_dedup_keys) {
    total_keys = dev.keys_tensor.mutable_data<uint64_t>(
        static_cast<int64_t>(total_length * 2 * sizeof(int64_t)), place);
    key2slot = dev.keys2slot.mutable_data<int>(
        static_cast<int64_t>(total_length * 5) * sizeof(int), place);
  } else {
    total_keys = dev.keys_tensor.mutable_data<uint64_t>(
        total_length * sizeof(int64_t), place);
    key2slot =
        dev.keys2slot.mutable_data<int>(total_length * sizeof(int), place);
  }

  int* total_dims =
      dev.dims_tensor.mutable_data<int>(total_length * sizeof(int), place);

  uint64_t** gpu_keys = dev.keys_ptr_tensor.mutable_data<uint64_t*>(
      static_cast<int>(keys.size() * sizeof(uint64_t*)), place);

  int64_t* slot_lens = dev.slot_lens.mutable_data<int64_t>(
      (slot_num + 1) * sizeof(int64_t), place);

  dev.copy_keys_timer.Resume();
  cudaMemcpyAsync(gpu_keys,
                  keys.data(),
                  keys.size() * sizeof(uint64_t*),
                  cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(slot_lens,
                  slot_lengths_lod.data(),
                  slot_lengths_lod.size() * sizeof(int64_t),
                  cudaMemcpyHostToDevice,
                  stream);
  this->CopyKeys(place,
                 gpu_keys,
                 total_keys,
                 slot_lens,
                 slot_num,
                 static_cast<int>(total_length),
                 key2slot);
  dev.copy_keys_timer.Pause();

  // dedup keys pull
  if (FLAGS_enable_pullpush_dedup_keys) {
    uint32_t* d_restore_idx =
        reinterpret_cast<uint32_t*>(&key2slot[total_length]);
    uint32_t* d_sorted_idx =
        reinterpret_cast<uint32_t*>(&d_restore_idx[total_length]);
    uint32_t* d_offset =
        reinterpret_cast<uint32_t*>(&d_sorted_idx[total_length]);
    uint32_t* d_merged_cnts =
        reinterpret_cast<uint32_t*>(&d_offset[total_length]);
    uint64_t* d_merged_keys =
        reinterpret_cast<uint64_t*>(&total_keys[total_length]);

    pull_dedup_timer.Resume();
    int dedup_size =
        boxps_ptr_->DedupKeysAndFillIdx(device_id,
                                        total_length,
                                        total_keys,     // input
                                        d_merged_keys,  // output
                                        d_restore_idx,  // pull fill idx
                                        d_sorted_idx,   // sort old idx
                                        d_offset,       // offset
                                        d_merged_cnts);
    pull_dedup_timer.Pause();

    PADDLE_ENFORCE_GT(dedup_size,
                      0,
                      platform::errors::PreconditionNotMet(
                          "dedup keys need more than zero failed in BoxPS."));
    dev.dedup_key_length = dedup_size;

    int64_t total_bytes = dedup_size * feature_pull_size_;
    void* total_values_gpu =
        dev.pull_push_tensor.mutable_data<void>(total_bytes, place);

    pull_boxps_timer.Resume();

    int ret =
        boxps_ptr_->PullSparseGPU(d_merged_keys,
                                  reinterpret_cast<void*>(total_values_gpu),
                                  static_cast<int>(dedup_size),
                                  device_id);
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        platform::errors::PreconditionNotMet("PullSparseGPU failed in BoxPS."));
    pull_boxps_timer.Pause();
    // values.size() not sure equal slot_num
    float** gpu_values = dev.values_ptr_tensor.mutable_data<float*>(
        static_cast<int>(values.size() * sizeof(float*)), place);

    dev.copy_values_timer.Resume();
    cudaMemcpyAsync(gpu_values,
                    values.data(),
                    values.size() * sizeof(float*),
                    cudaMemcpyHostToDevice,
                    stream);

    this->CopyForPull(place,
                      gpu_keys,
                      gpu_values,
                      total_values_gpu,
                      pull_offset,
                      slot_lens,
                      slot_num,
                      key2slot,
                      hidden_size,
                      expand_embed_dim,
                      total_length,
                      total_dims,
                      skip_offset,
                      expand_only,
                      d_restore_idx);
    dev.copy_values_timer.Pause();
  } else {
    int64_t total_bytes = total_length * feature_pull_size_;
    void* total_values_gpu =
        dev.pull_push_tensor.mutable_data<void>(total_bytes, place);

    pull_boxps_timer.Resume();

    int ret =
        boxps_ptr_->PullSparseGPU(total_keys,
                                  reinterpret_cast<void*>(total_values_gpu),
                                  static_cast<int>(total_length),
                                  device_id);
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        platform::errors::PreconditionNotMet("PullSparseGPU failed in BoxPS."));
    pull_boxps_timer.Pause();
    // values.size() not sure equal slot_num
    float** gpu_values = dev.values_ptr_tensor.mutable_data<float*>(
        static_cast<int>(values.size() * sizeof(float*)), place);

    dev.copy_values_timer.Resume();
    cudaMemcpyAsync(gpu_values,
                    values.data(),
                    values.size() * sizeof(float*),
                    cudaMemcpyHostToDevice,
                    stream);

    this->CopyForPull(place,
                      gpu_keys,
                      gpu_values,
                      total_values_gpu,
                      pull_offset,
                      slot_lens,
                      slot_num,
                      key2slot,
                      hidden_size,
                      expand_embed_dim,
                      total_length,
                      total_dims,
                      skip_offset,
                      expand_only);
    dev.copy_values_timer.Pause();
  }
  all_timer.Pause();
#endif
}

void BoxWrapper::PullSparseCaseCPU(const paddle::platform::Place& place,
                                   const std::vector<const uint64_t*>& keys,
                                   const std::vector<float*>& values,
                                   const std::vector<int64_t>& slot_lengths,
                                   const int hidden_size,
                                   const int expand_embed_dim,
                                   const int skip_offset,
                                   bool expand_only) {
  //  VLOG(3) << "Begin PullSparse";
  int device_id = GetPlaceDeviceId(place);
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_pull_timer;
  platform::Timer& pull_boxps_timer = dev.boxps_pull_timer;
  platform::Timer& pull_dedup_timer = dev.pull_dedup_timer;
  all_timer.Resume();

  int slot_num = static_cast<int>(slot_lengths.size());

  int64_t* slot_lens = dev.slot_lens.mutable_data<int64_t>(
      (slot_num + 1) * sizeof(int64_t), place);
  int64_t total_length = 0;
  slot_lens[0] = 0;
  for (int i = 0; i < slot_num; i++) {
    total_length += slot_lengths[i];
    slot_lens[i + 1] = total_length;
  }
  dev.total_key_length = total_length;

  uint64_t* total_keys = dev.keys_tensor.mutable_data<uint64_t>(
    static_cast<int64_t>(total_length * 2) * sizeof(int64_t), place);

  int* key2slot = dev.keys2slot.mutable_data<int>(
      static_cast<int64_t>(total_length * 5) * sizeof(int), place);
  int* total_dims =
      dev.dims_tensor.mutable_data<int>(total_length * sizeof(int), place);

  dev.copy_keys_timer.Resume();
  this->CopyCPUKeys(place,
                    keys,
                    total_keys,
                    slot_lens,
                    slot_num,
                    static_cast<int>(total_length),
                    key2slot);
  dev.copy_keys_timer.Pause();

  // dedup keys pull
  uint32_t* d_restore_idx =
      reinterpret_cast<uint32_t*>(&key2slot[total_length]);
  uint32_t* d_sorted_idx =
      reinterpret_cast<uint32_t*>(&d_restore_idx[total_length]);
  uint32_t* d_offset = reinterpret_cast<uint32_t*>(&d_sorted_idx[total_length]);
  uint32_t* d_merged_cnts =
      reinterpret_cast<uint32_t*>(&d_offset[total_length]);
  uint64_t* d_merged_keys =
      reinterpret_cast<uint64_t*>(&total_keys[total_length]);

  pull_dedup_timer.Resume();
  int dedup_size =
      boxps_ptr_->DedupKeysAndFillIdx(device_id,
                                      total_length,
                                      total_keys,     // input
                                      d_merged_keys,  // output
                                      d_restore_idx,  // pull fill idx
                                      d_sorted_idx,   // sort old idx
                                      d_offset,       // offset
                                      d_merged_cnts);
  pull_dedup_timer.Pause();
  PADDLE_ENFORCE_GT(dedup_size,
                    0,
                    platform::errors::PreconditionNotMet(
                        "dedup keys need more than zero failed in BoxPS."));
  dev.dedup_key_length = dedup_size;

  int64_t total_bytes = dedup_size * feature_pull_size_;
  void* total_values_gpu =
      dev.pull_push_tensor.mutable_data<void>(total_bytes, place);

  pull_boxps_timer.Resume();

  int ret = boxps_ptr_->PullSparseGPU(d_merged_keys,
                                      reinterpret_cast<void*>(total_values_gpu),
                                      static_cast<int>(dedup_size),
                                      device_id);
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      platform::errors::PreconditionNotMet("PullSparseGPU failed in BoxPS."));
  pull_boxps_timer.Pause();

  dev.copy_values_timer.Resume();
  this->CopyForPullCPU(place,
                       keys,
                       values,
                       total_values_gpu,
                       slot_lens,
                       slot_num,
                       key2slot,
                       hidden_size,
                       expand_embed_dim,
                       total_length,
                       total_dims,
                       skip_offset,
                       expand_only,
                       d_restore_idx);
  dev.copy_values_timer.Pause();

  all_timer.Pause();
}

#ifdef PADDLE_WITH_XPU_KP
void CheckValPtr(
    int dev_id,
    const std::vector<float*>& values,
    const std::vector<int64_t>& slot_lengths,
    uint32_t hidden_size,
    int expand_embed_dim,
    std::string prefix,
    int err_idx) {
  int slot_num = slot_lengths.size();
  for (uint32_t j = 0; j < slot_lengths.size(); ++j) {
    fprintf(stderr,
            "[%s-erridx:%d] dev: %d, pull_copy values[%d]:%p, slot_lengths[%d]:%d, " \
            "next_prt: %p, expand_grad_values[%d]:%p, next_prt: %p\n",
            prefix.c_str(), err_idx, dev_id, j, values[j],
            j, (int)slot_lengths[j],
            values[j] + slot_lengths[j] * hidden_size,
            j + slot_num, values[j + slot_num],
            values[j + slot_num] + slot_lengths[j] * expand_embed_dim);
  }
}

void CheckPullValue(
    int dev_id,
    const std::vector<float*>& values,
    const std::vector<int64_t>& slot_lengths,
    boxps::FeaturePullOffset * pull_info,
    uint32_t hidden_size,
    int cvm_offset,
    int expand_embed_dim,
    int * total_dims) {
  int val_len = 0;
  for (uint32_t i = 0; i < slot_lengths.size(); ++i) {
    val_len += slot_lengths[i];
  }
  int fixed_float_len = val_len * hidden_size;
  std::vector<float> h_values(fixed_float_len);
  int expand_float_len = val_len * expand_embed_dim;
  std::vector<float> h_expand_values(expand_float_len);
  if (expand_float_len > 0) {
    memset(&(h_expand_values[0]), 0, sizeof(float) * expand_float_len);
  }
  std::vector<int> h_total_dims(val_len);
  int copy_dim_ret = xpu_memcpy(&(h_total_dims[0]), total_dims, sizeof(int) * val_len, XPU_DEVICE_TO_HOST);
  PADDLE_ENFORCE_EQ(
       copy_dim_ret, 0,
       platform::errors::PreconditionNotMet("CheckPullValue copy total_dim error."));

  std::vector<int> val_2_slot(val_len);

  int offset = 0;
  int xpu_ret = 0;
  if (expand_embed_dim > 0) {
    PADDLE_ENFORCE_EQ(
        slot_lengths.size() * 2, values.size(),
        platform::errors::PreconditionNotMet("CheckPullValue slot_length vs values.size error."));
  } else {
    PADDLE_ENFORCE_EQ(
        slot_lengths.size(), values.size(),
        platform::errors::PreconditionNotMet("CheckPullValue slot_length vs values.size error."));
  }
  int slot_num = slot_lengths.size();
  for (int i = 0; i < slot_num; ++i) {
    if (values[i] == nullptr) {
      if (slot_lengths[i] != 0) {
        VLOG(0) << "CheckPullValue found slot[" << i << "] dval_ptr is null, while slot_length != 0:" << slot_lengths[i];
      }
      PADDLE_ENFORCE_EQ(
          slot_lengths[i], 0,
          platform::errors::PreconditionNotMet("CheckPullValue slot_length error."));
          continue;
    }
    if (slot_lengths[i] == 0) {
      continue;
    }
    for (int j = 0; j < slot_lengths[i]; ++j) {
        val_2_slot[offset + j] = i;
    }
    // copy show/clk/embed/embedx
    xpu_ret = xpu_memcpy(&(h_values[offset * hidden_size]), values[i],
        sizeof(float) * hidden_size * slot_lengths[i], XPU_DEVICE_TO_HOST);
    if (xpu_ret != 0) {
      VLOG(0) << "CheckPullValue xpu_memcpy for emb error: hidden_size:" << hidden_size
              << ", cvm_offset:" << cvm_offset << ", errslotidx:" << i;
      CheckValPtr(dev_id, values, slot_lengths, hidden_size, expand_embed_dim, "copyemberr", i);
    }
    PADDLE_ENFORCE_EQ(xpu_ret, 0,
      platform::errors::PreconditionNotMet("CheckPullValue xpu_memcpy for emb error."));

    // copy expand emb
    if (expand_embed_dim > 0 && values[i + slot_num] != nullptr) {
      xpu_ret = xpu_memcpy(&(h_expand_values[offset * expand_embed_dim]), values[i + slot_num],
          sizeof(float) * expand_embed_dim * slot_lengths[i], XPU_DEVICE_TO_HOST);
      PADDLE_ENFORCE_EQ(xpu_ret, 0,
          platform::errors::PreconditionNotMet("CheckPullValue xpu_memcpy error."));
    }
    if (xpu_ret != 0) {
      VLOG(0) << "CheckPullValue xpu_memcpy for expand error: hidden_size:" << hidden_size
              << ", cvm_offset:" << cvm_offset << ", errslotidx:" << i;
      CheckValPtr(dev_id, values, slot_lengths, hidden_size, expand_embed_dim, "copyexpanderr", i);
    }
    PADDLE_ENFORCE_EQ(xpu_ret, 0,
        platform::errors::PreconditionNotMet("CheckPullValue xpu_memcpy for expand error."));
    offset += slot_lengths[i];
  }
  int ret = 0;
  for (int i = 0; i < val_len; ++i) {
    for (int j = 0; j < cvm_offset - 1; ++j) {
      float & v = h_values[i * hidden_size + j];
      if (v < 0 || std::isnan(v) || std::isinf(v)) {
        VLOG(0) << "error-PullValue-cvm in Paddle:" << i << ":" << v;
        ret = -1;
        break;
      }
    }
    if (ret == 0 && (h_total_dims[i] & 0x01)) {
      for (int j = cvm_offset - 1; j < (int)hidden_size; ++j) {
        float & v = h_values[i * hidden_size + j];
        if (std::isnan(v) || std::isinf(v)) {
          VLOG(0) << "error-PullValue-w in Paddle:" << i << ":" << v;
          ret = -1;
          break;
        }
      }
    }
    if (ret == 0 && expand_embed_dim > 0 && (h_total_dims[i] & 0x02)) {
      for (int j = 0; j < expand_embed_dim; ++j) {
        float & v = h_expand_values[i * expand_embed_dim + j];
        if (std::isnan(v) || std::isinf(v)) {
          VLOG(0) << "error-PullValue-expand in Paddle:" << i << ":" << v;
          ret = -1;
          break;
        }
      }
    }
    if (ret != 0) {
      break;
    }
  }
  if (ret != 0) {
    auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm* ptm = localtime(&now_time);
    char date[100] = {0};
    snprintf(date, 100, "%d%02d%02d%02d%02d%02d",
            (int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
            (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
    std::stringstream name_ss;
    name_ss << "paddle-pull-val.dev-" << dev_id << "-" << date << ".dump";
    std::ofstream ofs;
    ofs.open(name_ss.str(), std::ios::app);

    ofs << "slot-length: ";
    for (uint32_t i = 0; i < slot_lengths.size(); ++i) {
      ofs << i << ":" << slot_lengths[i] << ",";
    }
    ofs << "\n";

    for (int i = 0; i < val_len; ++i) {
      ofs << "slot:" << val_2_slot[i] << "," << i<< "\t";
      ofs << hidden_size << ":";
      if (h_total_dims[i] & 0x01) {
        for (int k = 0; k < (int)hidden_size; ++k) {
          ofs << h_values[i * hidden_size + k] << ",";
        }
      }
      ofs << "\t";
      ofs << expand_embed_dim << ":";
      if (expand_embed_dim > 0 && (h_total_dims[i] & 0x02)) {
        for (int k = 0; k < expand_embed_dim; ++k) {
          ofs << h_expand_values[i * expand_embed_dim + k] << ",";
        }
      }
      ofs << "\n";
    }

    ofs.close();
  }

  PADDLE_ENFORCE_EQ(ret, 0,
      platform::errors::PreconditionNotMet("CheckPullValue detect error value."));
}

bool check_continuous_memory_pull(int dev_id,
                                  const std::vector<float*>& values,
                                  const std::vector<int64_t>& slot_lengths,
                                  uint32_t hidden_size,
                                  int expand_embed_dim,
                                  int total_length) {
  int slot_num = slot_lengths.size();

  bool ret = true;
  int error_idx = -1;
  float* head_ptr = nullptr;
  for (int i = 0; i < slot_num; i++) {
    if (values[i] != nullptr && slot_lengths[i]) {
      head_ptr = values[i];
      break;
    }
  }
  float* next_ptr = head_ptr;
  for (int i = 0; i < slot_num; i++) {
    if (values[i] && slot_lengths[i]) {
      if (next_ptr != values[i]) {
        ret = false;
        error_idx = i;
        break;
      }
      next_ptr = values[i] + slot_lengths[i] * hidden_size;
    }
  }

  if (expand_embed_dim > 0 && ret != false) {
    float* virtual_expand = head_ptr + hidden_size * total_length;
    for (int i = 0; i < slot_num; i++) {
      if (values[slot_num + i] && slot_lengths[i]) {
        if (virtual_expand != values[slot_num + i]) {
          ret = false;
          error_idx = i;
          break;
        }
      }
      virtual_expand += slot_lengths[i] * expand_embed_dim;
    }
  }
  if (ret == false) {
    float* virtual_expand = head_ptr + hidden_size * total_length;
    VLOG(0) << "dev: " << dev_id << ", error_idx: " << error_idx;
    for (int i = 0; i < slot_num; i++) {
      VLOG(0) << "dev: "
              << dev_id
              << ", pull_copy values["
              << i
              << "]: "
              << values[i]
              << ", slot_lengths["
              << i
              << "]: "
              << (int)slot_lengths[i]
              << ", next_prt: "
              << values[i] + slot_lengths[i] * hidden_size
              << ", expand_values["
              << slot_num + i
              << "]: "
              << values[i + slot_num]
              << ", virtual_expand_values["
              << slot_num + i
              << "]: "
              << virtual_expand
              << ", next_prt: "
              << virtual_expand + slot_lengths[i] * expand_embed_dim;
      virtual_expand += slot_lengths[i] * expand_embed_dim;
    }
  }
  PADDLE_ENFORCE_EQ(
      ret,
      true,
      platform::errors::PreconditionNotMet(
          "Check Memory Continuous failed before CopyForPull, make sure no "
          "layer between pull and fused_seqpool_cvm"));
  return ret;
}

bool check_continuous_memory_push(int dev_id,
                                  const std::vector<const float*>& grad_values,
                                  const std::vector<int64_t>& slot_lengths,
                                  uint32_t hidden_size,
                                  int expand_embed_dim) {
  int slot_num = slot_lengths.size();

  bool ret = true;
  int error_idx = -1;
  float* head_ptr = nullptr;
  for (int i = 0; i < slot_num; i++) {
    if (grad_values[i] != nullptr && slot_lengths[i]) {
      head_ptr = (float*)grad_values[i];
      break;
    }
  }
  float* next_ptr = head_ptr;
  for (int i = 0; i < slot_num; i++) {
    if (grad_values[i] && slot_lengths[i]) {
      if (next_ptr != grad_values[i]) {
        ret = false;
        error_idx = i;
        break;
      }
      next_ptr = (float*)grad_values[i] + slot_lengths[i] * hidden_size;
    }
  }

  if (expand_embed_dim > 0 && ret != false) {
    float* expand_head_ptr = nullptr;
    for (int i = 0; i < slot_num; i++) {
      if (grad_values[slot_num + i] != nullptr && slot_lengths[i]) {
        expand_head_ptr = (float*)grad_values[slot_num + i];
        break;
      }
    }
    float* expand_next_ptr = expand_head_ptr;
    for (int i = 0; i < slot_num; i++) {
      if (grad_values[slot_num + i] && slot_lengths[i]) {
        if (expand_next_ptr != grad_values[slot_num + i]) {
          ret = false;
          error_idx = i;
          break;
        }
        expand_next_ptr = (float*)grad_values[slot_num + i] +
                          slot_lengths[i] * expand_embed_dim;
      }
    }
  }
  if (ret == false) {
    VLOG(0) << "dev: " << dev_id << ", error_idx: " << error_idx;
    for (int i = 0; i < slot_num; i++) {
      VLOG(0) << "dev: "
              << dev_id
              << ", push_copy grad_values["
              << i
              << "]: "
              << grad_values[i]
              << ", slot_lengths["
              << i
              << "]: "
              << (int)slot_lengths[i]
              << ", next_prt: "
              << grad_values[i] + slot_lengths[i] * hidden_size
              << ", expand_grad_values["
              << slot_num + i
              << "]: "
              << grad_values[i + slot_num]
              << ", next_prt: "
              << grad_values[i + slot_num] + slot_lengths[i] * expand_embed_dim;
    }
  }

  // PADDLE_ENFORCE_EQ(
  //     ret,
  //     true,
  //     platform::errors::PreconditionNotMet(
  //         "Check Memory Continuous failed before CopyForPush, make sure no "
  //         "layer between pull and fused_seqpool_cvm"));
  return ret;
}
#endif

void BoxWrapper::PullSparseCaseXPU(const paddle::platform::Place& place,
                                   const std::vector<const uint64_t*>& keys,
                                   const std::vector<float*>& values,
                                   const std::vector<int64_t>& slot_lengths,
                                   const int hidden_size,
                                   const int expand_embed_dim,
                                   const int skip_offset,
                                   bool expand_only) {
#ifdef PADDLE_WITH_XPU_KP
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto ctx_xpu = static_cast<platform::XPUDeviceContext*>(dev_ctx)->x_context();
  static bool use_l3_tensor = std::getenv("XPU_PADDLE_L3_TENSOR") != NULL ?
                    (std::strcmp(std::getenv("XPU_PADDLE_L3_TENSOR"), "1") == 0 ? true:false) :
                    false;
  phi::Place l3_place =
    static_cast<platform::XPUDeviceContext*>(dev_ctx)->GetL3Place();
  int device_id = place.GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];

  platform::Timer all_timer;
  platform::Timer pull_boxps_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  int slot_num = static_cast<int>(slot_lengths.size());
  dev.total_key_length = total_length;

  VLOG(3) << "Begine BoxPs PullSparse";
  xpu::ctx_guard RAII_GUARD(ctx_xpu);

#ifdef TRACE_PROFILE
  TRACE_SCOPE_START("copy keys", xpu_wait(ctx_xpu->xpu_stream));
#endif
  VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
  // LoDTensor& total_keys_tensor = dev.keys_tensor;
  uint64_t* total_keys;
  int* key2slot = nullptr;
  if (FLAGS_enable_pullpush_dedup_keys && use_xpu_sparse_map_) {
    total_keys = dev.keys_tensor.mutable_data<uint64_t>(total_length * 3 * sizeof(int64_t), place);
  } else {
    if(use_l3_tensor) {
      total_keys = dev.keys_tensor.mutable_data<uint64_t>(total_length * sizeof(int64_t), l3_place);
    } else {
      total_keys = dev.keys_tensor.mutable_data<uint64_t>(total_length * sizeof(int64_t), place);
    }
  }
  key2slot = dev.keys2slot.mutable_data<int>(total_length * sizeof(int), place);
  // construct slot_level lod info
  std::vector<int64_t> slot_lengths_lod(slot_num + 1, 0);
  for (int i = 1; i <= slot_num ; i++) {
    slot_lengths_lod[i] = slot_lengths_lod[i - 1] + slot_lengths[i - 1];
  }

  int* total_dims = dev.dims_tensor.mutable_data<int>(total_length * sizeof(int), place);

  uint64_t** xpu_keys = dev.keys_ptr_tensor.mutable_data<uint64_t*>(
      static_cast<int>(keys.size() * sizeof(uint64_t*)), place);
  int64_t* slot_lens = dev.slot_lens.mutable_data<int64_t>((slot_num + 1) * sizeof(int64_t), place);
  xpu_memcpy(xpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
                  XPU_HOST_TO_DEVICE);
  xpu_memcpy(slot_lens, slot_lengths_lod.data(),
                  slot_lengths_lod.size() * sizeof(int64_t),
                  XPU_HOST_TO_DEVICE);

#ifdef TRACE_PROFILE
  TRACE_SCOPE_START("CopyKeys", xpu_wait(ctx_xpu->xpu_stream));
#endif
  if (use_xpu_sparse_map_) {
    box_wrapper_kernel_->CopyKeys(place, xpu_keys, (unsigned long long *)total_keys, slot_lens,
                  static_cast<int>(slot_lengths.size()),
                  static_cast<int>(total_length), key2slot);
  } else {
    box_wrapper_kernel_->CopyKeys(place, xpu_keys, (uint32_t *)total_keys, slot_lens,
                  static_cast<int>(slot_lengths.size()),
                  static_cast<int>(total_length), key2slot);
  }
  uint64_t* d_pull_keys = total_keys;
  int pull_size = total_length;
  int* d_merged_idx = nullptr;
  int* d_merged_offsets = nullptr;
  int* d_res_idx = nullptr;
  std::thread thread_get_restore_idx;

  if (FLAGS_enable_pullpush_dedup_keys && use_xpu_sparse_map_) {
    uint64_t* d_merged_keys = reinterpret_cast<uint64_t*>(&total_keys[total_length]);
    pull_size = boxps_ptr_->DedupKeysAndFillIdxXPU(device_id, total_length, total_keys, 
                                                   d_merged_keys, d_merged_idx, d_merged_offsets);
    d_pull_keys = d_merged_keys;
    d_res_idx = reinterpret_cast<int*>(&total_keys[2 * total_length]);

    thread_get_restore_idx = std::thread([&] {
      xpu_set_device(device_id);
      std::vector<int> h_idx(total_length);
      std::vector<int> h_offset(pull_size + 1);
      xpu_memcpy(h_idx.data(),
                 d_merged_idx,
                 h_idx.size() * sizeof(int),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      xpu_memcpy(h_offset.data(),
                 d_merged_offsets,
                 pull_size * sizeof(int),
                 XPUMemcpyKind::XPU_DEVICE_TO_HOST);
      h_offset[pull_size] = total_length - 1;
      std::vector<int> tmp1(total_length);

      for (size_t i = 0; i < (size_t)pull_size; i++) {
        if (i != 0) {
          tmp1[h_offset[i]] = tmp1[h_offset[i] - 1];
        }
        else {
          tmp1[0] = 0;
        }
        for (int j = h_offset[i] + 1; j < h_offset[i + 1]; j++) {
          tmp1[j] = tmp1[j - 1] + 1;
        }
      }
      if (h_offset[pull_size - 1] != h_offset[pull_size]) {
        tmp1[h_offset[pull_size]] = tmp1[h_offset[pull_size] - 1] + 1;
      } else {
        tmp1[h_offset[pull_size]] = tmp1[h_offset[pull_size] - 1];
      }
      std::vector<int> h_res_idx(total_length);
      for (size_t i = 0; i < (size_t)total_length; i++) {
        h_res_idx[h_idx[i]] = i - tmp1[i];
      }

      xpu_memcpy(d_res_idx,
                 h_res_idx.data(),
                 total_length * sizeof(int),
                 XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    });
  }

  void* total_values_xpu = dev.pull_push_tensor.mutable_data<void>(pull_size * feature_pull_size_, place);

  VLOG(3) << "Begin call PullSparseXPU in BoxPS, dev: " << device_id
            << " len: " << total_length;
#ifdef TRACE_PROFILE
  TRACE_SCOPE_END("CopyKeys", xpu_wait(ctx_xpu->xpu_stream));
  TRACE_SCOPE_END("copy keys", xpu_wait(ctx_xpu->xpu_stream));
  TRACE_SCOPE_START("PullSparseXPU", xpu_wait(ctx_xpu->xpu_stream));
#endif
  pull_boxps_timer.Start();

  boxps_ptr_->PullSparseXPU(d_pull_keys, total_values_xpu, pull_size, device_id);

  pull_boxps_timer.Pause();
#ifdef TRACE_PROFILE
  TRACE_SCOPE_END("PullSparseXPU", xpu_wait(ctx_xpu->xpu_stream));
#endif
  VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
          << "]";

#ifdef TRACE_PROFILE
  TRACE_SCOPE_START("pull copy", xpu_wait(ctx_xpu->xpu_stream));
#endif
  boxps::FeaturePullOffset* pull_offset = nullptr;
  if (dev.pull_offset.memory_size() == 0) {
    pull_offset = dev.pull_offset.mutable_data<boxps::FeaturePullOffset>(
        sizeof(boxps::FeaturePullOffset), place);
    xpu_memcpy(pull_offset, &pull_info_, sizeof(boxps::FeaturePullOffset),
                    XPU_HOST_TO_DEVICE);
  } else {
    pull_offset = dev.pull_offset.data<boxps::FeaturePullOffset>();
  }

  float** xpu_values = dev.values_ptr_tensor.mutable_data<float*>(
        static_cast<int>(values.size() * sizeof(float*)), place);
  xpu_memcpy(xpu_values, values.data(), values.size() * sizeof(float*),
                  XPU_HOST_TO_DEVICE);

#ifdef TRACE_PROFILE
  TRACE_SCOPE_START("CopyForPull", xpu_wait(ctx_xpu->xpu_stream));
#endif
  if (FLAGS_enable_pullpush_dedup_keys && use_xpu_sparse_map_) {
    thread_get_restore_idx.join();
  }

  if (is_xpu_continuous_memory_pull_ == -1) {
    int check_expand_dim = expand_only ? expand_embed_dim : expand_embed_dim + hidden_size;
    if (pull_info_.expand_size < 0) check_expand_dim = -1;
    is_xpu_continuous_memory_pull_ = check_continuous_memory_pull(device_id,
                                 values,
                                 slot_lengths,
                                 hidden_size,
                                 check_expand_dim,
                                 total_length);
  }
  box_wrapper_kernel_->CopyForPull(place, xpu_keys, (float**)values.data(), total_values_xpu,
                                   pull_offset, slot_lengths_lod.data(), slot_num, key2slot, d_res_idx, hidden_size,
                                   expand_embed_dim, total_length, total_dims, skip_offset,
                                   expand_only, d_merged_idx, d_merged_offsets, pull_size);

  if (check_xpu_nan_) {
    xpu_wait(ctx_xpu->xpu_stream);
    CheckPullValue(
        device_id, values, slot_lengths, &pull_info_,
        hidden_size, pull_info_.embedx_size - pull_info_.show, expand_embed_dim, total_dims);
  }
#ifdef TRACE_PROFILE
  TRACE_SCOPE_END("CopyForPull", xpu_wait(ctx_xpu->xpu_stream));
  TRACE_SCOPE_END("pull copy", xpu_wait(ctx_xpu->xpu_stream));
#endif
  all_timer.Pause();
#endif
}

void BoxWrapper::PullSparseCase(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<float*>& values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size,
                                const int expand_embed_dim,
                                const int skip_offset,
                                bool expand_only) {
  if (platform::is_cpu_place(place)) {
    PullSparseCaseCPU(place,
                      keys,
                      values,
                      slot_lengths,
                      hidden_size,
                      expand_embed_dim,
                      skip_offset,
                      expand_only);
  } else if (platform::is_gpu_place(place)) {
    PullSparseCaseGPU(place,
                      keys,
                      values,
                      slot_lengths,
                      hidden_size,
                      expand_embed_dim,
                      skip_offset,
                      expand_only);
  } else {
    PullSparseCaseXPU(place,
                      keys,
                      values,
                      slot_lengths,
                      hidden_size,
                      expand_embed_dim,
                      skip_offset,
                      expand_only);
  }
}

void BoxWrapper::PushSparseGradCaseGPU(
    const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths,
    const int hidden_size,
    const int expand_embed_dim,
    const int batch_size,
    const int skip_offset,
    bool expand_only) {
#if defined(PADDLE_WITH_CUDA)
  int device_id = place.GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_push_timer;
  platform::Timer& push_boxps_timer = dev.boxps_push_timer;
  platform::Timer& copy_push_timer = dev.copy_push_timer;

  auto stream = dynamic_cast<phi::GPUContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  cudaStreamSynchronize(stream);

  all_timer.Resume();

  uint64_t* total_keys =
      reinterpret_cast<uint64_t*>(dev.keys_tensor.data<int64_t>());
  int* total_dims = reinterpret_cast<int*>(dev.dims_tensor.data<int>());
  int slot_num = static_cast<int>(slot_lengths.size());
  if (dev.d_slot_vector.memory_size() == 0) {
    int* buf_slot_vector =
        dev.d_slot_vector.mutable_data<int>(slot_num * sizeof(int), place);
    cudaMemcpyAsync(buf_slot_vector,
                    slot_vector_.data(),
                    slot_num * sizeof(int),
                    cudaMemcpyHostToDevice,
                    stream);
  }

  boxps::FeaturePushOffset* push_offset = nullptr;
  if (dev.push_offset.memory_size() == 0) {
    push_offset = dev.push_offset.mutable_data<boxps::FeaturePushOffset>(
        sizeof(boxps::FeaturePushOffset), place);
    cudaMemcpyAsync(push_offset,
                    &push_info_,
                    sizeof(boxps::FeaturePushOffset),
                    cudaMemcpyHostToDevice,
                    stream);
  } else {
    push_offset = dev.push_offset.data<boxps::FeaturePushOffset>();
  }

  const int64_t* slot_lens =
      reinterpret_cast<int64_t*>(dev.slot_lens.data<int64_t>());
  const int* d_slot_vector = dev.d_slot_vector.data<int>();
  const int* key2slot = reinterpret_cast<int*>(dev.keys2slot.data<int>());
  float** gpu_values = dev.values_ptr_tensor.data<float*>();
  cudaMemcpyAsync(gpu_values,
                  grad_values.data(),
                  grad_values.size() * sizeof(float*),
                  cudaMemcpyHostToDevice,
                  stream);

  int64_t total_length = dev.total_key_length;
  // dedup keys pull
  if (FLAGS_enable_pullpush_dedup_keys) {
    const uint32_t* d_restore_idx =
        reinterpret_cast<const uint32_t*>(&key2slot[total_length]);
    const uint32_t* d_sorted_idx =
        reinterpret_cast<const uint32_t*>(&key2slot[total_length * 2]);
    const uint32_t* d_offset =
        reinterpret_cast<const uint32_t*>(&d_sorted_idx[total_length]);
    const uint32_t* d_merged_cnts =
        reinterpret_cast<const uint32_t*>(&d_offset[total_length]);
    uint64_t* d_merged_keys = &total_keys[total_length];

    int64_t dedup_size = dev.dedup_key_length;
    int64_t total_bytes = dedup_size * feature_push_size_;
    void* total_grad_values_gpu =
        dev.pull_push_tensor.mutable_data<void>(total_bytes, place);

    copy_push_timer.Resume();
    this->CopyForPush(place,
                      gpu_values,
                      total_grad_values_gpu,
                      push_offset,
                      total_length,
                      dedup_size,
                      d_slot_vector,
                      slot_lens,
                      slot_num,
                      hidden_size,
                      expand_embed_dim,
                      batch_size,
                      total_dims,
                      key2slot,
                      skip_offset,
                      expand_only,
                      d_sorted_idx,
                      d_offset,
                      d_merged_cnts,
                      d_restore_idx);
    copy_push_timer.Pause();
    push_boxps_timer.Resume();
    int ret = boxps_ptr_->PushSparseGPU(
        d_merged_keys,
        reinterpret_cast<void*>(total_grad_values_gpu),
        static_cast<int>(dedup_size),
        device_id);
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        platform::errors::PreconditionNotMet("PushSparseGPU failed in BoxPS."));
    push_boxps_timer.Pause();
  } else {
    int64_t total_bytes = total_length * feature_push_size_;
    void* total_grad_values_gpu =
        dev.pull_push_tensor.mutable_data<void>(total_bytes, place);
    copy_push_timer.Resume();
    this->CopyForPush(place,
                      gpu_values,
                      total_grad_values_gpu,
                      push_offset,
                      total_length,
                      0,
                      d_slot_vector,
                      slot_lens,
                      slot_num,
                      hidden_size,
                      expand_embed_dim,
                      batch_size,
                      total_dims,
                      key2slot,
                      skip_offset,
                      expand_only);
    copy_push_timer.Pause();
    push_boxps_timer.Resume();
    int ret = boxps_ptr_->PushSparseGPU(
        total_keys,
        reinterpret_cast<void*>(total_grad_values_gpu),
        static_cast<int>(total_length),
        device_id);
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        platform::errors::PreconditionNotMet("PushSparseGPU failed in BoxPS."));
    push_boxps_timer.Pause();
  }
  all_timer.Pause();
#endif
}

#ifdef PADDLE_WITH_XPU_KP
void CheckPushValue(
    int dev_id,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths,
    boxps::FeaturePushOffset * push_info,
    uint32_t hidden_size,
    int cvm_offset,
    int expand_embed_dim,
    int * total_dims) {
  int val_len = 0;
  for (uint32_t i = 0; i < slot_lengths.size(); ++i) {
    val_len += slot_lengths[i];
  }
  int fixed_float_len = val_len * hidden_size;
  std::vector<float> h_values(fixed_float_len);
  int expand_float_len = val_len * expand_embed_dim;
  std::vector<float> h_expand_values(expand_float_len);
  if (expand_float_len > 0) {
    memset(&(h_expand_values[0]), 0, sizeof(float) * expand_float_len);
  }
  std::vector<int> h_total_dims(val_len);
  int copy_dim_ret = xpu_memcpy(&(h_total_dims[0]), total_dims, sizeof(int) * val_len, XPU_DEVICE_TO_HOST);
  PADDLE_ENFORCE_EQ(
       copy_dim_ret, 0,
       platform::errors::PreconditionNotMet("CheckPushValue copy total_dim error."));

  std::vector<int> val_2_slot(val_len);

  int offset = 0;
  if (expand_embed_dim > 0) {
    PADDLE_ENFORCE_EQ(
       slot_lengths.size() * 2, grad_values.size(),
       platform::errors::PreconditionNotMet("CheckPushValue slot_length vs grad_values  error."));
  } else {
    PADDLE_ENFORCE_EQ(
       slot_lengths.size(), grad_values.size(),
       platform::errors::PreconditionNotMet("CheckPushValue slot_length vs grad_values  error."));
  }
  int slot_num = slot_lengths.size();
  for (int i = 0; i < slot_num; ++i) {
    if (grad_values[i] == nullptr) {
      if (slot_lengths[i] != 0) {
        VLOG(0) << "CheckPushValue found slot[" << i << "] dval_ptr is null, while slot_length != 0:" << slot_lengths[i];
      }
      PADDLE_ENFORCE_EQ(slot_lengths[i], 0,
          platform::errors::PreconditionNotMet("CheckPushValue slot_length error."));
      continue;
    }
    if (slot_lengths[i] == 0) {
      continue;
    }
    for (int j = 0; j < slot_lengths[i]; ++j) {
        val_2_slot[offset + j] = i;
    }
    // copy show/clk/embed/embedx
    xpu_memcpy(&(h_values[offset * hidden_size]), grad_values[i],
        sizeof(float) * hidden_size * slot_lengths[i], XPU_DEVICE_TO_HOST);
    // copy expand emb
    if (expand_embed_dim > 0 && grad_values[i + slot_num] != nullptr) {
        xpu_memcpy(&(h_expand_values[offset * expand_embed_dim]), grad_values[i + slot_num],
                    sizeof(float) * expand_embed_dim * slot_lengths[i], XPU_DEVICE_TO_HOST);
    }
    offset += slot_lengths[i];
  }
  int ret = 0;
  for (int i = 0; i < val_len; ++i) {
    for (int j = 0; j < cvm_offset - 1; ++j) {
      float & v = h_values[i * hidden_size + j];
      if (v < 0 || std::isnan(v) || std::isinf(v)) {
        VLOG(0) << "error-PushValue-cvm in Paddle:" << i << ":" << v;
        ret = -1;
        break;
      }
    }
    if (ret == 0 && (h_total_dims[i] & 0x01)) {
      for (int j = cvm_offset - 1; j < (int)hidden_size; ++j) {
        float & v = h_values[i * hidden_size + j];
        if (std::isnan(v) || std::isinf(v)) {
          VLOG(0) << "error-PushValue-w in Paddle:" << i << ":" << v;
          ret = -1;
          break;
        }
      }
    }
    if (ret == 0 && expand_embed_dim > 0 && (h_total_dims[i] & 0x02)) {
      for (int j = 0; j < expand_embed_dim; ++j) {
        float & v = h_expand_values[i * expand_embed_dim + j];
        if (std::isnan(v) || std::isinf(v)) {
          VLOG(0) << "error-PushValue-expand in Paddle:" << i << ":" << v;
          ret = -1;
          break;
        }
      }
    }
    if (ret != 0) {
      break;
    }
  }

  if (ret != 0) {
    auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm* ptm = localtime(&now_time);
    char date[100] = {0};
    snprintf(date, 100, "%d%02d%02d%02d%02d%02d",
            (int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
            (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
    std::stringstream name_ss;
    name_ss << "paddle-push-val.dev-" << dev_id << "-" << date << ".dump";
    std::ofstream ofs;
    ofs.open(name_ss.str(), std::ios::app);

    ofs << "slot-length: ";
    for (uint32_t i = 0; i < slot_lengths.size(); ++i) {
      ofs << i << ":" << slot_lengths[i] << ",";
    }
    ofs << "\n";

    for (int i = 0; i < val_len; ++i) {
      ofs << "slot:" << val_2_slot[i] << "," << i<< "\t";
      ofs << hidden_size << ":";
      if (h_total_dims[i] & 0x01) {
        for (int k = 0; k < (int)hidden_size; ++k) {
          ofs << h_values[i * hidden_size + k] << ",";
        }
      }
      ofs << "\t";
      ofs << expand_embed_dim << ":";
      if (expand_embed_dim > 0 && (h_total_dims[i] & 0x02)) {
        for (int k = 0; k < expand_embed_dim; ++k) {
          ofs << h_expand_values[i * expand_embed_dim + k] << ",";
        }
      }
      ofs << "\n";
    }
    ofs.close();
  }

  PADDLE_ENFORCE_EQ(ret, 0,
      platform::errors::PreconditionNotMet("CheckPushValue detect error value."));
}

void CheckPushValue(
    int dev_id,
    float * grad_values,
    int val_len,
    boxps::FeaturePushOffset * push_info,
    int cvm_offset,
    int push_float_num) {
  int float_len = val_len * push_float_num;
  std::vector<float> h_values(float_len);
  xpu_memcpy(&(h_values[0]), grad_values, sizeof(float) * float_len, XPU_DEVICE_TO_HOST);
  int ret = 0;
  for (int i = 0; i < val_len; ++i) {
    for (int j = push_info->show; j < push_info->embed_g; ++j) {
      float & v = h_values[i * push_float_num + j];
      if (v < 0 || std::isnan(v) || std::isinf(v)) {
        VLOG(0) << "error-PushValue-cvm in Paddle-preboxps:" << i << ":" << v;
        ret = -1;
        break;
      }
    }
    if (ret == 0) {
      for (int j = push_info->embed_g; j < push_float_num; ++j) {
        float & v = h_values[i * push_float_num + j];
        if (std::isnan(v) || std::isinf(v)) {
          VLOG(0) << "error-PushValue-w in Paddle-preboxps:" << i << ":" << v;
          ret = -1;
          break;
        }
      }
    }
    if (ret != 0) {
      break;
    }
  }

  if (ret != 0) {
    auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm* ptm = localtime(&now_time);
    char date[100] = {0};
    snprintf(date, 100, "%d%02d%02d%02d%02d%02d",
            (int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
            (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
    std::stringstream name_ss;
    name_ss << "paddlepreboxps-push-val.dev-" << dev_id << "-" << date << ".dump";
    std::ofstream ofs;
    ofs.open(name_ss.str(), std::ios::app);

    for (int i = 0; i < val_len; ++i) {
      ofs << i << "\t" << *(int *)&(h_values[push_info->slot]) << "\t";
      for (int j = push_info->show; j < push_info->embed_g; ++j) {
        float & v = h_values[i * push_float_num + j];
        ofs << v << "\t";
      }
      for (int j = push_info->embed_g; j < push_float_num; ++j) {
        float & v = h_values[i * push_float_num + j];
        ofs << v << ",";
      }
      ofs << "\n";
    }
    ofs.close();
  }

  PADDLE_ENFORCE_EQ(ret, 0,
      platform::errors::PreconditionNotMet("CheckPushValue-preboxps detect error value."));
}
#endif

void BoxWrapper::PushSparseGradCaseXPU(const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths,
    const int hidden_size,
    const int expand_embed_dim,
    const int batch_size,
    const int skip_offset,
    bool expand_only) {
#ifdef PADDLE_WITH_XPU_KP
  int device_id = place.GetDeviceId();
  DeviceBoxData& dev = device_caches_[device_id];
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto ctx_xpu = static_cast<platform::XPUDeviceContext*>(dev_ctx)->x_context();
  ctx_xpu = ctx_xpu;//avoid unused error without TRACE_PROFILE

  platform::Timer& all_timer = dev.all_push_timer;
  platform::Timer& push_boxps_timer = dev.boxps_push_timer;

  all_timer.Resume();

  if (check_xpu_nan_) {
    xpu_wait(ctx_xpu->xpu_stream);
    int * tmp_dims = reinterpret_cast<int*>(dev.dims_tensor.data<int>());
    CheckPushValue(
      device_id, grad_values, slot_lengths, &push_info_, hidden_size,
      pull_info_.embedx_size - pull_info_.show, expand_embed_dim, tmp_dims);
  }

#ifdef TRACE_PROFILE
  TRACE_SCOPE_START("push copy", xpu_wait(ctx_xpu->xpu_stream));
#endif
  int64_t total_length = dev.total_key_length;
  int64_t total_bytes = total_length * feature_push_size_;
  void* total_grad_values_xpu =
      dev.pull_push_tensor.mutable_data<void>(total_bytes, place);
  uint64_t* total_keys =
      reinterpret_cast<uint64_t*>(dev.keys_tensor.data<int64_t>());
  int* total_dims = reinterpret_cast<int*>(dev.dims_tensor.data<int>());
  int slot_num = static_cast<int>(slot_lengths.size());

  if (!dev.d_slot_vector.IsInitialized()) {
    int* buf_slot_vector = dev.d_slot_vector.mutable_data<int>(slot_num * sizeof(int), place);
    xpu_memcpy(buf_slot_vector, slot_vector_.data(),
                    slot_num * sizeof(int), XPU_HOST_TO_DEVICE);
  }

  boxps::FeaturePushOffset* push_offset = nullptr;
  if (dev.push_offset.memory_size() == 0) {
    push_offset = dev.push_offset.mutable_data<boxps::FeaturePushOffset>(
        sizeof(boxps::FeaturePushOffset), place);
    xpu_memcpy(push_offset, &push_info_, sizeof(boxps::FeaturePushOffset),
        XPU_HOST_TO_DEVICE);
  } else {
    push_offset = dev.push_offset.data<boxps::FeaturePushOffset>();
  }

  auto slot_lengths_lod = slot_lengths;
  for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
    slot_lengths_lod[i] += slot_lengths_lod[i - 1];
  }
  const int64_t* slot_len_lod =
      reinterpret_cast<int64_t*>(dev.slot_lens.data<int64_t>());
  const int* slot_vector = dev.d_slot_vector.data<int>();
  const int* key2slot = reinterpret_cast<int*>(dev.keys2slot.data<int>());
  float** xpu_values = dev.values_ptr_tensor.data<float*>();
  xpu_memcpy(xpu_values, grad_values.data(),
                  grad_values.size() * sizeof(float*), XPU_HOST_TO_DEVICE);
#ifdef TRACE_PROFILE
  TRACE_SCOPE_START("CopyForPush's xpu::copy", xpu_wait(ctx_xpu->xpu_stream));
  TRACE_SCOPE_END("CopyForPush's xpu::copy", xpu_wait(ctx_xpu->xpu_stream));

  TRACE_SCOPE_START("CopyForPush", xpu_wait(ctx_xpu->xpu_stream));
#endif
//   float* real_grad_values;
//   for (int i = 0; i < slot_num; i++) {
//     if(grad_values[i] != nullptr) {
//       real_grad_values = const_cast<float*>(grad_values[i]);
//       break;
//     }
//   }
  std::vector<int> slot_inner_offset(total_length);
  int out_count = 0;
  for (int i = 0; i < slot_num; i++) {
    for (int64_t j = 0; j < slot_lengths[i]; j++) {
      slot_inner_offset[out_count++] = j;
    }
  }
  auto d_slot_inner_offset_tmp = memory::Alloc(place, total_length * sizeof(int));
  int* d_slot_inner_offset = reinterpret_cast<int*>(d_slot_inner_offset_tmp->ptr());
  memory::Copy(place,
               d_slot_inner_offset,
               platform::CPUPlace(),
               slot_inner_offset.data(),
               total_length * sizeof(int));

  if (is_xpu_continuous_memory_push_ == -1) {
    int check_expand_dim = expand_only ? expand_embed_dim : expand_embed_dim + hidden_size;
    if (pull_info_.expand_size < 0) check_expand_dim = -1;
    is_xpu_continuous_memory_push_ = check_continuous_memory_push(device_id,
                                 grad_values,
                                 slot_lengths,
                                 hidden_size,
                                 check_expand_dim);
  }

  box_wrapper_kernel_->CopyForPush(place, xpu_values, total_grad_values_xpu,
      push_offset, total_length, slot_vector, (int*)d_slot_inner_offset, slot_len_lod, slot_num,
      hidden_size, batch_size, total_dims, skip_offset, key2slot,
      expand_embed_dim,
      push_float_num_,
      expand_only,
      (bool)is_xpu_continuous_memory_push_);

  push_boxps_timer.Resume();
#ifdef TRACE_PROFILE
  TRACE_SCOPE_END("CopyForPush", xpu_wait(ctx_xpu->xpu_stream));
  TRACE_SCOPE_END("push copy", xpu_wait(ctx_xpu->xpu_stream));

  TRACE_SCOPE_START("PushSparseXPU", xpu_wait(ctx_xpu->xpu_stream));
#endif

  if (check_xpu_nan_) {
    xpu_wait(ctx_xpu->xpu_stream);
    CheckPushValue(
        device_id, (float *)total_grad_values_xpu, static_cast<int>(total_length),
        &push_info_, pull_info_.embedx_size - pull_info_.show, push_float_num_); 
  }

  int ret = boxps_ptr_->PushSparseXPU(total_keys,
      reinterpret_cast<void*>(total_grad_values_xpu),
      static_cast<int>(total_length), device_id);

  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                              "PushSparseXPU failed in BoxPS."));
  push_boxps_timer.Pause();
#ifdef TRACE_PROFILE
  TRACE_SCOPE_END("PushSparseXPU", xpu_wait(ctx_xpu->xpu_stream));
#endif
  all_timer.Pause();

#endif
}

void BoxWrapper::PushSparseGradCaseCPU(
    const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths,
    const int hidden_size,
    const int expand_embed_dim,
    const int batch_size,
    const int skip_offset,
    bool expand_only) {
  int device_id = GetPlaceDeviceId(place);
  DeviceBoxData& dev = device_caches_[device_id];
  platform::Timer& all_timer = dev.all_push_timer;
  platform::Timer& push_boxps_timer = dev.boxps_push_timer;

  all_timer.Resume();

  uint64_t* total_keys =
      reinterpret_cast<uint64_t*>(dev.keys_tensor.data<int64_t>());
  int* total_dims = reinterpret_cast<int*>(dev.dims_tensor.data<int>());
  int slot_num = static_cast<int>(slot_lengths.size());

  const int64_t* slot_lens =
      reinterpret_cast<int64_t*>(dev.slot_lens.data<int64_t>());
  const int* key2slot = reinterpret_cast<int*>(dev.keys2slot.data<int>());

  int64_t total_length = dev.total_key_length;
  // dedup keys pull
  const uint32_t* d_sorted_idx =
      reinterpret_cast<const uint32_t*>(&key2slot[total_length * 2]);
  const uint32_t* d_offset =
      reinterpret_cast<const uint32_t*>(&d_sorted_idx[total_length]);
  const uint32_t* d_merged_cnts =
      reinterpret_cast<const uint32_t*>(&d_offset[total_length]);
  uint64_t* d_merged_keys = &total_keys[total_length];

  int64_t dedup_size = dev.dedup_key_length;
  int64_t total_bytes = dedup_size * feature_push_size_;
  void* total_grad_values_gpu =
      dev.pull_push_tensor.mutable_data<void>(total_bytes, place);
  this->CopyForPushCPU(place,
                       grad_values,
                       total_grad_values_gpu,
                       slot_vector_.data(),
                       slot_lens,
                       slot_num,
                       hidden_size,
                       expand_embed_dim,
                       dedup_size,
                       batch_size,
                       total_dims,
                       key2slot,
                       skip_offset,
                       expand_only,
                       d_sorted_idx,
                       d_offset,
                       d_merged_cnts);

  push_boxps_timer.Resume();
  int ret =
      boxps_ptr_->PushSparseGPU(d_merged_keys,
                                reinterpret_cast<void*>(total_grad_values_gpu),
                                static_cast<int>(dedup_size),
                                device_id);
  PADDLE_ENFORCE_EQ(
      ret,
      0,
      platform::errors::PreconditionNotMet("PushSparseGPU failed in BoxPS."));
  push_boxps_timer.Pause();

  all_timer.Pause();
}

void BoxWrapper::PushSparseGradCase(
    const paddle::platform::Place& place,
    const std::vector<const uint64_t*>& keys,
    const std::vector<const float*>& grad_values,
    const std::vector<int64_t>& slot_lengths,
    const int hidden_size,
    const int expand_embed_dim,
    const int batch_size,
    const int skip_offset,
    bool expand_only) {

  if (platform::is_cpu_place(place)) {
    PushSparseGradCaseCPU(place,
                          keys,
                          grad_values,
                          slot_lengths,
                          hidden_size,
                          expand_embed_dim,
                          batch_size,
                          skip_offset,
                          expand_only);
  } else if (platform::is_gpu_place(place)) {
    PushSparseGradCaseGPU(place,
                          keys,
                          grad_values,
                          slot_lengths,
                          hidden_size,
                          expand_embed_dim,
                          batch_size,
                          skip_offset,
                          expand_only);
  } else {
    PushSparseGradCaseXPU(place,
                          keys,
                          grad_values,
                          slot_lengths,
                          hidden_size,
                          expand_embed_dim,
                          batch_size,
                          skip_offset,
                          expand_only);
  }
}

}  // namespace framework
}  // namespace paddle
#endif
