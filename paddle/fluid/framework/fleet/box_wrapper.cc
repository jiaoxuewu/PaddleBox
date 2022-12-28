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
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/memory/allocation/allocator_facade.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/gpu_info.h"

DECLARE_bool(use_gpu_replica_cache);
DECLARE_int32(gpu_replica_cache_dim);
DECLARE_bool(enable_force_hbm_recyle);
DECLARE_bool(enable_force_mem_recyle);
DECLARE_bool(enbale_slotpool_auto_clear);
namespace paddle {
namespace framework {

std::shared_ptr<BoxWrapper> BoxWrapper::s_instance_ = nullptr;
std::shared_ptr<boxps::PaddleShuffler> BoxWrapper::data_shuffle_ = nullptr;
cudaStream_t BoxWrapper::stream_list_[MAX_GPU_NUM];

void BasicAucCalculator::add_unlock_data(double pred, int label) {
  PADDLE_ENFORCE_GE(pred, 0.0, platform::errors::PreconditionNotMet(
                                   "pred should be greater than 0"));
  PADDLE_ENFORCE_LE(pred, 1.0, platform::errors::PreconditionNotMet(
                                   "pred should be lower than 1"));
  PADDLE_ENFORCE_EQ(
      label * label, label,
      platform::errors::PreconditionNotMet(
          "label must be equal to 0 or 1, but its value is: %d", label));
  int pos = std::min(static_cast<int>(pred * _table_size), _table_size - 1);
  PADDLE_ENFORCE_GE(
      pos, 0,
      platform::errors::PreconditionNotMet(
          "pos must be equal or greater than 0, but its value is: %d", pos));
  PADDLE_ENFORCE_LT(
      pos, _table_size,
      platform::errors::PreconditionNotMet(
          "pos must be less than table_size, but its value is: %d", pos));
  _local_abserr += fabs(pred - label);
  _local_sqrerr += (pred - label) * (pred - label);
  _local_pred += pred;
  ++_table[label][pos];
}

void BasicAucCalculator::add_unlock_data(double pred, int label,
                                         float sample_scale) {
  PADDLE_ENFORCE_GE(pred, 0.0, platform::errors::PreconditionNotMet(
                                   "pred should be greater than 0"));
  PADDLE_ENFORCE_LE(pred, 1.0, platform::errors::PreconditionNotMet(
                                   "pred should be lower than 1"));
  PADDLE_ENFORCE_EQ(
      label * label, label,
      platform::errors::PreconditionNotMet(
          "label must be equal to 0 or 1, but its value is: %d", label));
  int pos = std::min(static_cast<int>(pred * _table_size), _table_size - 1);
  PADDLE_ENFORCE_GE(
      pos, 0,
      platform::errors::PreconditionNotMet(
          "pos must be equal or greater than 0, but its value is: %d", pos));
  PADDLE_ENFORCE_LT(
      pos, _table_size,
      platform::errors::PreconditionNotMet(
          "pos must be less than table_size, but its value is: %d", pos));
  _local_abserr += fabs(pred - label);
  _local_sqrerr += (pred - label) * (pred - label);

  _local_pred += pred * sample_scale;
  _table[label][pos] += sample_scale;
}

void BasicAucCalculator::add_unlock_data_with_float_label(double pred,
                                                          double label) {
  PADDLE_ENFORCE_GE(pred, 0.0, platform::errors::PreconditionNotMet(
                                   "pred should be greater than 0"));
  PADDLE_ENFORCE_LE(pred, 1.0, platform::errors::PreconditionNotMet(
                                   "pred should be lower than 1"));

  int pos = std::min(static_cast<int>(pred * _table_size), _table_size - 1);
  PADDLE_ENFORCE_GE(
      pos, 0,
      platform::errors::PreconditionNotMet(
          "pos must be equal or greater than 0, but its value is: %d", pos));
  PADDLE_ENFORCE_LT(
      pos, _table_size,
      platform::errors::PreconditionNotMet(
          "pos must be less than table_size, but its value is: %d", pos));
  _local_abserr += fabs(pred - label);
  _local_sqrerr += (pred - label) * (pred - label);
  _local_pred += pred;
  _table[0][pos] += 1 - label;
  _table[1][pos] += label;
}

void BasicAucCalculator::add_data(const float* d_pred, const int64_t* d_label,
                                  int batch_size,
                                  const paddle::platform::Place& place) {
  if (_mode_collect_in_gpu) {
    cuda_add_data(place, d_label, d_pred, batch_size);
  } else {
    thread_local std::vector<float> h_pred;
    thread_local std::vector<int64_t> h_label;
    h_pred.resize(batch_size);
    h_label.resize(batch_size);
    cudaMemcpy(h_pred.data(), d_pred, sizeof(float) * batch_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_label.data(), d_label, sizeof(int64_t) * batch_size,
               cudaMemcpyDeviceToHost);

    std::lock_guard<std::mutex> lock(_table_mutex);
    for (int i = 0; i < batch_size; ++i) {
      add_unlock_data(h_pred[i], h_label[i]);
    }
  }
}

void BasicAucCalculator::add_sample_data(
    const float* d_pred, const int64_t* d_label,
    const std::vector<float>& d_sample_scale, int batch_size,
    const paddle::platform::Place& place) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<int64_t> h_label;
  h_pred.resize(batch_size);
  h_label.resize(batch_size);
  cudaMemcpy(h_pred.data(), d_pred, sizeof(float) * batch_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_label.data(), d_label, sizeof(int64_t) * batch_size,
             cudaMemcpyDeviceToHost);

  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    add_unlock_data(h_pred[i], h_label[i], d_sample_scale[i]);
  }
}

// add mask data
void BasicAucCalculator::add_mask_data(const float* d_pred,
                                       const int64_t* d_label,
                                       const int64_t* d_mask, int batch_size,
                                       const paddle::platform::Place& place) {
  if (_mode_collect_in_gpu) {
    cuda_add_mask_data(place, d_label, d_pred, d_mask, batch_size);
  } else {
    thread_local std::vector<float> h_pred;
    thread_local std::vector<int64_t> h_label;
    thread_local std::vector<int64_t> h_mask;
    h_pred.resize(batch_size);
    h_label.resize(batch_size);
    h_mask.resize(batch_size);

    cudaMemcpy(h_pred.data(), d_pred, sizeof(float) * batch_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_label.data(), d_label, sizeof(int64_t) * batch_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask.data(), d_mask, sizeof(int64_t) * batch_size,
               cudaMemcpyDeviceToHost);

    std::lock_guard<std::mutex> lock(_table_mutex);
    for (int i = 0; i < batch_size; ++i) {
      if (h_mask[i]) {
        add_unlock_data(h_pred[i], h_label[i]);
      }
    }
  }
}

// add float mask data
void BasicAucCalculator::add_float_mask_data(
    const float* d_pred, const float* d_label, const int64_t* d_mask,
    int batch_size, const paddle::platform::Place& place) {
  if (_mode_collect_in_gpu) {
    cuda_add_float_mask_data(place, d_label, d_pred, d_mask, batch_size);
  } else {
    thread_local std::vector<float> h_pred;
    thread_local std::vector<float> h_label;
    thread_local std::vector<int64_t> h_mask;
    h_pred.resize(batch_size);
    h_label.resize(batch_size);
    h_mask.resize(batch_size);

    cudaMemcpy(h_pred.data(), d_pred, sizeof(float) * batch_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_label.data(), d_label, sizeof(float) * batch_size,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask.data(), d_mask, sizeof(int64_t) * batch_size,
               cudaMemcpyDeviceToHost);

    std::lock_guard<std::mutex> lock(_table_mutex);
    for (int i = 0; i < batch_size; ++i) {
      if (h_mask[i]) {
        add_unlock_data_with_float_label(h_pred[i], h_label[i]);
      }
    }
  }
}

void BasicAucCalculator::init(int table_size, int max_batch_size) {
  if (_mode_collect_in_gpu) {
    PADDLE_ENFORCE_GE(
        max_batch_size, 0,
        platform::errors::PreconditionNotMet(
            "max_batch_size should be greater than 0 in mode_collect_in_gpu"));
  }
  set_table_size(table_size);
  set_max_batch_size(max_batch_size);
  // init CPU memory
  for (int i = 0; i < 2; i++) {
    _table[i] = std::vector<double>();
  }
  // init GPU memory
  if (_mode_collect_in_gpu) {
    for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
      auto place = platform::CUDAPlace(i);
      _d_positive.emplace_back(
          memory::AllocShared(place, _table_size * sizeof(double)));
      _d_negative.emplace_back(
          memory::AllocShared(place, _table_size * sizeof(double)));
      _d_abserr.emplace_back(
          memory::AllocShared(place, _max_batch_size * sizeof(double)));
      _d_sqrerr.emplace_back(
          memory::AllocShared(place, _max_batch_size * sizeof(double)));
      _d_pred.emplace_back(
          memory::AllocShared(place, _max_batch_size * sizeof(double)));
    }
  }
  // reset
  reset();
}

void BasicAucCalculator::reset() {
  // reset CPU counter
  for (int i = 0; i < 2; i++) {
    _table[i].assign(_table_size, 0.0);
  }
  _local_abserr = 0;
  _local_sqrerr = 0;
  _local_pred = 0;
  // reset GPU counter
  if (_mode_collect_in_gpu) {
    // backup orginal device
    int ori_device;
    cudaGetDevice(&ori_device);
    for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
      cudaSetDevice(i);
      auto place = platform::CUDAPlace(i);
      platform::CUDADeviceContext* context =
          dynamic_cast<platform::CUDADeviceContext*>(
              platform::DeviceContextPool::Instance().Get(place));
      auto stream = context->stream();
      cudaMemsetAsync(_d_positive[i]->ptr(), 0, sizeof(double) * _table_size,
                      stream);
      cudaMemsetAsync(_d_negative[i]->ptr(), 0, sizeof(double) * _table_size,
                      stream);
      cudaMemsetAsync(_d_abserr[i]->ptr(), 0, sizeof(double) * _max_batch_size,
                      stream);
      cudaMemsetAsync(_d_sqrerr[i]->ptr(), 0, sizeof(double) * _max_batch_size,
                      stream);
      cudaMemsetAsync(_d_pred[i]->ptr(), 0, sizeof(double) * _max_batch_size,
                      stream);
    }
    // restore device
    cudaSetDevice(ori_device);
  }
}

void BasicAucCalculator::collect_data_nccl() {
  // backup orginal device
  int ori_device;
  cudaGetDevice(&ori_device);
  // transfer to CPU
  platform::dynload::ncclGroupStart();
  // nccl allreduce sum
  for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
    cudaSetDevice(i);
    auto place = platform::CUDAPlace(i);
    auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                      platform::DeviceContextPool::Instance().Get(place))
                      ->stream();
    auto comm = platform::NCCLCommContext::Instance().Get(0, place);
    platform::dynload::ncclAllReduce(
        _d_positive[i]->ptr(), _d_positive[i]->ptr(), _table_size, ncclFloat64,
        ncclSum, comm->comm(), stream);
    platform::dynload::ncclAllReduce(
        _d_negative[i]->ptr(), _d_negative[i]->ptr(), _table_size, ncclFloat64,
        ncclSum, comm->comm(), stream);
    platform::dynload::ncclAllReduce(_d_abserr[i]->ptr(), _d_abserr[i]->ptr(),
                                     _max_batch_size, ncclFloat64, ncclSum,
                                     comm->comm(), stream);
    platform::dynload::ncclAllReduce(_d_sqrerr[i]->ptr(), _d_sqrerr[i]->ptr(),
                                     _max_batch_size, ncclFloat64, ncclSum,
                                     comm->comm(), stream);
    platform::dynload::ncclAllReduce(_d_pred[i]->ptr(), _d_pred[i]->ptr(),
                                     _max_batch_size, ncclFloat64, ncclSum,
                                     comm->comm(), stream);
  }
  platform::dynload::ncclGroupEnd();
  // sync
  for (int i = 0; i < platform::GetCUDADeviceCount(); ++i) {
    cudaSetDevice(i);
    auto place = platform::CUDAPlace(i);
    auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                      platform::DeviceContextPool::Instance().Get(place))
                      ->stream();
    cudaStreamSynchronize(stream);
  }
  // restore device
  cudaSetDevice(ori_device);
}

void BasicAucCalculator::copy_data_d2h(int device) {
  // backup orginal device
  int ori_device;
  cudaGetDevice(&ori_device);
  cudaSetDevice(device);
  auto place = platform::CUDAPlace(device);
  auto stream = dynamic_cast<platform::CUDADeviceContext*>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();

  std::vector<double> h_abserr;
  std::vector<double> h_sqrerr;
  std::vector<double> h_pred;
  h_abserr.resize(_max_batch_size);
  h_sqrerr.resize(_max_batch_size);
  h_pred.resize(_max_batch_size);
  cudaMemcpyAsync(&_table[0][0], _d_negative[device]->ptr(),
                  _table_size * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&_table[1][0], _d_positive[device]->ptr(),
                  _table_size * sizeof(double), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_abserr.data(), _d_abserr[device]->ptr(),
                  _max_batch_size * sizeof(double), cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(h_sqrerr.data(), _d_sqrerr[device]->ptr(),
                  _max_batch_size * sizeof(double), cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(h_pred.data(), _d_pred[device]->ptr(),
                  _max_batch_size * sizeof(double), cudaMemcpyDeviceToHost,
                  stream);
  cudaStreamSynchronize(stream);

  cudaSetDevice(ori_device);

  _local_abserr = 0;
  _local_sqrerr = 0;
  _local_pred = 0;

  // sum in CPU
  for (int i = 0; i < _max_batch_size; ++i) {
    _local_abserr += h_abserr[i];
    _local_sqrerr += h_sqrerr[i];
    _local_pred += h_pred[i];
  }
  // restore device
  cudaSetDevice(ori_device);
}

void BasicAucCalculator::compute() {
  if (_mode_collect_in_gpu) {
    // collect data
    collect_data_nccl();
    // copy from GPU0
    copy_data_d2h(0);
  }

  double* table[2] = {&_table[0][0], &_table[1][0]};
  if (boxps::MPICluster::Ins().size() > 1) {
    boxps::MPICluster::Ins().allreduce_sum(table[0], _table_size);
    boxps::MPICluster::Ins().allreduce_sum(table[1], _table_size);
  }

  double area = 0;
  double fp = 0;
  double tp = 0;

  for (int i = _table_size - 1; i >= 0; i--) {
    double newfp = fp + table[0][i];
    double newtp = tp + table[1][i];
    area += (newfp - fp) * (tp + newtp) / 2;
    fp = newfp;
    tp = newtp;
  }

  if (fp < 1e-3 || tp < 1e-3) {
    _auc = -0.5;  // which means all nonclick or click
  } else {
    _auc = area / (fp * tp);
  }

  if (boxps::MPICluster::Ins().size() > 1) {
    // allreduce sum
    double local_err[3] = {_local_abserr, _local_sqrerr, _local_pred};
    boxps::MPICluster::Ins().allreduce_sum(local_err, 3);

    _mae = local_err[0] / (fp + tp);
    _rmse = sqrt(local_err[1] / (fp + tp));
    _predicted_ctr = local_err[2] / (fp + tp);
  } else {
    _mae = _local_abserr / (fp + tp);
    _rmse = sqrt(_local_sqrerr / (fp + tp));
    _predicted_ctr = _local_pred / (fp + tp);
  }
  _actual_ctr = tp / (fp + tp);

  _size = fp + tp;

  calculate_bucket_error();
}

void BoxWrapper::PullSparse(const paddle::platform::Place& place,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<float*>& values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size, const int expand_embed_dim,
                            const int skip_offset, bool expand_only) {
  PullSparseCase(place, keys, values, slot_lengths, hidden_size,
                 expand_embed_dim, skip_offset, expand_only);
}

void BoxWrapper::PushSparseGrad(const paddle::platform::Place& place,
                                const std::vector<const uint64_t*>& keys,
                                const std::vector<const float*>& grad_values,
                                const std::vector<int64_t>& slot_lengths,
                                const int hidden_size,
                                const int expand_embed_dim,
                                const int batch_size, const int skip_offset,
                                bool expand_only) {
  PushSparseGradCase(place, keys, grad_values, slot_lengths, hidden_size,
                     expand_embed_dim, batch_size, skip_offset, expand_only);
}

void BasicAucCalculator::calculate_bucket_error() {
  double last_ctr = -1;
  double impression_sum = 0;
  double ctr_sum = 0.0;
  double click_sum = 0.0;
  double error_sum = 0.0;
  double error_count = 0;
  double* table[2] = {&_table[0][0], &_table[1][0]};
  for (int i = 0; i < _table_size; i++) {
    double click = table[1][i];
    double show = table[0][i] + table[1][i];
    double ctr = static_cast<double>(i) / _table_size;
    if (fabs(ctr - last_ctr) > kMaxSpan) {
      last_ctr = ctr;
      impression_sum = 0.0;
      ctr_sum = 0.0;
      click_sum = 0.0;
    }
    impression_sum += show;
    ctr_sum += ctr * show;
    click_sum += click;
    double adjust_ctr = ctr_sum / impression_sum;
    double relative_error =
        sqrt((1 - adjust_ctr) / (adjust_ctr * impression_sum));
    if (relative_error < kRelativeErrorBound) {
      double actual_ctr = click_sum / impression_sum;
      double relative_ctr_error = fabs(actual_ctr / adjust_ctr - 1);
      error_sum += relative_ctr_error * impression_sum;
      error_count += impression_sum;
      last_ctr = -1;
    }
  }
  _bucket_error = error_count > 0 ? error_sum / error_count : 0.0;
}

// Deprecated: should use BeginFeedPass & EndFeedPass
void BoxWrapper::FeedPass(int date,
                          const std::vector<uint64_t>& feasgin_to_box) {
  int ret = boxps_ptr_->FeedPass(date, feasgin_to_box);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "FeedPass failed in BoxPS."));
}

void BoxWrapper::BeginFeedPass(int date, boxps::PSAgentBase** agent) {
  if (FLAGS_enable_force_mem_recyle) {
    SlotRecordPool().disable_pool(FLAGS_enbale_slotpool_auto_clear);
  } else {
    SlotRecordPool().disable_pool(FLAGS_enbale_slotpool_auto_clear &&
                                  boxps_ptr_->CheckNeedLimitMem());
  }
  int ret = boxps_ptr_->BeginFeedPass(date, *agent);
  if (FLAGS_use_gpu_replica_cache) {
    int dim = FLAGS_gpu_replica_cache_dim;
    VLOG(3) << "gpu cache dim:" << dim;
    gpu_replica_cache.emplace_back(dim);
  }
  if (input_table_dim_ > 0) {
    VLOG(3) << "lookup input dim: " << input_table_dim_;
    input_table_deque_.emplace_back(input_table_dim_);
  }
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "BeginFeedPass failed in BoxPS."));
}

void BoxWrapper::EndFeedPass(boxps::PSAgentBase* agent) {
  if (FLAGS_use_gpu_replica_cache) {
    auto& t = gpu_replica_cache.back();
    t.ToHBM();
    VLOG(0) << "gpu cache memory: " << t.GpuMemUsed() << "MB";
  }
  if (input_table_dim_ > 0) {
    auto& t = input_table_deque_.back();
    VLOG(0) << "input table size: " << t.size() << " miss: " << t.miss()
            << ", cpu memory: " << t.CpuMemUsed() << "MB";
  }
  int ret = boxps_ptr_->EndFeedPass(agent);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "EndFeedPass failed in BoxPS."));
  RelaseAgent(agent);
}

void BoxWrapper::BeginPass() {
  int ret = boxps_ptr_->BeginPass();
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "BeginPass failed in BoxPS."));
  // auto disable or enable slotrecord pool recyle memory
  SlotRecordPool().disable_pool(FLAGS_enbale_slotpool_auto_clear &&
                                boxps_ptr_->CheckNeedLimitMem());
}

void BoxWrapper::SetTestMode(bool is_test) const {
  boxps_ptr_->SetTestMode(is_test);
}

void BoxWrapper::EndPass(bool need_save_delta) {
  if (FLAGS_use_gpu_replica_cache) {
    gpu_replica_cache.pop_front();
  }
  if (input_table_dim_ > 0) {
    input_table_deque_.pop_front();
  }
  int ret = boxps_ptr_->EndPass(need_save_delta);
  PADDLE_ENFORCE_EQ(
      ret, 0, platform::errors::PreconditionNotMet("EndPass failed in BoxPS."));
  // clear all gpu memory
  if (FLAGS_enable_force_hbm_recyle) {
    for (int i = 0; i < gpu_num_; ++i) {
      memory::allocation::AllocatorFacade::Instance().Release(
          platform::CUDAPlace(i));
    }
    size_t total = 0;
    size_t available = 0;
    platform::GpuMemoryUsage(&available, &total);
    VLOG(0) << "release gpu memory total: " << (total >> 20)
            << "MB, available: " << (available >> 20) << "MB";
  }
}

void BoxWrapper::RecordReplace(std::vector<SlotRecord>* records,
                               const std::set<uint16_t>& slots) {
  VLOG(0) << "Begin RecordReplace";

  platform::Timer timer;
  timer.Start();

  std::vector<std::thread> threads;
  std::valarray<int> del_num(0, auc_runner_thread_num_);
  std::valarray<int> add_num(0, auc_runner_thread_num_);
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads.push_back(std::thread([this, records, tid, &slots, &del_num,
                                   &add_num]() {
      size_t ins_num = records->size();
      int start = tid * ins_num / auc_runner_thread_num_;
      int end = (tid + 1) * ins_num / auc_runner_thread_num_;
      VLOG(3) << "ReplaceRecord begin for thread[" << tid << "], and process ["
              << start << ", " << end << "), total ins: " << ins_num;
      for (int j = start; j < end; ++j) {
        auto record = records->at(j);
        auto info = get_auc_runner_info(record);
        auto& random_pool = random_ins_pool_list[info->pool_id_];
        FeasignValuesCandidate& candidate =
            random_pool.GetUseReplaceId(info->replaced_id_);
        record_replacers_[info->record_id_].replace(
            &record->slot_uint64_feasigns_, candidate.feasign_values_, slots,
            &del_num[tid], &add_num[tid]);
      }
      VLOG(3) << "thread[" << tid << "]: erase feasign num: " << del_num[tid]
              << " repush feasign num: " << add_num[tid];
    }));
  }
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads[tid].join();
  }

  timer.Pause();
  VLOG(0) << "End RecordReplace: " << timer.ElapsedMS() << std::endl
          << "del feasign num: " << del_num.sum() << std::endl
          << "add feasign num: " << add_num.sum();
}

void BoxWrapper::RecordReplaceBack(std::vector<SlotRecord>* records,
                                   const std::set<uint16_t>& slots) {
  VLOG(0) << "Begin RecordReplaceBack";

  platform::Timer timer;
  timer.Start();

  std::vector<std::thread> threads;
  std::valarray<int> del_num(0, auc_runner_thread_num_);
  std::valarray<int> add_num(0, auc_runner_thread_num_);
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads.push_back(
        std::thread([this, records, tid, &slots, &del_num, &add_num]() {
          size_t ins_num = records->size();
          int start = tid * ins_num / auc_runner_thread_num_;
          int end = (tid + 1) * ins_num / auc_runner_thread_num_;
          VLOG(3) << "RecordReplaceBack begin for thread[" << tid
                  << "], and process [" << start << ", " << end
                  << "), total ins: " << ins_num;
          for (int j = start; j < end; ++j) {
            auto record = records->at(j);
            auto info = get_auc_runner_info(record);
            record_replacers_[info->record_id_].replace_back(
                &record->slot_uint64_feasigns_, slots, &del_num[tid],
                &add_num[tid]);
          }
          VLOG(3) << "thread[" << tid
                  << "]: erase feasign num: " << del_num[tid]
                  << " repush feasign num: " << add_num[tid];
        }));
  }
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads[tid].join();
  }
  timer.Pause();
  VLOG(0) << "End RecordReplaceBack: " << timer.ElapsedMS() << std::endl
          << "del feasign num: " << del_num.sum() << std::endl
          << "add feasign num: " << add_num.sum();
}

void BoxWrapper::GetRandomReplace(std::vector<SlotRecord>* records) {
  VLOG(0) << "Begin GetRandomReplace";
  platform::Timer timer;
  timer.Start();

  std::lock_guard<std::mutex> lock(mutex4random_pool_);
  size_t ins_num = records->size();
  std::vector<std::thread> threads;
  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads.push_back(std::thread([this, records, tid, ins_num]() {
      int start = tid * ins_num / auc_runner_thread_num_;
      int end = (tid + 1) * ins_num / auc_runner_thread_num_;
      VLOG(3) << "GetRandomReplace begin for thread[" << tid
              << "], and process [" << start << ", " << end
              << "), total ins: " << ins_num;
      auto& random_pool = random_ins_pool_list[tid];
      for (int i = start; i < end; ++i) {
        auto record = records->at(i);
        auto info = get_auc_runner_info(record);
        info->record_id_ = i;
        info->pool_id_ = tid;
        info->replaced_id_ =
            random_pool.AddAndGet(record->slot_uint64_feasigns_);
      }
    }));
  }

  for (int tid = 0; tid < auc_runner_thread_num_; ++tid) {
    threads[tid].join();
  }
  timer.Pause();

  VLOG(0) << "End GetRandomReplace: " << timer.ElapsedMS();
}

void BoxWrapper::AddReplaceFeasign(boxps::PSAgentBase* p_agent,
                                   int feed_pass_thread_num) {
  VLOG(0) << "Begin AddReplaceFeasign";
  platform::Timer timer;
  timer.Start();

  std::lock_guard<std::mutex> lock(mutex4random_pool_);
  std::vector<std::thread> threads;
  for (int tid = 0; tid < feed_pass_thread_num; ++tid) {
    threads.push_back(std::thread([this, tid, p_agent, feed_pass_thread_num]() {
      VLOG(3) << "AddReplaceFeasign begin for thread[" << tid << "]";
      for (size_t pool_id = tid; pool_id < random_ins_pool_list.size();
           pool_id += feed_pass_thread_num) {
        auto& random_pool = random_ins_pool_list[pool_id];
        for (size_t i = 0; i < random_pool.Size(); ++i) {
          auto& candidate = random_pool.GetUseId(i);
          for (const auto& pair : candidate.feasign_values_) {
            for (const auto feasign : pair.second) {
              p_agent->AddKey(feasign, tid);
            }
          }
        }
      }
    }));
  }

  for (int tid = 0; tid < feed_pass_thread_num; ++tid) {
    threads[tid].join();
  }
  timer.Pause();

  VLOG(0) << "End AddReplaceFeasign: " << timer.ElapsedMS();
}
//================================ auc
//============================================

class MultiTaskMetricMsg : public MetricMsg {
 public:
  MultiTaskMetricMsg(const std::string& label_varname,
                     const std::string& pred_varname_list, int metric_phase,
                     const std::string& cmatch_rank_group,
                     const std::string& cmatch_rank_varname,
                     int bucket_size = 1000000) {
    label_varname_ = label_varname;
    cmatch_rank_varname_ = cmatch_rank_varname;
    metric_phase_ = metric_phase;
    calculator = new BasicAucCalculator();
    calculator->init(bucket_size);
    for (auto& cmatch_rank : string::split_string(cmatch_rank_group)) {
      const std::vector<std::string>& cur_cmatch_rank =
          string::split_string(cmatch_rank, "_");
      PADDLE_ENFORCE_EQ(
          cur_cmatch_rank.size(), 2,
          platform::errors::PreconditionNotMet("illegal multitask auc spec: %s",
                                               cmatch_rank.c_str()));
      cmatch_rank_v.emplace_back(atoi(cur_cmatch_rank[0].c_str()),
                                 atoi(cur_cmatch_rank[1].c_str()));
    }
    for (const auto& pred_varname : string::split_string(pred_varname_list)) {
      pred_v.emplace_back(pred_varname);
    }
    PADDLE_ENFORCE_EQ(cmatch_rank_v.size(), pred_v.size(),
                      platform::errors::PreconditionNotMet(
                          "cmatch_rank's size [%lu] should be equal to pred "
                          "list's size [%lu], but ther are not equal",
                          cmatch_rank_v.size(), pred_v.size()));
  }
  virtual ~MultiTaskMetricMsg() {}
  void add_data(const Scope* exe_scope,
                const paddle::platform::Place& place) override {
    std::vector<int64_t> cmatch_rank_data;
    get_data<int64_t>(exe_scope, cmatch_rank_varname_, &cmatch_rank_data);
    std::vector<int64_t> label_data;
    get_data<int64_t>(exe_scope, label_varname_, &label_data);
    size_t batch_size = cmatch_rank_data.size();
    PADDLE_ENFORCE_EQ(
        batch_size, label_data.size(),
        platform::errors::PreconditionNotMet(
            "illegal batch size: batch_size[%lu] and label_data[%lu]",
            batch_size, label_data.size()));

    std::vector<std::vector<float>> pred_data_list(pred_v.size());
    for (size_t i = 0; i < pred_v.size(); ++i) {
      get_data<float>(exe_scope, pred_v[i], &pred_data_list[i]);
    }
    for (size_t i = 0; i < pred_data_list.size(); ++i) {
      PADDLE_ENFORCE_EQ(
          batch_size, pred_data_list[i].size(),
          platform::errors::PreconditionNotMet(
              "illegal batch size: batch_size[%lu] and pred_data[%lu]",
              batch_size, pred_data_list[i].size()));
    }
    auto cal = GetCalculator();
    std::lock_guard<std::mutex> lock(cal->table_mutex());
    for (size_t i = 0; i < batch_size; ++i) {
      auto cmatch_rank_it =
          std::find(cmatch_rank_v.begin(), cmatch_rank_v.end(),
                    parse_cmatch_rank(cmatch_rank_data[i]));
      if (cmatch_rank_it != cmatch_rank_v.end()) {
        cal->add_unlock_data(
            pred_data_list[std::distance(cmatch_rank_v.begin(), cmatch_rank_it)]
                          [i],
            label_data[i]);
      }
    }
  }

 protected:
  std::vector<std::pair<int, int>> cmatch_rank_v;
  std::vector<std::string> pred_v;
  std::string cmatch_rank_varname_;
};
class CmatchRankMetricMsg : public MetricMsg {
 public:
  CmatchRankMetricMsg(const std::string& label_varname,
                      const std::string& pred_varname, int metric_phase,
                      const std::string& cmatch_rank_group,
                      const std::string& cmatch_rank_varname,
                      bool ignore_rank = false, int bucket_size = 1000000) {
    label_varname_ = label_varname;
    pred_varname_ = pred_varname;
    cmatch_rank_varname_ = cmatch_rank_varname;
    metric_phase_ = metric_phase;
    ignore_rank_ = ignore_rank;
    calculator = new BasicAucCalculator();
    calculator->init(bucket_size);
    for (auto& cmatch_rank : string::split_string(cmatch_rank_group)) {
      if (ignore_rank) {  // CmatchAUC
        cmatch_rank_v.emplace_back(atoi(cmatch_rank.c_str()), 0);
        continue;
      }
      const std::vector<std::string>& cur_cmatch_rank =
          string::split_string(cmatch_rank, "_");
      PADDLE_ENFORCE_EQ(
          cur_cmatch_rank.size(), 2,
          platform::errors::PreconditionNotMet(
              "illegal cmatch_rank auc spec: %s", cmatch_rank.c_str()));
      cmatch_rank_v.emplace_back(atoi(cur_cmatch_rank[0].c_str()),
                                 atoi(cur_cmatch_rank[1].c_str()));
    }
  }
  virtual ~CmatchRankMetricMsg() {}
  void add_data(const Scope* exe_scope,
                const paddle::platform::Place& place) override {
    std::vector<int64_t> cmatch_rank_data;
    get_data<int64_t>(exe_scope, cmatch_rank_varname_, &cmatch_rank_data);
    std::vector<int64_t> label_data;
    get_data<int64_t>(exe_scope, label_varname_, &label_data);
    std::vector<float> pred_data;
    get_data<float>(exe_scope, pred_varname_, &pred_data);
    size_t batch_size = cmatch_rank_data.size();
    PADDLE_ENFORCE_EQ(
        batch_size, label_data.size(),
        platform::errors::PreconditionNotMet(
            "illegal batch size: cmatch_rank[%lu] and label_data[%lu]",
            batch_size, label_data.size()));
    PADDLE_ENFORCE_EQ(
        batch_size, pred_data.size(),
        platform::errors::PreconditionNotMet(
            "illegal batch size: cmatch_rank[%lu] and pred_data[%lu]",
            batch_size, pred_data.size()));
    auto cal = GetCalculator();
    std::lock_guard<std::mutex> lock(cal->table_mutex());
    for (size_t i = 0; i < batch_size; ++i) {
      const auto& cur_cmatch_rank = parse_cmatch_rank(cmatch_rank_data[i]);
      for (size_t j = 0; j < cmatch_rank_v.size(); ++j) {
        bool is_matched = false;
        if (ignore_rank_) {
          is_matched = cmatch_rank_v[j].first == cur_cmatch_rank.first;
        } else {
          is_matched = cmatch_rank_v[j] == cur_cmatch_rank;
        }
        if (is_matched) {
          cal->add_unlock_data(pred_data[i], label_data[i]);
          break;
        }
      }
    }
  }

 protected:
  std::vector<std::pair<int, int>> cmatch_rank_v;
  std::string cmatch_rank_varname_;
  bool ignore_rank_;
};
class MaskMetricMsg : public MetricMsg {
 public:
  MaskMetricMsg(const std::string& label_varname,
                const std::string& pred_varname, int metric_phase,
                const std::string& mask_varname, int bucket_size = 1000000,
                bool mode_collect_in_gpu = false, int max_batch_size = 0) {
    label_varname_ = label_varname;
    pred_varname_ = pred_varname;
    mask_varname_ = mask_varname;
    metric_phase_ = metric_phase;
    calculator = new BasicAucCalculator(mode_collect_in_gpu);
    calculator->init(bucket_size, max_batch_size);
  }
  virtual ~MaskMetricMsg() {}
  void add_data(const Scope* exe_scope,
                const paddle::platform::Place& place) override {
    int label_len = 0;
    const int64_t* label_data = NULL;
    get_data<int64_t>(exe_scope, label_varname_, &label_data, &label_len);

    int pred_len = 0;
    const float* pred_data = NULL;
    get_data<float>(exe_scope, pred_varname_, &pred_data, &pred_len);

    int mask_len = 0;
    const int64_t* mask_data = NULL;
    get_data<int64_t>(exe_scope, mask_varname_, &mask_data, &mask_len);
    PADDLE_ENFORCE_EQ(label_len, mask_len,
                      platform::errors::PreconditionNotMet(
                          "the predict data length should be consistent with "
                          "the label data length"));
    auto cal = GetCalculator();
    cal->add_mask_data(pred_data, label_data, mask_data, label_len, place);
  }

 protected:
  std::string mask_varname_;
};

class MultiMaskMetricMsg : public MetricMsg {
 public:
  MultiMaskMetricMsg(const std::string& label_varname,
                     const std::string& pred_varname, int metric_phase,
                     const std::string& mask_varname_list,
                     const std::string& mask_varvalue_list,
                     int bucket_size = 1000000,
                     bool mode_collect_in_gpu = false, int max_batch_size = 0) {
    label_varname_ = label_varname;
    pred_varname_ = pred_varname;
    mask_varname_list_ = string::split_string(mask_varname_list, " ");
    const std::vector<std::string> tmp_val_lst =
        string::split_string(mask_varvalue_list, " ");
    for (const auto& it : tmp_val_lst) {
      mask_varvalue_list_.emplace_back(atoi(it.c_str()));
    }
    PADDLE_ENFORCE_EQ(
        mask_varname_list_.size(), mask_varvalue_list_.size(),
        platform::errors::PreconditionNotMet(
            "mast var num[%zu] should be equal to mask val num[%zu]",
            mask_varname_list_.size(), mask_varvalue_list_.size()));

    metric_phase_ = metric_phase;
    calculator = new BasicAucCalculator(mode_collect_in_gpu);
    calculator->init(bucket_size);
  }
  virtual ~MultiMaskMetricMsg() {}
  void add_data(const Scope* exe_scope,
                const paddle::platform::Place& place) override {
    std::vector<int64_t> label_data;
    get_data<int64_t>(exe_scope, label_varname_, &label_data);

    std::vector<float> pred_data;
    get_data<float>(exe_scope, pred_varname_, &pred_data);

    PADDLE_ENFORCE_EQ(label_data.size(), pred_data.size(),
                      platform::errors::PreconditionNotMet(
                          "the predict data length should be consistent with "
                          "the label data length"));

    std::vector<std::vector<int64_t>> mask_value_data_list(
        mask_varname_list_.size());
    for (size_t name_idx = 0; name_idx < mask_varname_list_.size();
         ++name_idx) {
      get_data<int64_t>(exe_scope, mask_varname_list_[name_idx],
                        &mask_value_data_list[name_idx]);
      PADDLE_ENFORCE_EQ(
          label_data.size(), mask_value_data_list[name_idx].size(),
          platform::errors::PreconditionNotMet(
              "the label data length[%d] should be consistent with "
              "the %s[%zu] length",
              label_data.size(), mask_value_data_list[name_idx].size()));
    }
    auto cal = GetCalculator();
    std::lock_guard<std::mutex> lock(cal->table_mutex());
    size_t batch_size = label_data.size();
    bool flag = true;
    for (size_t ins_idx = 0; ins_idx < batch_size; ++ins_idx) {
      flag = true;
      for (size_t val_idx = 0; val_idx < mask_varvalue_list_.size();
           ++val_idx) {
        if (mask_value_data_list[val_idx][ins_idx] !=
            mask_varvalue_list_[val_idx]) {
          flag = false;
          break;
        }
      }
      if (flag) {
        cal->add_unlock_data(pred_data[ins_idx], label_data[ins_idx]);
      }
    }
  }

 protected:
  std::vector<int> mask_varvalue_list_;
  std::vector<std::string> mask_varname_list_;
  std::string cmatch_rank_varname_;
};

class FloatMaskMetricMsg : public MetricMsg {
 public:
  FloatMaskMetricMsg(const std::string& label_varname,
                     const std::string& pred_varname, int metric_phase,
                     const std::string& mask_varname, int bucket_size = 1000000,
                     bool mode_collect_in_gpu = false, int max_batch_size = 0) {
    label_varname_ = label_varname;
    pred_varname_ = pred_varname;
    mask_varname_ = mask_varname;
    metric_phase_ = metric_phase;
    calculator = new BasicAucCalculator(mode_collect_in_gpu);
    calculator->init(bucket_size, max_batch_size);
  }
  virtual ~FloatMaskMetricMsg() {}
  void add_data(const Scope* exe_scope,
                const paddle::platform::Place& place) override {
    int label_len = 0;
    const float* label_data = NULL;
    get_data<float>(exe_scope, label_varname_, &label_data, &label_len);

    int pred_len = 0;
    const float* pred_data = NULL;
    get_data<float>(exe_scope, pred_varname_, &pred_data, &pred_len);

    int mask_len = 0;
    const int64_t* mask_data = NULL;
    get_data<int64_t>(exe_scope, mask_varname_, &mask_data, &mask_len);
    PADDLE_ENFORCE_EQ(label_len, mask_len,
                      platform::errors::PreconditionNotMet(
                          "the predict data length should be consistent with "
                          "the label data length"));
    auto cal = GetCalculator();
    cal->add_float_mask_data(pred_data, label_data, mask_data, label_len,
                             place);
  }

 protected:
  std::string mask_varname_;
};

class CmatchRankMaskMetricMsg : public MetricMsg {
 public:
  CmatchRankMaskMetricMsg(const std::string& label_varname,
                          const std::string& pred_varname, int metric_phase,
                          const std::string& cmatch_rank_group,
                          const std::string& cmatch_rank_varname,
                          bool ignore_rank = false,
                          const std::string& mask_varname = "",
                          int bucket_size = 1000000) {
    label_varname_ = label_varname;
    pred_varname_ = pred_varname;
    cmatch_rank_varname_ = cmatch_rank_varname;
    metric_phase_ = metric_phase;
    ignore_rank_ = ignore_rank;
    mask_varname_ = mask_varname;
    calculator = new BasicAucCalculator();
    calculator->init(bucket_size);
    for (auto& cmatch_rank : string::split_string(cmatch_rank_group)) {
      if (ignore_rank) {  // CmatchAUC
        cmatch_rank_v.emplace_back(atoi(cmatch_rank.c_str()), 0);
        continue;
      }
      const std::vector<std::string>& cur_cmatch_rank =
          string::split_string(cmatch_rank, "_");
      PADDLE_ENFORCE_EQ(
          cur_cmatch_rank.size(), 2,
          platform::errors::PreconditionNotMet(
              "illegal cmatch_rank auc spec: %s", cmatch_rank.c_str()));
      cmatch_rank_v.emplace_back(atoi(cur_cmatch_rank[0].c_str()),
                                 atoi(cur_cmatch_rank[1].c_str()));
    }
  }
  virtual ~CmatchRankMaskMetricMsg() {}
  void add_data(const Scope* exe_scope,
                const paddle::platform::Place& place) override {
    std::vector<int64_t> cmatch_rank_data;
    get_data<int64_t>(exe_scope, cmatch_rank_varname_, &cmatch_rank_data);
    std::vector<int64_t> label_data;
    get_data<int64_t>(exe_scope, label_varname_, &label_data);
    std::vector<float> pred_data;
    get_data<float>(exe_scope, pred_varname_, &pred_data);
    size_t batch_size = cmatch_rank_data.size();
    PADDLE_ENFORCE_EQ(
        batch_size, label_data.size(),
        platform::errors::PreconditionNotMet(
            "illegal batch size: cmatch_rank[%lu] and label_data[%lu]",
            batch_size, label_data.size()));
    PADDLE_ENFORCE_EQ(
        batch_size, pred_data.size(),
        platform::errors::PreconditionNotMet(
            "illegal batch size: cmatch_rank[%lu] and pred_data[%lu]",
            batch_size, pred_data.size()));

    std::vector<int64_t> mask_data;
    if (!mask_varname_.empty()) {
      get_data<int64_t>(exe_scope, mask_varname_, &mask_data);
      PADDLE_ENFORCE_EQ(
          batch_size, mask_data.size(),
          platform::errors::PreconditionNotMet(
              "illegal batch size: cmatch_rank[%lu] and mask_data[%lu]",
              batch_size, mask_data.size()));
    }

    auto cal = GetCalculator();
    std::lock_guard<std::mutex> lock(cal->table_mutex());
    for (size_t i = 0; i < batch_size; ++i) {
      const auto& cur_cmatch_rank = parse_cmatch_rank(cmatch_rank_data[i]);
      for (size_t j = 0; j < cmatch_rank_v.size(); ++j) {
        if (!mask_data.empty() && !mask_data[i]) {
          continue;
        }
        bool is_matched = false;
        if (ignore_rank_) {
          is_matched = cmatch_rank_v[j].first == cur_cmatch_rank.first;
        } else {
          is_matched = cmatch_rank_v[j] == cur_cmatch_rank;
        }
        if (is_matched) {
          cal->add_unlock_data(pred_data[i], label_data[i]);
          break;
        }
      }
    }
  }

 protected:
  std::vector<std::pair<int, int>> cmatch_rank_v;
  std::string cmatch_rank_varname_;
  bool ignore_rank_;
  std::string mask_varname_;
};

const std::vector<std::string> BoxWrapper::GetMetricNameList(
    int metric_phase) const {
  VLOG(0) << "Want to Get metric phase: " << metric_phase;
  if (metric_phase == -1) {
    return metric_name_list_;
  } else {
    std::vector<std::string> ret;
    for (const auto& name : metric_name_list_) {
      const auto iter = metric_lists_.find(name);
      PADDLE_ENFORCE_NE(iter, metric_lists_.end(),
                        platform::errors::InvalidArgument(
                            "The metric name you provided is not registered."));

      if (iter->second->MetricPhase() == metric_phase) {
        VLOG(3) << name << "'s phase is " << iter->second->MetricPhase()
                << ", we want";
        ret.push_back(name);
      } else {
        VLOG(3) << name << "'s phase is " << iter->second->MetricPhase()
                << ", not we want";
      }
    }
    return ret;
  }
}

void BoxWrapper::InitMetric(const std::string& method, const std::string& name,
                            const std::string& label_varname,
                            const std::string& pred_varname,
                            const std::string& cmatch_rank_varname,
                            const std::string& mask_varname, int metric_phase,
                            const std::string& cmatch_rank_group,
                            bool ignore_rank, int bucket_size,
                            bool mode_collect_in_gpu, int max_batch_size,
                            const std::string& sample_scale_varname) {
  if (method == "AucCalculator") {
    metric_lists_.emplace(
        name, new MetricMsg(label_varname, pred_varname, metric_phase,
                            bucket_size, mode_collect_in_gpu, max_batch_size,
                            sample_scale_varname));
  } else if (method == "MultiTaskAucCalculator") {
    metric_lists_.emplace(
        name, new MultiTaskMetricMsg(label_varname, pred_varname, metric_phase,
                                     cmatch_rank_group, cmatch_rank_varname,
                                     bucket_size));
  } else if (method == "CmatchRankAucCalculator") {
    metric_lists_.emplace(
        name, new CmatchRankMetricMsg(label_varname, pred_varname, metric_phase,
                                      cmatch_rank_group, cmatch_rank_varname,
                                      ignore_rank, bucket_size));
  } else if (method == "MaskAucCalculator") {
    metric_lists_.emplace(
        name, new MaskMetricMsg(label_varname, pred_varname, metric_phase,
                                mask_varname, bucket_size, mode_collect_in_gpu,
                                max_batch_size));
  } else if (method == "MultiMaskAucCalculator") {
    metric_lists_.emplace(
        name,
        new MultiMaskMetricMsg(label_varname, pred_varname, metric_phase,
                               mask_varname, cmatch_rank_group, bucket_size,
                               mode_collect_in_gpu, max_batch_size));
  } else if (method == "CmatchRankMaskAucCalculator") {
    metric_lists_.emplace(
        name, new CmatchRankMaskMetricMsg(
                  label_varname, pred_varname, metric_phase, cmatch_rank_group,
                  cmatch_rank_varname, ignore_rank, mask_varname, bucket_size));
  } else if (method == "FloatMaskAucCalculator") {
    metric_lists_.emplace(
        name, new FloatMaskMetricMsg(label_varname, pred_varname, metric_phase,
                                     mask_varname, bucket_size,
                                     mode_collect_in_gpu, max_batch_size));
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "PaddleBox only support AucCalculator, MultiTaskAucCalculator, "
        "CmatchRankAucCalculator, MaskAucCalculator, FloatMaskAucCalculator "
        "and "
        "CmatchRankMaskAucCalculator"));
  }
  metric_name_list_.emplace_back(name);
}

const std::vector<double> BoxWrapper::GetMetricMsg(const std::string& name) {
  const auto iter = metric_lists_.find(name);
  PADDLE_ENFORCE_NE(iter, metric_lists_.end(),
                    platform::errors::InvalidArgument(
                        "The metric name you provided is not registered."));
  std::vector<double> metric_return_values_(8, 0.0);
  auto* auc_cal_ = iter->second->GetCalculator();
  auc_cal_->compute();
  metric_return_values_[0] = auc_cal_->auc();
  metric_return_values_[1] = auc_cal_->bucket_error();
  metric_return_values_[2] = auc_cal_->mae();
  metric_return_values_[3] = auc_cal_->rmse();
  metric_return_values_[4] = auc_cal_->actual_ctr();
  metric_return_values_[5] = auc_cal_->predicted_ctr();
  metric_return_values_[6] = auc_cal_->actual_ctr() / auc_cal_->predicted_ctr();
  metric_return_values_[7] = auc_cal_->size();
  auc_cal_->reset();
  return metric_return_values_;
}
void BoxWrapper::PrintSyncTimer(int device, double train_span) {
  auto& dev = device_caches_[device];
  size_t total = 0;
  size_t available = 0;
  size_t used = memory::allocation::AllocatorFacade::Instance().GetTotalMemInfo(
      platform::CUDAPlace(device), &total, &available);
  LOG(WARNING) << "gpu: " << device << ", phase: " << phase_
               << ", train dnn: " << train_span
               << ", sparse pull span: " << dev.all_pull_timer.ElapsedSec()
               << ", dedup span: " << dev.pull_dedup_timer.ElapsedSec()
               << ", boxps span: " << dev.boxps_pull_timer.ElapsedSec()
               << ", push span: " << dev.all_push_timer.ElapsedSec()
               << ", copy span:" << dev.copy_push_timer.ElapsedSec()
               << ", boxps span:" << dev.boxps_push_timer.ElapsedSec()
               << ", dense nccl:" << dev.dense_nccl_timer.ElapsedSec()
               << ", sync stream:" << dev.dense_sync_timer.ElapsedSec()
               << ", wrapper gpu memory:" << dev.GpuMemUsed() << "MB"
               << ", total cache:" << (total >> 20) << "MB"
               << ", used:" << (used >> 20) << "MB"
               << ", free:" << (available >> 20) << "MB";
  dev.ResetTimer();
}
// get feature offset info
void BoxWrapper::GetFeatureOffsetInfo(void) {
  feature_pull_size_ = boxps_ptr_->GetFeaturePullSize(pull_info_);
  feature_push_size_ = boxps_ptr_->GetFeaturePushSize(push_info_);
  pull_float_num_ = feature_pull_size_ / sizeof(float);
  push_float_num_ = feature_push_size_ / sizeof(float);
  // set cvm offset
  cvm_offset_ = pull_info_.embedx_size - pull_info_.show;
  LOG(WARNING) << "feature pull info byte=" << feature_pull_size_
               << ", float num=" << pull_float_num_
               << "[quant=" << pull_info_.is_quant
               << ", slot=" << pull_info_.slot << ", show=" << pull_info_.show
               << ", clk=" << pull_info_.clk
               << ", embed_num=" << pull_info_.embed_num
               << ", embed_w=" << pull_info_.embed_w
               << ", embedx_size=" << pull_info_.embedx_size
               << ", embedx=" << pull_info_.embedx
               << ", expand_size=" << pull_info_.expand_size
               << ", expand=" << pull_info_.expand
               << "], push info byte=" << feature_push_size_
               << ", float num=" << push_float_num_
               << "[slot=" << push_info_.slot << ", show=" << push_info_.show
               << ", clk=" << push_info_.clk
               << ", embed_num=" << push_info_.embed_num
               << ", embed_g=" << push_info_.embed_g
               << ", embedx_g=" << push_info_.embedx_g
               << ", expand_g=" << push_info_.expand_g
               << "], cvm_offset=" << cvm_offset_;

  // pull push cvm offset check
  PADDLE_ENFORCE_EQ(cvm_offset_, push_info_.embedx_g - push_info_.show,
                    platform::errors::PreconditionNotMet(
                        "set pull push cvm offset error in boxps."));
  PADDLE_ENFORCE_EQ(feature_pull_size_ % sizeof(float), 0,
                    platform::errors::PreconditionNotMet(
                        "feature pull size must sizeof(float) number."));
  PADDLE_ENFORCE_EQ(feature_push_size_ % sizeof(float), 0,
                    platform::errors::PreconditionNotMet(
                        "feature push size must sizeof(float) number."));
}

//============================== other =====================================

boxps::PSAgentBase* BoxWrapper::GetAgent() {
  boxps::PSAgentBase* p_agent = nullptr;
  std::lock_guard<std::mutex> lock(mutex_);
  if (psagents_.empty()) {
    p_agent = boxps::PSAgentBase::GetIns(feedpass_thread_num_);
    p_agent->Init();
  } else {
    p_agent = psagents_.front();
    psagents_.pop_front();
  }
  return p_agent;
}
void BoxWrapper::RelaseAgent(boxps::PSAgentBase* agent) {
  std::lock_guard<std::mutex> lock(mutex_);
  psagents_.push_back(agent);
}
void BoxWrapper::InitializeGPUAndLoadModel(
    const char* conf_file, const std::vector<int>& slot_vector,
    const std::vector<std::string>& slot_omit_in_feedpass,
    const std::string& model_path, const std::map<std::string, float>& lr_map) {
  if (nullptr != s_instance_) {
    VLOG(3) << "Begin InitializeGPU";
    std::vector<cudaStream_t*> stream_list;
    gpu_num_ = platform::GetCUDADeviceCount();
    CHECK(gpu_num_ <= MAX_GPU_NUM) << "gpu card num: " << gpu_num_
                                   << ", more than max num: " << MAX_GPU_NUM;
    for (int i = 0; i < gpu_num_; ++i) {
      VLOG(3) << "before get context i[" << i << "]";
      platform::CUDADeviceContext* context =
          dynamic_cast<platform::CUDADeviceContext*>(
              platform::DeviceContextPool::Instance().Get(
                  platform::CUDAPlace(i)));
      stream_list_[i] = context->stream();
      stream_list.push_back(&stream_list_[i]);
    }
    VLOG(2) << "Begin call InitializeGPU in BoxPS";
    // the second parameter is useless
    boxps_ptr_->InitializeGPUAndLoadModel(conf_file, -1, stream_list,
                                          slot_vector, model_path);
    for (const auto& slot_name : slot_omit_in_feedpass) {
      slot_name_omited_in_feedpass_.insert(slot_name);
    }
    slot_vector_ = slot_vector;
    device_caches_ = new DeviceBoxData[gpu_num_];

    VLOG(0) << "lr_map.size(): " << lr_map.size();
    for (const auto e : lr_map) {
      VLOG(0) << e.first << "'s lr is " << e.second;
      if (e.first.find("param") != std::string::npos) {
        lr_map_[e.first + ".w_0"] = e.second;
        lr_map_[e.first + ".b_0"] = e.second;
      }
    }
  }
}

void BoxWrapper::Finalize() {
  VLOG(3) << "Begin Finalize";
  if (s_instance_ == nullptr) {
    return;
  }
  if (file_manager_ != nullptr) {
    file_manager_->destory();
    file_manager_ = nullptr;
  }
  if (data_shuffle_ != nullptr) {
    data_shuffle_->destory();
    data_shuffle_ = nullptr;
  }
  if (boxps_ptr_ != nullptr) {
    boxps_ptr_->Finalize();
    boxps_ptr_ = nullptr;
  }
  if (!psagents_.empty()) {
    for (auto agent : psagents_) {
      delete agent;
    }
    psagents_.clear();
  }
  if (device_caches_ != nullptr) {
    delete[] device_caches_;
    device_caches_ = nullptr;
  }
  s_instance_ = nullptr;
}

void BoxWrapper::ReleasePool(void) {
  // after one day train release memory pool slot record
  platform::Timer timer;
  timer.Start();
  size_t capacity = SlotRecordPool().capacity();
  SlotRecordPool().clear();
  SlotRecordPool().disable_pool(false);
  timer.Pause();
  LOG(WARNING) << "ReleasePool Size=" << capacity
               << ", Time=" << timer.ElapsedSec() << "sec";
}

const std::string BoxWrapper::SaveBase(const char* batch_model_path,
                                       const char* xbox_model_path,
                                       const std::string& date) {
  VLOG(3) << "Begin SaveBase";
  int day_id = -1;
  if (!date.empty()) {
    PADDLE_ENFORCE_EQ(
        date.length(), 8,
        platform::errors::PreconditionNotMet(
            "date[%s] is invalid, correct example is 20190817", date.c_str()));
    int year = std::stoi(date.substr(0, 4));
    int month = std::stoi(date.substr(4, 2));
    int day = std::stoi(date.substr(6, 2));

    struct std::tm b;
    b.tm_year = year - 1900;
    b.tm_mon = month - 1;
    b.tm_mday = day;
    b.tm_hour = FLAGS_fix_dayid ? 8 : 0;
    b.tm_min = b.tm_sec = 0;
    day_id = std::mktime(&b) / 86400;
  }
  std::string ret_str;
  int ret =
      boxps_ptr_->SaveBase(batch_model_path, xbox_model_path, ret_str, day_id);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "SaveBase failed in BoxPS."));
  return ret_str;
}

const std::string BoxWrapper::SaveDelta(const char* xbox_model_path) {
  VLOG(3) << "Begin SaveDelta";
  std::string ret_str;
  int ret = boxps_ptr_->SaveDelta(xbox_model_path, ret_str);
  PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
                                "SaveDelta failed in BoxPS."));
  return ret_str;
}
// load ssd2mem
bool BoxWrapper::LoadSSD2Mem(const std::string& date) {
  VLOG(3) << "Begin Load SSD to Memory";
  int year = std::stoi(date.substr(0, 4));
  int month = std::stoi(date.substr(4, 2));
  int day = std::stoi(date.substr(6, 2));

  struct std::tm b;
  b.tm_year = year - 1900;
  b.tm_mon = month - 1;
  b.tm_mday = day;
  b.tm_hour = FLAGS_fix_dayid ? 8 : 0;
  b.tm_min = b.tm_sec = 0;
  std::time_t seconds_from_1970 = std::mktime(&b);
  int day_id = seconds_from_1970 / 86400;
  return boxps_ptr_->LoadSSD2Mem(day_id);
}
//===================== box filemgr ===============================
BoxFileMgr::BoxFileMgr() {}
BoxFileMgr::~BoxFileMgr() { destory(); }
bool BoxFileMgr::init(const std::string& fs_name, const std::string& fs_ugi,
                      const std::string& conf_path) {
  if (mgr_ != nullptr) {
    mgr_->destory();
  }
  mgr_.reset(boxps::PaddleFileMgr::New());
  auto split = fs_ugi.find(",");
  std::string user = fs_ugi.substr(0, split);
  std::string pwd = fs_ugi.substr(split + 1);
  bool ret = mgr_->initialize(fs_name, user, pwd, conf_path);
  if (!ret) {
    LOG(WARNING) << "init afs api[" << fs_name << "," << fs_ugi << ","
                 << conf_path << "] failed";
    mgr_ = nullptr;
  }
  return ret;
}
void BoxFileMgr::destory(void) {
  if (mgr_ == nullptr) {
    return;
  }
  mgr_->destory();
  mgr_ = nullptr;
}
std::vector<std::string> BoxFileMgr::list_dir(const std::string& path) {
  std::vector<std::string> files;
  if (!mgr_->list_dir(path, files)) {
    LOG(WARNING) << "list dir path:[" << path << "] failed";
  }
  return files;
}
bool BoxFileMgr::makedir(const std::string& path) {
  return mgr_->makedir(path);
}
bool BoxFileMgr::exists(const std::string& path) { return mgr_->exists(path); }
bool BoxFileMgr::down(const std::string& remote, const std::string& local) {
  return mgr_->down(remote, local);
}
bool BoxFileMgr::upload(const std::string& local, const std::string& remote) {
  return mgr_->upload(local, remote);
}
bool BoxFileMgr::remove(const std::string& path) { return mgr_->remove(path); }
int64_t BoxFileMgr::file_size(const std::string& path) {
  return mgr_->file_size(path);
}
std::vector<std::pair<std::string, int64_t>> BoxFileMgr::dus(
    const std::string& path) {
  std::vector<std::pair<std::string, int64_t>> files;
  if (!mgr_->dus(path, files)) {
    LOG(WARNING) << "dus dir path:[" << path << "] failed";
  }
  return files;
}
bool BoxFileMgr::truncate(const std::string& path, const size_t len) {
  return mgr_->truncate(path, len);
}
bool BoxFileMgr::touch(const std::string& path) { return mgr_->touch(path); }
bool BoxFileMgr::rename(const std::string& src, const std::string& dest) {
  return mgr_->rename(src, dest);
}
std::vector<std::pair<std::string, int64_t>> BoxFileMgr::list_info(
    const std::string& path) {
  std::vector<std::pair<std::string, int64_t>> files;
  if (!mgr_->list_info(path, files)) {
    LOG(WARNING) << "list dir info path:[" << path << "] failed";
  }
  return files;
}
int64_t BoxFileMgr::count(const std::string& path) { return mgr_->count(path); }

}  // end namespace framework
}  // end namespace paddle
#endif
