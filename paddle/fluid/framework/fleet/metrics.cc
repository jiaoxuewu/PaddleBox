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

#include <algorithm>
#include <ctime>
#include <memory>
#include <numeric>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#ifdef PADDLE_WITH_BOX_PS
#include <boxps_extends.h>
#endif

#if defined(PADDLE_WITH_PSLIB) || defined(PADDLE_WITH_PSCORE) || defined(PADDLE_WITH_BOX_PS)
namespace paddle {
namespace framework {

std::shared_ptr<Metric> Metric::s_instance_ = nullptr;

void BasicAucCalculator::add_unlock_data(double pred, int label) {
  PADDLE_ENFORCE_GE(pred, 0.0,
      platform::errors::PreconditionNotMet("pred should be greater than 0, pred=%f", pred));
  PADDLE_ENFORCE_LE(pred, 1.0,
      platform::errors::PreconditionNotMet("pred should be lower than 1, pred=%f", pred));
  PADDLE_ENFORCE_EQ(label * label, label,
      platform::errors::PreconditionNotMet(
          "label must be equal to 0 or 1, but its value is: %d", label));

  int pos = static_cast<int>(pred * _table_size);
  _local_abserr += fabs(pred - label);
  _local_sqrerr += (pred - label) * (pred - label);
  _local_pred += pred;
  ++_table[label][pos];
}

void BasicAucCalculator::add_unlock_data(double pred, int label, float sample_scale) {
  PADDLE_ENFORCE_GE(pred, 0.0,
      platform::errors::PreconditionNotMet("pred should be greater than 0, pred=%f", pred));
  PADDLE_ENFORCE_LE(pred, 1.0,
      platform::errors::PreconditionNotMet("pred should be lower than 1, pred=%f", pred));
  PADDLE_ENFORCE_EQ(label * label, label,
      platform::errors::PreconditionNotMet(
          "label must be equal to 0 or 1, but its value is: %d", label));

  int pos = static_cast<int>(pred * _table_size);
  _local_abserr += fabs(pred - label);
  _local_sqrerr += (pred - label) * (pred - label);
  _local_pred += pred * sample_scale;
  _table[label][pos] += sample_scale;
}

void BasicAucCalculator::add_unlock_data_with_float_label(double pred, double label) {
  PADDLE_ENFORCE_GE(pred, 0.0, platform::errors::PreconditionNotMet(
                                   "pred should be greater than 0"));
  PADDLE_ENFORCE_LE(pred, 1.0, platform::errors::PreconditionNotMet(
                                   "pred should be lower than 1"));

  int pos = static_cast<int>(pred * _table_size);
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

void BasicAucCalculator::add_unlock_data_with_continue_label(double pred,
                                                             double label) {
  _local_abserr += fabs(pred - label);
  _local_sqrerr += (pred - label) * (pred - label);
  _local_pred += pred;
  _local_label += label;
  ++_local_total_num;
}

void BasicAucCalculator::add_nan_inf_unlock_data(float pred, int label){
  _size++;
  if(std::isnan(pred)){
    ++_nan_cnt;
  } else if(std::isinf(pred)){
    ++_inf_cnt;
  }
}

void BasicAucCalculator::add_data(const float* d_pred,
                                  const int64_t* d_label,
                                  int batch_size,
                                  const paddle::platform::Place& place_pred,
                                  const paddle::platform::Place& place_label) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<int64_t> h_label;
  const float* add_pred = d_pred;
  const int64_t* add_label = d_label;
  if (platform::is_gpu_place(place_pred) || platform::is_xpu_place(place_pred)) {
    h_pred.resize(batch_size);
    SyncCopyD2H(h_pred.data(), d_pred, batch_size, place_pred);
    add_pred = h_pred.data();
  }
  if (platform::is_gpu_place(place_label) || platform::is_xpu_place(place_label)) {
    h_label.resize(batch_size);
    SyncCopyD2H(h_label.data(), d_label, batch_size, place_label);
    add_label = h_label.data();
  }
  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    add_unlock_data(add_pred[i], add_label[i]);
  }
}

void BasicAucCalculator::add_sample_data(
    const float* d_pred,
    const int64_t* d_label,
    const std::vector<float>& d_sample_scale,
    int batch_size,
    const paddle::platform::Place& place_pred,
    const paddle::platform::Place& place_label) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<int64_t> h_label;
  const float* add_pred = d_pred;
  const int64_t* add_label = d_label;
  if (platform::is_gpu_place(place_pred) || platform::is_xpu_place(place_pred)) {
    h_pred.resize(batch_size);
    SyncCopyD2H(h_pred.data(), d_pred, batch_size, place_pred);
    add_pred = h_pred.data();
  }
  if (platform::is_gpu_place(place_label) || platform::is_xpu_place(place_label)) {
    h_label.resize(batch_size);
    SyncCopyD2H(h_label.data(), d_label, batch_size, place_label);
    add_label = h_label.data();
  }
  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    add_unlock_data(add_pred[i], add_label[i], d_sample_scale[i]);
  }
}

// add mask data
void BasicAucCalculator::add_mask_data(const float* d_pred,
                                       const int64_t* d_label,
                                       const int64_t* d_mask,
                                       int batch_size,
                                       const paddle::platform::Place& place_pred,
                                       const paddle::platform::Place& place_label,
                                       const paddle::platform::Place& place_mask) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<int64_t> h_label;
  thread_local std::vector<int64_t> h_mask;
  const float* add_pred = d_pred;
  const int64_t* add_label = d_label;
  const int64_t* add_mask = d_mask;
  if (platform::is_gpu_place(place_pred) || platform::is_xpu_place(place_pred)) {
    h_pred.resize(batch_size);
    SyncCopyD2H(h_pred.data(), d_pred, batch_size, place_pred);
    add_pred = h_pred.data();
  }
  if (platform::is_gpu_place(place_label) || platform::is_xpu_place(place_label)) {
    h_label.resize(batch_size);
    SyncCopyD2H(h_label.data(), d_label, batch_size, place_label);
    add_label = h_label.data();
  }
  if (platform::is_gpu_place(place_mask) || platform::is_xpu_place(place_mask)) {
    h_mask.resize(batch_size);
    SyncCopyD2H(h_mask.data(), d_mask, batch_size, place_mask);
    add_mask = h_mask.data();
  }
  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    if (add_mask[i]) {
      add_unlock_data(add_pred[i], add_label[i]);
    }
  }
}
// add float mask data
void BasicAucCalculator::add_float_mask_data(const float* d_pred,
                                             const float* d_label,
                                             const int64_t* d_mask, int batch_size,
                                             const paddle::platform::Place& place_pred,
                                             const paddle::platform::Place& place_label,
                                             const paddle::platform::Place& place_mask) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<float> h_label;
  thread_local std::vector<int64_t> h_mask;
  const float* add_pred = d_pred;
  const float* add_label = d_label;
  const int64_t* add_mask = d_mask;
  if (platform::is_gpu_place(place_pred) || platform::is_xpu_place(place_pred)) {
    h_pred.resize(batch_size);
    SyncCopyD2H(h_pred.data(), d_pred, batch_size, place_pred);
    add_pred = h_pred.data();
  }
  if (platform::is_gpu_place(place_label) || platform::is_xpu_place(place_label)) {
    h_label.resize(batch_size);
    SyncCopyD2H(h_label.data(), d_label, batch_size, place_label);
    add_label = h_label.data();
  }
  if (platform::is_gpu_place(place_mask) || platform::is_xpu_place(place_mask)) {
    h_mask.resize(batch_size);
    SyncCopyD2H(h_mask.data(), d_mask, batch_size, place_mask);
    add_mask = h_mask.data();
  }
  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    if (add_mask[i]) {
      add_unlock_data_with_float_label(add_pred[i], add_label[i]);
    }
  }
}

// add continue mask data
void BasicAucCalculator::add_continue_mask_data(
    const float* d_pred,
    const float* d_label,
    const int64_t* d_mask,
    int batch_size,
    const paddle::platform::Place& place_pred,
    const paddle::platform::Place& place_label,
    const paddle::platform::Place& place_mask) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<float> h_label;
  thread_local std::vector<int64_t> h_mask;
  const float* add_pred = d_pred;
  const float* add_label = d_label;
  const int64_t* add_mask = d_mask;
  if (platform::is_gpu_place(place_pred) || platform::is_xpu_place(place_pred)) {
    h_pred.resize(batch_size);
    SyncCopyD2H(h_pred.data(), d_pred, batch_size, place_pred);
    add_pred = h_pred.data();
  }
  if (platform::is_gpu_place(place_label) || platform::is_xpu_place(place_label)) {
    h_label.resize(batch_size);
    SyncCopyD2H(h_label.data(), d_label, batch_size, place_label);
    add_label = h_label.data();
  }
  if (platform::is_gpu_place(place_mask) || platform::is_xpu_place(place_mask)) {
    h_mask.resize(batch_size);
    SyncCopyD2H(h_mask.data(), d_mask, batch_size, place_mask);
    add_mask = h_mask.data();
  }
  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    if (add_mask[i]) {
      add_unlock_data_with_continue_label(add_pred[i], add_label[i]);
    }
  }
}

void BasicAucCalculator::init(int table_size, int max_batch_size) {
  set_table_size(table_size);
  set_max_batch_size(max_batch_size);
  // init CPU memory
  for (int i = 0; i < 2; i++) {
    _table[i] = std::vector<double>();
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
  _local_label = 0;
  _local_total_num = 0;
}

void BasicAucCalculator::compute() {
  int node_size = 1;
  double* table[2] = {&_table[0][0], &_table[1][0]};
#ifdef PADDLE_WITH_BOX_PS
  node_size = boxps::MPICluster::Ins().size();
  if (node_size > 1) {
    boxps::MPICluster::Ins().allreduce_sum(table[0], _table_size);
    boxps::MPICluster::Ins().allreduce_sum(table[1], _table_size);
  }
#elif defined(PADDLE_WITH_GLOO)
  auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  if (!gloo_wrapper->IsInitialized()) {
    VLOG(0) << "GLOO is not inited";
    gloo_wrapper->Init();
  }
  std::vector<double> neg_table;
  std::vector<double> pos_table;
  node_size = gloo_wrapper->Size();
  if (node_size > 1) {
    neg_table = gloo_wrapper->AllReduce(_table[0], "sum");
    pos_table = gloo_wrapper->AllReduce(_table[1], "sum");
    table[0] = &neg_table[0];
    table[1] = &pos_table[0];
  }
#endif

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

  if (node_size > 1) {
#ifdef PADDLE_WITH_BOX_PS
    // allreduce sum
    double local_err[3] = {_local_abserr, _local_sqrerr, _local_pred};
    boxps::MPICluster::Ins().allreduce_sum(local_err, 3);
#elif defined(PADDLE_WITH_GLOO)
    // allreduce sum
    std::vector<double> local_err_temp{_local_abserr, _local_sqrerr, _local_pred};
    auto local_err = gloo_wrapper->AllReduce(local_err_temp, "sum");
#else
    // allreduce sum
    double local_err[3] = {_local_abserr, _local_sqrerr, _local_pred};
#endif
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

  calculate_bucket_error(table[0], table[1]);
}

void BasicAucCalculator::calculate_bucket_error(
    const double *neg_table,
    const double *pos_table) {
  double last_ctr = -1;
  double impression_sum = 0;
  double ctr_sum = 0.0;
  double click_sum = 0.0;
  double error_sum = 0.0;
  double error_count = 0;
  const double* table[2] = {neg_table, pos_table};
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

void BasicAucCalculator::reset_records() {
  // reset wuauc_records_
  wuauc_records_.clear();
  _user_cnt = 0;
  _size = 0;
  _uauc = 0;
  _wuauc = 0;
}

// add uid data
void BasicAucCalculator::add_uid_data(const float* d_pred,
                                      const int64_t* d_label,
                                      const int64_t* d_uid,
                                      int batch_size,
                                      const paddle::platform::Place& place_pred,
                                      const paddle::platform::Place& place_label,
                                      const paddle::platform::Place& place_uid) {
  thread_local std::vector<float> h_pred;
  thread_local std::vector<int64_t> h_label;
  thread_local std::vector<int64_t> h_uid;
  const float* add_pred = d_pred;
  const int64_t* add_label = d_label;
  const int64_t* add_uid = d_uid;
  if (platform::is_gpu_place(place_pred) || platform::is_xpu_place(place_pred)) {
    h_pred.resize(batch_size);
    SyncCopyD2H(h_pred.data(), d_pred, batch_size, place_pred);
    add_pred = h_pred.data();
  }
  if (platform::is_gpu_place(place_label) || platform::is_xpu_place(place_label)) {
    h_label.resize(batch_size);
    SyncCopyD2H(h_label.data(), d_label, batch_size, place_label);
    add_label = h_label.data();
  }
  if (platform::is_gpu_place(place_uid) || platform::is_xpu_place(place_uid)) {
    h_uid.resize(batch_size);
    SyncCopyD2H(h_uid.data(), d_label, batch_size, place_uid);
    add_uid = h_uid.data();
  }
  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
      add_uid_unlock_data(add_pred[i], add_label[i],
          static_cast<uint64_t>(add_uid[i]));
  }
}

void BasicAucCalculator::add_nan_inf_data(const float* d_pred,
                                          const int64_t* d_label,
                                          int batch_size,
                                          const paddle::platform::Place& place_pred,
                                          const paddle::platform::Place& place_label){
  thread_local std::vector<float> h_pred;
  thread_local std::vector<int64_t> h_label;
  const float* add_pred = d_pred;
  const int64_t* add_label = d_label;
  if (platform::is_gpu_place(place_pred) || platform::is_xpu_place(place_pred)) {
    h_pred.resize(batch_size);
    SyncCopyD2H(h_pred.data(), d_pred, batch_size, place_pred);
    add_pred = h_pred.data();
  }
  if (platform::is_gpu_place(place_label) || platform::is_xpu_place(place_label)) {
    h_label.resize(batch_size);
    SyncCopyD2H(h_label.data(), d_label, batch_size, place_label);
    add_label = h_label.data();
  }
  std::lock_guard<std::mutex> lock(_table_mutex);
  for (int i = 0; i < batch_size; ++i) {
    add_nan_inf_unlock_data(add_pred[i], add_label[i]);
  }
}

void BasicAucCalculator::add_uid_unlock_data(double pred,
                                             int label,
                                             uint64_t uid) {
  PADDLE_ENFORCE_GE(
      pred,
      0.0,
      platform::errors::PreconditionNotMet("pred should be greater than 0"));
  PADDLE_ENFORCE_LE(
      pred,
      1.0,
      platform::errors::PreconditionNotMet("pred should be lower than 1"));
  PADDLE_ENFORCE_EQ(
      label * label,
      label,
      platform::errors::PreconditionNotMet(
          "label must be equal to 0 or 1, but its value is: %d", label));

  WuaucRecord record;
  record.uid_ = uid;
  record.label_ = label;
  record.pred_ = pred;
  wuauc_records_.emplace_back(std::move(record));
}

void BasicAucCalculator::computeWuAuc() {
  std::sort(wuauc_records_.begin(),
            wuauc_records_.end(),
            [](const WuaucRecord& lhs, const WuaucRecord& rhs) {
              if (lhs.uid_ == rhs.uid_) {
                if (lhs.pred_ == rhs.pred_) {
                  return lhs.label_ < rhs.label_;
                } else {
                  return lhs.pred_ > rhs.pred_;
                }
              } else {
                return lhs.uid_ > rhs.uid_;
              }
            });

  WuaucRocData roc_data;
  uint64_t prev_uid = 0;
  size_t prev_pos = 0;
  for (size_t i = 0; i < wuauc_records_.size(); ++i) {
    if (wuauc_records_[i].uid_ != prev_uid) {
      std::vector<WuaucRecord> single_user_recs(
          wuauc_records_.begin() + prev_pos, wuauc_records_.begin() + i);
      roc_data = computeSingelUserAuc(single_user_recs);
      if (roc_data.auc_ != -1) {
        double ins_num = (roc_data.tp_ + roc_data.fp_);
        _user_cnt += 1;
        _size += ins_num;
        _uauc += roc_data.auc_;
        _wuauc += roc_data.auc_ * ins_num;
      }

      prev_uid = wuauc_records_[i].uid_;
      prev_pos = i;
    }
  }

  std::vector<WuaucRecord> single_user_recs(wuauc_records_.begin() + prev_pos,
                                            wuauc_records_.end());
  roc_data = computeSingelUserAuc(single_user_recs);
  if (roc_data.auc_ != -1) {
    double ins_num = (roc_data.tp_ + roc_data.fp_);
    _user_cnt += 1;
    _size += ins_num;
    _uauc += roc_data.auc_;
    _wuauc += roc_data.auc_ * ins_num;
  }
}

BasicAucCalculator::WuaucRocData BasicAucCalculator::computeSingelUserAuc(
    const std::vector<WuaucRecord>& records) {
  double tp = 0.0;
  double fp = 0.0;
  double newtp = 0.0;
  double newfp = 0.0;
  double area = 0.0;
  double auc = -1;
  size_t i = 0;

  while (i < records.size()) {
    newtp = tp;
    newfp = fp;
    if (records[i].label_ == 1) {
      newtp += 1;
    } else {
      newfp += 1;
    }
    // check i+1
    while (i < records.size() - 1 && records[i].pred_ == records[i + 1].pred_) {
      if (records[i + 1].label_ == 1) {
        newtp += 1;
      } else {
        newfp += 1;
      }
      i += 1;
    }
    area += (newfp - fp) * (tp + newtp) / 2.0;
    tp = newtp;
    fp = newfp;
    i += 1;
  }
  if (tp > 0 && fp > 0) {
    auc = area / (fp * tp + 1e-9);
  } else {
    auc = -1;
  }
  return {tp, fp, auc};
}

void BasicAucCalculator::reset_nan_inf(){
  _nan_cnt = 0;
  _inf_cnt = 0;
  _nan_rate = 0;
  _inf_rate = 0;
  _nan_inf_rate = 0;
  _size = 0;
}

void BasicAucCalculator::computeContinueMsg() {
  int node_size = 1;
#ifdef PADDLE_WITH_BOX_PS
  node_size = boxps::MPICluster::Ins().size();
#elif defined(PADDLE_WITH_GLOO)
  auto gloo_wrapper = paddle::framework::GlooWrapper::GetInstance();
  if (!gloo_wrapper->IsInitialized()) {
    VLOG(0) << "GLOO is not inited";
    gloo_wrapper->Init();
  }
  std::vector<double> pair_table;
  node_size = gloo_wrapper->Size();
#endif
  if (node_size > 1) {
#ifdef PADDLE_WITH_BOX_PS
    // allreduce sum
    double local_err[5] = {_local_abserr,
                           _local_sqrerr,
                           _local_pred,
                           _local_label,
                           _local_total_num};
    boxps::MPICluster::Ins().allreduce_sum(local_err, 5);
#elif defined(PADDLE_WITH_GLOO)
    // allreduce sum
    std::vector<double> local_err_temp{_local_abserr,
                                       _local_sqrerr,
                                       _local_pred,
                                       _local_label,
                                       _local_total_num};
    auto local_err = gloo_wrapper->AllReduce(local_err_temp, "sum");
#else
    // allreduce sum
    double local_err[5] = {_local_abserr,
                           _local_sqrerr,
                           _local_pred,
                           _local_label,
                           _local_total_num};
#endif
    _mae = local_err[0] / _local_total_num;
    _rmse = sqrt(local_err[1] / _local_total_num);
    _predicted_value = local_err[2] / _local_total_num;
    _actual_value = local_err[3] / _local_total_num;
  } else {
    _mae = _local_abserr / _local_total_num;
    _rmse = sqrt(_local_sqrerr / _local_total_num);
    _predicted_value = _local_pred / _local_total_num;
    _actual_value = _local_label / _local_total_num;
  }

  _size = _local_total_num;
}

void BasicAucCalculator::computeNanInfMsg() {
  _nan_rate = _nan_cnt / _size;
  _inf_rate = _inf_cnt / _size;
  _nan_inf_rate = (_nan_cnt + _inf_cnt) / _size;
}

}  // namespace framework
}  // namespace paddle
#endif