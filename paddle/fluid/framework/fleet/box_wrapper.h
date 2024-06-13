/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <boxps_extends.h>
#include <boxps_public.h>
#include <dirent.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif
#include <glog/logging.h>

#include <algorithm>
#include <atomic>
#include <ctime>
#include <deque>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/metrics.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/string/string_helper.h"

#include "paddle/fluid/framework/fleet/metrics.h"
#include "paddle/fluid/framework/fleet/box_wrapper_kernel.h"

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

#define BUF_SIZE 1024 * 1024

DECLARE_bool(padbox_auc_runner_mode);
DECLARE_bool(enable_dense_nccl_barrier);
DECLARE_int32(padbox_dataset_shuffle_thread_num);

namespace paddle {
namespace framework {

extern int make_day_id(const int &y, const int &m, const int &d);

#ifdef PADDLE_WITH_BOX_PS
#define MAX_GPU_NUM 16

class GpuReplicaCache {
 public:
  explicit GpuReplicaCache(int dim) { emb_dim_ = dim; }

  ~GpuReplicaCache() {
    for (size_t i = 0; i < d_embs_.size(); ++i) {
#if defined(PADDLE_WITH_CUDA)
      cudaFree(d_embs_[i]);
#endif
    }
  }
  int AddItems(const std::vector<float>& emb) {
    int r;
    h_emb_mtx_.lock();
    h_emb_.insert(h_emb_.end(), emb.begin(), emb.end());
    r = h_emb_count_;
    ++h_emb_count_;
    h_emb_mtx_.unlock();
    return r;
  }

  void ToHBM() {
    int gpu_num = GetDeviceCount();
    for (int i = 0; i < gpu_num; ++i) {
      d_embs_.push_back(NULL);
#if defined(PADDLE_WITH_CUDA)
      cudaSetDevice(i);
      cudaMalloc(&d_embs_.back(), h_emb_count_ * emb_dim_ * sizeof(float));
      auto place = platform::CUDAPlace(i);
      auto stream = dynamic_cast<phi::GPUContext*>(
                        platform::DeviceContextPool::Instance().Get(place))
                        ->stream();
      cudaMemcpyAsync(d_embs_.back(),
                      h_emb_.data(),
                      h_emb_count_ * emb_dim_ * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream);
#else
      PADDLE_THROW(phi::errors::Unimplemented("not supported platform."));
#endif
    }
  }
#if defined(PADDLE_WITH_CUDA)
  void PullCacheValue(uint64_t* d_keys, float* d_vals, int num, int gpu_id);
#else
  void PullCacheValue(uint64_t* d_keys, float* d_vals, int num, int gpu_id) {
    PADDLE_THROW(phi::errors::Unimplemented("not supported platform."));
  }
#endif
  int emb_dim_ = 0;
  std::vector<float*> d_embs_;
  double GpuMemUsed(void) {
    return h_emb_count_ * emb_dim_ * sizeof(float) / 1024.0 / 1024.0;
  }

 private:
  int h_emb_count_ = 0;
  std::mutex h_emb_mtx_;
  std::vector<float> h_emb_;
};

class InputTable {
 public:
  explicit InputTable(uint64_t dim) : dim_(dim), miss_(0) {
    // add default vec 0 => [0, 0, ...]
    std::vector<float> vec(dim_, 0);
    AddIndexData("-", vec);
  }

  void AddIndexData(const std::string& key, const std::vector<float>& vec) {
    PADDLE_ENFORCE_EQ(vec.size(), dim_);

    table_mutex_.lock();
    key_offset_.emplace(key, table_.size());
    table_.insert(table_.end(), vec.begin(), vec.end());
    table_mutex_.unlock();
  }

  uint64_t GetIndexOffset(const std::string& key) {
    auto it = key_offset_.find(key);
    if (it == key_offset_.end()) {
      ++miss_;
      return 0;
    }

    return it->second;
  }

  void LookupInput(uint64_t* keys,
                   float* values,
                   uint64_t num,
                   size_t device_id) {
    std::vector<uint64_t> d_keys;
    std::vector<float> d_values;
    d_keys.resize(num);
    d_values.resize(num * dim_);
#if defined(PADDLE_WITH_CUDA)
    cudaSetDevice(device_id);
    cudaMemcpy(d_keys.data(),
               keys,
               d_keys.size() * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
#else
    PADDLE_THROW(phi::errors::Unimplemented("not supported platform."));
#endif
    for (size_t i = 0; i < num; ++i) {
      memcpy(&d_values[i * dim_], &table_[d_keys[i]], dim_ * sizeof(float));
    }
#if defined(PADDLE_WITH_CUDA)
    cudaMemcpy(values,
               d_values.data(),
               d_values.size() * sizeof(float),
               cudaMemcpyHostToDevice);
#else
    PADDLE_THROW(phi::errors::Unimplemented("not supported platform."));
#endif
  }

  size_t size() const { return key_offset_.size(); }

  size_t miss() const { return miss_; }

  size_t dim() const { return dim_; }

  double CpuMemUsed(void) {
    return (key_offset_.size() * dim_ * sizeof(float)) / 1024.0 / 1024.0;
  }

 protected:
  uint64_t dim_;
  std::mutex table_mutex_;
  std::unordered_map<std::string, uint64_t> key_offset_;
  std::vector<float> table_;
  std::atomic<size_t> miss_;
};
class DCacheBuffer {
 public:
  DCacheBuffer() : d_buf_(nullptr), total_bytes_(0), buf_(nullptr) {}
  ~DCacheBuffer() {
#ifdef PADDLE_WITH_CUDA
    if (d_buf_ != nullptr) {
      cudaFree(d_buf_);
      d_buf_ = nullptr;
    }
#endif
  }
  /**
   * @Brief get data
   */
  template <typename T>
  T* mutable_data(const size_t total_bytes,
                  const paddle::platform::Place& place) {
#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(place)) {
      if (d_buf_ == nullptr) {
        total_bytes_ = total_bytes;
        cudaMalloc(&d_buf_, total_bytes);
      } else if (total_bytes_ < total_bytes) {
        total_bytes_ = total_bytes;
        cudaFree(d_buf_);
        cudaMalloc(&d_buf_, total_bytes);
      }
      return reinterpret_cast<T*>(d_buf_);
    } else {
#endif
      if (buf_ == nullptr) {
        buf_ = memory::AllocShared(place, total_bytes);
      } else if (buf_->size() < total_bytes) {
        buf_.reset();
        buf_ = memory::AllocShared(place, total_bytes);
      }
      return reinterpret_cast<T*>(buf_->ptr());
#ifdef PADDLE_WITH_CUDA
    }
#endif
  }
  template <typename T>
  T* data() {
#ifdef PADDLE_WITH_CUDA
    if (d_buf_ != nullptr) {
      return reinterpret_cast<T*>(d_buf_);
    }
#endif
    return reinterpret_cast<T*>(buf_->ptr());
  }
  size_t memory_size() {
    if (buf_ == nullptr && d_buf_ == nullptr) {
      return 0;
    }
#ifdef PADDLE_WITH_CUDA
    if (d_buf_ != nullptr) {
      return total_bytes_;
    }
#endif
    return buf_->size();
  }

  bool IsInitialized() {
#ifdef PADDLE_WITH_CUDA
    return d_buf_ != nullptr;
#endif
    return buf_ != nullptr;
  }
 private:
  void* d_buf_ = nullptr;
  size_t total_bytes_ = 0;
  std::shared_ptr<memory::Allocation> buf_ = nullptr;
};
class MetricMsg {
 public:
  MetricMsg() {}
  MetricMsg(const std::string& label_varname,
            const std::string& pred_varname,
            int metric_phase,
            int bucket_size = 1000000,
            bool mode_collect_in_gpu = false,
            int max_batch_size = 0,
            const std::string& sample_scale_varname = "")
      : label_varname_(label_varname),
        pred_varname_(pred_varname),
        sample_scale_varname_(sample_scale_varname),
        metric_phase_(metric_phase) {
    calculator = new BasicAucCalculator(mode_collect_in_gpu);
    calculator->init(bucket_size, max_batch_size);
  }
  virtual ~MetricMsg() {}

  int MetricPhase() const { return metric_phase_; }
  BasicAucCalculator* GetCalculator() { return calculator; }
  inline phi::Place GetVarPlace(const paddle::framework::Scope *exe_scope, const std::string &varname) {
    auto* var = exe_scope->FindVar(varname.c_str());
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("Error: var %s is not found in scope.",
                                        varname.c_str()));
    auto& gpu_tensor = var->Get<LoDTensor>();
    auto place = gpu_tensor.place();
    return place;
  }
  virtual void add_data(const Scope* exe_scope,
                        const paddle::platform::Place& place) {
    int label_len = 0;
    const int64_t* label_data = NULL;
    int pred_len = 0;
    const float* pred_data = NULL;
    get_data<int64_t>(exe_scope, label_varname_, &label_data, &label_len);
    get_data<float>(exe_scope, pred_varname_, &pred_data, &pred_len);
    PADDLE_ENFORCE_EQ(label_len,
                      pred_len,
                      platform::errors::PreconditionNotMet(
                          "the predict data length should be consistent with "
                          "the label data length"));
    auto pre_var_place = GetVarPlace(exe_scope, pred_varname_);
    auto label_var_place = GetVarPlace(exe_scope, label_varname_);
    std::vector<float> sample_scale_data;
    if (!sample_scale_varname_.empty()) {
      get_data<float>(exe_scope, sample_scale_varname_, &sample_scale_data);
      PADDLE_ENFORCE_EQ(
          label_len,
          sample_scale_data.size(),
          platform::errors::PreconditionNotMet(
              "lable size [%lu] and sample_scale_data[%lu] should be same",
              label_len,
              sample_scale_data.size()));
      calculator->add_sample_data(
          pred_data, label_data, sample_scale_data, label_len, pre_var_place, label_var_place);
    } else {
      calculator->add_data(pred_data, label_data, label_len, pre_var_place, label_var_place);
    }
  }
  template <class T = float>
  static void get_data(const Scope* exe_scope,
                       const std::string& varname,
                       const T** data,
                       int* len) {
    auto* var = exe_scope->FindVar(varname.c_str());
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::NotFound("Error: var %s is not found in scope.",
                                   varname.c_str()));
    auto& gpu_tensor = var->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(
        gpu_tensor.IsInitialized(),
        true,
        platform::errors::InvalidArgument(
            "Error: monitor var `%s` uninitialized Tensor.", varname.c_str()));
    *data = gpu_tensor.data<T>();
    *len = gpu_tensor.numel();
  }
  template <class T = float>
  static void get_data(const Scope* exe_scope,
                       const std::string& varname,
                       std::vector<T>* data) {
    auto* var = exe_scope->FindVar(varname.c_str());
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::NotFound("Error: var %s is not found in scope.",
                                   varname.c_str()));
    auto& gpu_tensor = var->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(
        gpu_tensor.IsInitialized(),
        true,
        platform::errors::InvalidArgument(
            "Error: monitor var `%s` uninitialized Tensor.", varname.c_str()));
    auto* gpu_data = gpu_tensor.data<T>();
    auto len = gpu_tensor.numel();
    data->resize(len);
    SyncCopyD2H(data->data(), gpu_data, len, gpu_tensor.place());
  }
  static inline std::pair<int, int> parse_cmatch_rank(uint64_t x) {
    // first 32 bit store cmatch and second 32 bit store rank
    return std::make_pair(static_cast<int>(x >> 32),
                          static_cast<int>(x & 0xff));
  }

 protected:
  std::string label_varname_;
  std::string pred_varname_;
  std::string sample_scale_varname_;
  int metric_phase_;
  BasicAucCalculator* calculator;
};
class BoxWrapper {
  struct DeviceBoxData {
    DCacheBuffer keys_tensor;
    DCacheBuffer dims_tensor;
    DCacheBuffer pull_push_tensor;
    DCacheBuffer keys_ptr_tensor;
    DCacheBuffer values_ptr_tensor;

    DCacheBuffer slot_lens;
    DCacheBuffer d_slot_vector;
    DCacheBuffer keys2slot;

    DCacheBuffer pull_offset;
    DCacheBuffer push_offset;

    LoDTensor qvalue;

    platform::Timer all_pull_timer;
    platform::Timer boxps_pull_timer;
    platform::Timer all_push_timer;
    platform::Timer boxps_push_timer;
    platform::Timer dense_nccl_timer;
    platform::Timer dense_sync_timer;
    platform::Timer pull_dedup_timer;
    platform::Timer copy_keys_timer;
    platform::Timer copy_values_timer;
    platform::Timer copy_push_timer;

    int64_t total_key_length = 0;
    int64_t dedup_key_length = 0;

    void ResetTimer(void) {
      all_pull_timer.Reset();
      boxps_pull_timer.Reset();
      all_push_timer.Reset();
      boxps_push_timer.Reset();
      dense_nccl_timer.Reset();
      dense_sync_timer.Reset();
      pull_dedup_timer.Reset();
      copy_keys_timer.Reset();
      copy_values_timer.Reset();
      copy_push_timer.Reset();
    }
    double GpuMemUsed(void) {
      size_t total = 0;
      total += keys_tensor.memory_size();
      total += dims_tensor.memory_size();
      total += pull_push_tensor.memory_size();
      total += keys_ptr_tensor.memory_size();
      total += values_ptr_tensor.memory_size();
      total += slot_lens.memory_size();
      total += d_slot_vector.memory_size();
      total += keys2slot.memory_size();
      total += qvalue.memory_size();
      total += pull_offset.memory_size();
      total += push_offset.memory_size();
      return total / 1024.0 / 1024.0;
    }
  };

 public:
  std::deque<GpuReplicaCache> gpu_replica_cache;
  std::deque<InputTable> input_table_deque_;

  virtual ~BoxWrapper() {}
  BoxWrapper() {
    fprintf(stdout, "init box wrapper\n");
    boxps::MPICluster::Ins();
#ifdef PADDLE_WITH_XPU_KP
    box_wrapper_kernel_ = std::make_unique<BoxWrapperKernel>();
    use_xpu_sparse_map_ = false;
    auto env_str = std::getenv("USE_XPU_SPARSE_MAP");
    if (env_str != nullptr && (strcmp(env_str, "true") == 0 || strcmp(env_str, "1") == 0)) {
      use_xpu_sparse_map_ = true;
    }

    check_xpu_nan_ = false;
    env_str = std::getenv("CHECK_XPU_BOXPS_NAN");
    if (env_str != nullptr && (strcmp(env_str, "true") == 0 || strcmp(env_str, "1") == 0)) {
      check_xpu_nan_ = true;
      VLOG(0) << "CHECK_XPU_BOXPS_NAN has been set to check paddle pull&push";
    }
    check_xpu_continuous_memory_ = true;
    env_str = std::getenv("CHECK_XPU_CONTINUOUS_MEMORY");
    if (env_str != nullptr && (strcmp(env_str, "false") == 0 || strcmp(env_str, "0") == 0)) {
      check_xpu_continuous_memory_ = false;
      VLOG(0) << "CHECK_XPU_CONTINUOUS_MEMORY has been unset to check paddle pull&push";
    }

#endif
#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
    // Client side to produce the tracepoints.
    factory = std::make_shared<scalopus::TransportLoopbackFactory>();
    const auto server = factory->serve();
    server->addEndpoint(std::make_shared<scalopus::EndpointTraceMapping>());
    server->addEndpoint(std::make_shared<scalopus::EndpointIntrospect>());
    server->addEndpoint(std::make_shared<scalopus::EndpointNativeTraceSender>());
    auto endpoint_process_info = std::make_shared<scalopus::EndpointProcessInfo>();
    endpoint_process_info->setProcessName("BoxWrapper");
    server->addEndpoint(endpoint_process_info);

    // Catapult recorder side.
    manager = std::make_shared<scalopus::EndpointManagerPoll>(factory);
    manager->addEndpointFactory<scalopus::EndpointTraceMapping>();
    manager->addEndpointFactory<scalopus::EndpointProcessInfo>();
    auto native_trace_provider = std::make_shared<scalopus::NativeTraceProvider>(manager);
    manager->addEndpointFactory(scalopus::EndpointNativeTraceSender::name, native_trace_provider);

    catapult_recorder = std::make_shared<scalopus::CatapultRecorder>();
    catapult_recorder->addProvider(native_trace_provider);
    catapult_recorder->addProvider(std::make_shared<scalopus::GeneralProvider>(manager));

    auto logging_function = [](const std::string& msg) { std::cout << msg << std::endl; };
    logging_function("Logging can be enabled in the source, uncomment the lines below this one.");

    manager->connect(server->getAddress());  // Connect the manager to the server.
    catapult_recorder->start();              // start the recorder thread.
    catapult_recorder->startInterval();      // start recording.

    TRACE_THREAD_NAME("BoxWrapper");
    std::cout<<"start profile in BoxWrapper"<<std::endl;
#endif
  }
  int GetMpiSize() { return boxps::MPICluster::Ins().size(); }
  int GetMpiRank() { return boxps::MPICluster::Ins().rank(); }
  int GetNCCLRankId(const int& device_id) {
    return (GetMpiRank() * gpu_num_ + device_id);
  }
  int GetGpuNum() { return gpu_num_; }
  void SetDatasetName(const std::string& name) {}
  void SetInputTableDim(size_t dim) { input_table_dim_ = dim; }
  void FeedPass(int date, const std::vector<uint64_t>& feasgin_to_box);
  void BeginFeedPass(int date, boxps::PSAgentBase** agent);
  void EndFeedPass(boxps::PSAgentBase* agent);
  void BeginPass();
  void EndPass(bool need_save_delta);
  void SetTestMode(bool is_test) const;

  void PullSparseCase(const paddle::platform::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<float*>& values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size,
                      const int expand_embed_dim,
                      const int skip_offset,
                      bool expand_only);

  void PullSparseCaseGPU(const paddle::platform::Place& place,
                         const std::vector<const uint64_t*>& keys,
                         const std::vector<float*>& values,
                         const std::vector<int64_t>& slot_lengths,
                         const int hidden_size,
                         const int expand_embed_dim,
                         const int skip_offset,
                         bool expand_only);

  void PullSparseCaseXPU(const paddle::platform::Place& place,
                         const std::vector<const uint64_t*>& keys,
                         const std::vector<float*>& values,
                         const std::vector<int64_t>& slot_lengths,
                         const int hidden_size,
                         const int expand_embed_dim,
                         const int skip_offset,
                         bool expand_only);

  void PullSparseCaseCPU(const paddle::platform::Place& place,
                         const std::vector<const uint64_t*>& keys,
                         const std::vector<float*>& values,
                         const std::vector<int64_t>& slot_lengths,
                         const int hidden_size,
                         const int expand_embed_dim,
                         const int skip_offset,
                         bool expand_only);

  void PullSparse(const paddle::platform::Place& place,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size,
                  const int expand_embed_dim,
                  const int skip_offset,
                  bool expand_only);

  void PushSparseGradCase(const paddle::platform::Place& place,
                          const std::vector<const uint64_t*>& keys,
                          const std::vector<const float*>& grad_values,
                          const std::vector<int64_t>& slot_lengths,
                          const int hidden_size,
                          const int expand_embed_dim,
                          const int batch_size,
                          const int skip_offset,
                          bool expand_only);

  void PushSparseGradCaseGPU(const paddle::platform::Place& place,
                             const std::vector<const uint64_t*>& keys,
                             const std::vector<const float*>& grad_values,
                             const std::vector<int64_t>& slot_lengths,
                             const int hidden_size,
                             const int expand_embed_dim,
                             const int batch_size,
                             const int skip_offset,
                             bool expand_only);

  void PushSparseGradCaseXPU(const paddle::platform::Place& place,
                             const std::vector<const uint64_t*>& keys,
                             const std::vector<const float*>& grad_values,
                             const std::vector<int64_t>& slot_lengths,
                             const int hidden_size,
                             const int expand_embed_dim,
                             const int batch_size,
                             const int skip_offset,
                             bool expand_only);

  void PushSparseGradCaseCPU(const paddle::platform::Place& place,
                             const std::vector<const uint64_t*>& keys,
                             const std::vector<const float*>& grad_values,
                             const std::vector<int64_t>& slot_lengths,
                             const int hidden_size,
                             const int expand_embed_dim,
                             const int batch_size,
                             const int skip_offset,
                             bool expand_only);

  void PushSparseGrad(const paddle::platform::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size,
                      const int expand_embed_dim,
                      const int batch_size,
                      const int skip_offset,
                      bool expand_only);

  void CopyForPull(const paddle::platform::Place& place,
                   uint64_t** gpu_keys,
                   float** gpu_values,
                   void* total_values_gpu,
                   boxps::FeaturePullOffset* pull_offset,
                   const int64_t* slot_lens,
                   const int slot_num,
                   const int* key2slot,
                   const int hidden_size,
                   const int expand_embed_dim,
                   const int64_t total_length,
                   int* total_dims,
                   const int skip_offset,
                   bool expand_only,
                   const uint32_t* gpu_restore_idx = nullptr);

  void CopyForPullCPU(const paddle::platform::Place& place,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<float*>& values,
                      void* total_values_gpu,
                      const int64_t* slot_lens,
                      const int slot_num,
                      const int* key2slot,
                      const int hidden_size,
                      const int expand_embed_dim,
                      const int64_t total_length,
                      int* total_dims,
                      const int skip_offset,
                      bool expand_only,
                      const uint32_t* gpu_restore_idx = nullptr);

  void CopyForPush(const paddle::platform::Place& place,
                   float** grad_values,
                   void* total_grad_values_gpu,
                   boxps::FeaturePushOffset* push_offset,
                   const int64_t total_length,
                   const int64_t dedup_length,
                   const int* slots,
                   const int64_t* slot_lens,
                   const int slot_num,
                   const int hidden_size,
                   const int expand_embed_dim,
                   const int batch_size,
                   const int* total_dims,
                   const int* key2slot,
                   const int skip_offset,
                   bool expand_only,
                   const uint32_t* gpu_sort_idx = nullptr,
                   const uint32_t* gpu_sort_offset = nullptr,
                   const uint32_t* gpu_sort_lens = nullptr,
                   const uint32_t* gpu_restore_idx = nullptr);

  void CopyForPushCPU(const paddle::platform::Place& place,
                      const std::vector<const float*>& grad_values,
                      void* total_grad_values_gpu,
                      const int* slots,
                      const int64_t* slot_lens,
                      const int slot_num,
                      const int hidden_size,
                      const int expand_embed_dim,
                      const int64_t total_length,
                      const int batch_size,
                      const int* total_dims,
                      const int* key2slot,
                      const int skip_offset,
                      bool expand_only,
                      const uint32_t* gpu_sort_idx = nullptr,
                      const uint32_t* gpu_sort_offset = nullptr,
                      const uint32_t* gpu_sort_lens = nullptr);

  void CopyKeys(const paddle::platform::Place& place,
                uint64_t** origin_keys,
                uint64_t* total_keys,
                const int64_t* gpu_len,
                int slot_num,
                int total_len,
                int* key2slot);
  // copy cpu keys
  void CopyCPUKeys(const paddle::platform::Place& place,
                   const std::vector<const uint64_t*>& keys,
                   uint64_t* total_keys,
                   const int64_t* slot_lengths_lod,
                   int slot_num,
                   int total_len,
                   int* key2slot);

#ifdef PADDLE_WITH_XPU_KP
  void SetDataFuncForCacheManager(int batch_num,
      std::function<void(int, std::vector<std::vector<std::pair<uint64_t*, int>>>*)> data_func);
  int PrepareNextBatch(int dev_id);
  std::vector<uint64_t> * GetFid2SginMap() { return fid2sign_map_; }
#endif

  boxps::PSAgentBase* GetAgent();
  void RelaseAgent(boxps::PSAgentBase* agent);
  void InitializeGPUAndLoadModel(
      const char* conf_file,
      const std::vector<int>& slot_vector,
      const std::vector<std::string>& slot_omit_in_feedpass,
      const std::string& model_path,
      const std::map<std::string, float>& lr_map);
  int GetFeedpassThreadNum() const { return feedpass_thread_num_; }
  void Finalize();
  void ReleasePool(void);
  const std::string SaveBase(const char* batch_model_path,
                             const char* xbox_model_path,
                             const std::string& date);
  const std::string SaveDelta(const char* xbox_model_path);
  // mem table shrink
  bool ShrinkTable() { return boxps_ptr_->ShrinkTable(); }
  // load ssd2mem
  bool LoadSSD2Mem(const std::string& date);

  static std::shared_ptr<BoxWrapper> GetInstance() {
    PADDLE_ENFORCE_EQ(
        s_instance_ == nullptr,
        false,
        platform::errors::PreconditionNotMet(
            "GetInstance failed in BoxPs, you should use SetInstance firstly"));
    return s_instance_;
  }

  static std::shared_ptr<BoxWrapper> SetInstance(
      int embedx_dim = 8,
      int expand_embed_dim = 0,
      int feature_type = 0,
      float pull_embedx_scale = 1.0) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (nullptr == s_instance_) {
      VLOG(3) << "s_instance_ is null";
      s_instance_.reset(new paddle::framework::BoxWrapper());
      s_instance_->boxps_ptr_.reset(boxps::BoxPSBase::GetInsEx(
          embedx_dim, expand_embed_dim, feature_type));
      s_instance_->embedx_dim_ = embedx_dim;
      s_instance_->expand_embed_dim_ = expand_embed_dim;
      s_instance_->feature_type_ = feature_type;
      s_instance_->pull_embedx_scale_ = pull_embedx_scale;
      s_instance_->gpu_num_ = GetDeviceCount();
      // get feature offset info
      s_instance_->GetFeatureOffsetInfo();

      if (boxps::MPICluster::Ins().size() > 1) {
        data_shuffle_.reset(boxps::PaddleShuffler::New());
        data_shuffle_->init(FLAGS_padbox_dataset_shuffle_thread_num);
      }
    } else {
      if (nullptr == s_instance_->boxps_ptr_) {
        VLOG(0) << "reset boxps ptr";
        s_instance_->boxps_ptr_.reset(boxps::BoxPSBase::GetInsEx(
            embedx_dim, expand_embed_dim, feature_type));
      }
      LOG(WARNING) << "You have already used SetInstance() before";
    }
    return s_instance_;
  }

  bool SyncDense(boxps::StreamType stream,
                 const int size,
                 const void* sendbuf,
                 void* recvbuf,
                 const int deviceid = 0,
                 bool allgather = false) {
    return boxps_ptr_->SyncDense(stream,
                                 size,
                                 reinterpret_cast<const char*>(sendbuf),
                                 reinterpret_cast<char*>(recvbuf),
                                 deviceid,
                                 allgather);
  }

  void DenseNcclTimer(const int deviceid, bool pause, int flag = 1) {
    auto& dev = device_caches_[deviceid];
    if (flag & 0x01) {
      if (pause) {
        dev.dense_nccl_timer.Pause();
      } else {
        dev.dense_nccl_timer.Resume();
      }
    }
    if (flag & 0x02) {
      if (pause) {
        if (FLAGS_enable_dense_nccl_barrier) {
          boxps::MPICluster::Ins().barrier();
        }
        dev.dense_sync_timer.Pause();
      } else {
        dev.dense_sync_timer.Resume();
      }
    }
  }

  void InitAfsAPI(const std::string& fs_name,
                  const std::string& fs_ugi,
                  const std::string& conf_path) {
    file_manager_.reset(boxps::PaddleFileMgr::New());
    auto split = fs_ugi.find(",");
    std::string user = fs_ugi.substr(0, split);
    std::string pwd = fs_ugi.substr(split + 1);
    bool ret = file_manager_->initialize(fs_name, user, pwd, conf_path);
    PADDLE_ENFORCE_EQ(ret,
                      true,
                      platform::errors::PreconditionNotMet(
                          "Called AFSAPI Init Interface Failed."));
    use_afs_api_ = true;
  }

  bool UseAfsApi() const { return use_afs_api_; }

  std::shared_ptr<FILE> OpenReadFile(const std::string& path,
                                     const std::string& pipe_command) {
    return boxps::fopen_read(file_manager_.get(), path, pipe_command);
  }

  boxps::PaddleFileMgr* GetFileMgr(void) { return file_manager_.get(); }
  // get dataset id
  uint16_t GetDataSetId(void) { return dataset_id_.fetch_add(1); }
  uint16_t GetRoundId(void) { return round_id_.fetch_add(1); }

  // this performs better than rand_r, especially large data
  static std::default_random_engine& LocalRandomEngine() {
    struct engine_wrapper_t {
      std::default_random_engine engine;
      engine_wrapper_t() {
        struct timespec tp;
        clock_gettime(CLOCK_REALTIME, &tp);
        double cur_time = tp.tv_sec + tp.tv_nsec * 1e-9;
        static std::atomic<uint64_t> x(0);
        std::seed_seq sseq = {x++, x++, x++, (uint64_t)(cur_time * 1000)};
        engine.seed(sseq);
      }
    };
    thread_local engine_wrapper_t r;
    return r.engine;
  }

  const std::unordered_set<std::string>& GetOmitedSlot() const {
    return slot_name_omited_in_feedpass_;
  }

  const std::vector<std::string> GetMetricNameList(int metric_phase = -1) const;
  int Phase() const { return phase_; }
  int PhaseNum() const { return phase_num_; }
  void FlipPhase() { phase_ = (phase_ + 1) % phase_num_; }
  void SetPhase(int phase) { phase_ = phase; }
  const std::map<std::string, float> GetLRMap() const { return lr_map_; }
  std::map<std::string, MetricMsg*>& GetMetricList() { return metric_lists_; }

  void InitMetric(const std::string& method,
                  const std::string& name,
                  const std::string& label_varname,
                  const std::string& pred_varname,
                  const std::string& cmatch_rank_varname,
                  const std::string& mask_varname,
                  int metric_phase,
                  const std::string& cmatch_rank_group,
                  bool ignore_rank,
                  int bucket_size = 1000000,
                  bool mode_collect_in_gpu = false,
                  int max_batch_size = 0,
                  const std::string& sample_scale_varname = "");
  const std::vector<double> GetMetricMsg(const std::string& name);
  const std::vector<double> GetContinueMetricMsg(const std::string& name);
  const std::vector<double> GetNanInfMetricMsg(const std::string& name);
  // pcoc qvalue tensor
  LoDTensor& GetQTensor(int device) { return device_caches_[device].qvalue; }
  void PrintSyncTimer(int device, double train_span);
  // get expand embed dim
  int GetExpandEmbedDim(void) { return expand_embed_dim_; }
  // shrink boxps resource
  void ShrinkResource(void) { return boxps_ptr_->ShrinkResource(); }
  // merge model interface
  int MergeModel(const std::string& path) {
    return boxps_ptr_->MergeModel(path);
  }
  void PrintDeviceInfo(double span) {
    for (int i = 0; i < gpu_num_; ++i) {
      PrintSyncTimer(i, span);
    }
    boxps_ptr_->CheckNeedLimitMem();
  }
  // merge multi models interface
  int MergeMultiModels(const std::string& path,
                       const std::string& update_type,
                       const int& model_index) {
    return boxps_ptr_->MergeMultiModels(path, update_type, model_index);
  }
  // get device id
  int GetPlaceDeviceId(const paddle::platform::Place& place) {
    if (platform::is_gpu_place(place)) {
      return place.GetDeviceId();
    }
    if (platform::is_xpu_place(place)) {
      return place.GetDeviceId();
    }
    thread_local int device_id = -1;
    static std::atomic<int> dev_id{0};
    if (device_id < 0) {
      device_id = dev_id.fetch_add(1);
    }
    return device_id;
  }
  // get feature offset info
  void GetFeatureOffsetInfo(void);
  // execute func
  void ExecuteFunc(const paddle::platform::Place& place,
                   const size_t& num,
                   std::function<void(const size_t&)> func) {
    boxps_ptr_->ExecuteFunc(GetPlaceDeviceId(place), num, func);
  }
  // execute func
  void ExecRangeFunc(const paddle::platform::Place& place,
                     const size_t& num,
                     std::function<void(const size_t&, const size_t&)> func) {
    boxps_ptr_->ExecRangeFunc(GetPlaceDeviceId(place), num, func);
  }
  // get slot vector
  const std::vector<int>& GetSlotVector(void) { return slot_vector_; }
  // add skip gc var
  void AddSkipGCVar(const std::string& str) {
    if (str.empty()) {
      return;
    }
    auto var_names = string::split_string(str);
    for (auto& name : var_names) {
      auto it = std::find(skip_gc_vars_.begin(), skip_gc_vars_.end(), name);
      if (it != skip_gc_vars_.end()) {
        // return;
        continue;
      }
      skip_gc_vars_.push_back(name);
    }
  }
  // get need skip gc var
  const std::vector<std::string>& GetSkipGCVars(void) { return skip_gc_vars_; }

 private:
  static boxps::StreamType stream_list_[MAX_GPU_NUM];
  static std::shared_ptr<BoxWrapper> s_instance_;
  std::shared_ptr<boxps::BoxPSBase> boxps_ptr_ = nullptr;

 private:
  std::mutex mutex_;
  std::deque<boxps::PSAgentBase*> psagents_;
  // TODO(hutuxian): magic number, will add a config to specify
  const int feedpass_thread_num_ = 30;  // magic number
  std::unordered_set<std::string> slot_name_omited_in_feedpass_;
  // EMBEDX_DIM and EXPAND_EMBED_DIM
  int embedx_dim_ = 8;
  int expand_embed_dim_ = 0;
  int feature_type_ = 0;
  float pull_embedx_scale_ = 1.0;
  int cvm_offset_ = 3;
  // Need to refactor wrapper.cu
  size_t feature_pull_size_ = 0;
  size_t feature_push_size_ = 0;
  boxps::FeaturePullOffset pull_info_;
  boxps::FeaturePushOffset push_info_;
  size_t pull_float_num_ = 0;
  size_t push_float_num_ = 0;

  // Metric Related
  int phase_ = 1;
  int phase_num_ = 2;
  std::map<std::string, MetricMsg*> metric_lists_;
  std::vector<std::string> metric_name_list_;
  std::vector<int> slot_vector_;
  bool use_afs_api_ = false;
  std::shared_ptr<boxps::PaddleFileMgr> file_manager_ = nullptr;
  // box device cache
  DeviceBoxData* device_caches_ = nullptr;
  std::map<std::string, float> lr_map_;
  size_t input_table_dim_ = 0;
  int gpu_num_ = GetDeviceCount();
#ifdef PADDLE_WITH_XPU_KP
  bool check_xpu_nan_;
  bool check_xpu_continuous_memory_;
  bool use_xpu_sparse_map_;
  std::vector<uint64_t> * fid2sign_map_ = nullptr;
  std::unique_ptr<BoxWrapperKernel> box_wrapper_kernel_;
#endif

 public:
  static std::shared_ptr<boxps::PaddleShuffler> data_shuffle_;

  // Auc Runner
 public:
  void InitializeAucRunner(std::vector<std::vector<std::string>> slot_eval,
                           int thread_num,
                           int pool_size,
                           std::vector<std::string> slot_list) {
    //    PADDLE_ENFORCE_EQ(FLAGS_padbox_auc_runner_mode, true,
    //                      platform::errors::InvalidArgument(
    //                          "you should export
    //                          FLAGS_padbox_auc_runner_mode=true "
    //                          "in auc runner mode."));
#ifdef PADDLE_WITH_XPU_KP
    setenv("ENABLE_ALWAYS_SIGN2FID", "true", 1);
#endif
    size_t object_bytes = sizeof(SlotRecordObject) +
                          sizeof(float) * FLAGS_padbox_slotrecord_extend_dim +
                          sizeof(AucRunnerInfo);
    SlotRecordPool().set_slotrecord_size(object_bytes);

    mode_ = 1;
    phase_num_ = static_cast<int>(slot_eval.size());
    phase_ = phase_num_ - 1;
    auc_runner_thread_num_ = thread_num;
    pass_done_semi_ = paddle::framework::MakeChannel<int>();
    random_ins_pool_list.resize(thread_num);
    for (size_t i = 0; i < random_ins_pool_list.size(); ++i) {
      random_ins_pool_list[i].Resize(pool_size);
    }

    slot_eval_set_.clear();
    for (size_t i = 0; i < slot_eval.size(); ++i) {
      for (const auto& slot : slot_eval[i]) {
        slot_eval_set_.insert(slot);
      }
    }

    VLOG(0) << "AucRunner configuration: thread number[" << thread_num
            << "], pool size[" << pool_size << "], runner_group[" << phase_num_
            << "], eval size:[" << slot_eval_set_.size() << "]"
            << ", object size:[" << object_bytes << "]";
    //    VLOG(0) << "Slots that need to be evaluated:";
    //    for (auto e : slot_index_to_replace_) {
    //      VLOG(0) << e << ": " << slot_list[e];
    //    }
  }
  void GetRandomReplace(std::vector<SlotRecord>* records);
  void PostUpdate();
  void AddReplaceFeasign(boxps::PSAgentBase* p_agent, int feed_pass_thread_num);
  void GetRandomData(const std::vector<Record>& pass_data,
                     const std::unordered_set<uint16_t>& slots_to_replace,
                     std::vector<Record>* result);
  int Mode() const { return mode_; }

  void PushAucRunnerResource(size_t records_len) {
    platform::Timer timer;
    timer.Start();

    for (auto& pool : random_ins_pool_list) {
      pool.Push();
    }

    timer.Pause();
    VLOG(0) << "PushAucRunnerResource cost: " << timer.ElapsedMS();
  }

  void PopAucRunnerResource() {
    platform::Timer timer;
    timer.Start();

    std::lock_guard<std::mutex> lock(mutex4random_pool_);
    for (auto& pool : random_ins_pool_list) {
      pool.Pop();
    }
    record_replacers_.clear();
    last_slots_idx_.clear();

    timer.Pause();
    VLOG(0) << "PopAucRunnerResource cost: " << timer.ElapsedMS();
  }

  std::vector<FeasignValuesReplacer> record_replacers_;
  std::set<uint16_t> last_slots_idx_;

  void RecordReplace(std::vector<SlotRecord>* records,
                     const std::set<uint16_t>& slots);
  void RecordReplaceBack(std::vector<SlotRecord>* records,
                         const std::set<uint16_t>& slots);

  // aucrunner
  void SetReplacedSlots(const std::set<uint16_t>& slot_index_to_replace) {
    for (int i = 0; i < auc_runner_thread_num_; ++i) {
      random_ins_pool_list[i].SetReplacedSlots(slot_index_to_replace);
    }
  }
  const std::set<std::string>& GetEvalSlotSet() { return slot_eval_set_; }

 private:
  int mode_ = 0;  // 0 means train/test 1 means auc_runner
  int auc_runner_thread_num_ = 1;
  bool init_done_ = false;
  paddle::framework::Channel<int> pass_done_semi_;

  std::vector<FeasignValuesCandidateList> random_ins_pool_list;
  std::mutex mutex4random_pool_;
  std::set<std::string> slot_eval_set_;
  std::atomic<uint16_t> dataset_id_{0};
  std::atomic<uint16_t> round_id_{0};

#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
  scalopus::TransportLoopbackFactory::Ptr factory;
  std::shared_ptr<scalopus::EndpointManagerPoll> manager;
  scalopus::CatapultRecorder::Ptr catapult_recorder;
#endif

  // skip gc vars
  std::vector<std::string> skip_gc_vars_;
};
/**
 * @brief file mgr
 */
class BoxFileMgr {
 public:
  BoxFileMgr();
  ~BoxFileMgr();
  bool init(const std::string& fs_name,
            const std::string& fs_ugi,
            const std::string& conf_path);
  void destory(void);
  std::vector<std::string> list_dir(const std::string& path);
  bool makedir(const std::string& path);
  bool exists(const std::string& path);
  bool down(const std::string& remote, const std::string& local);
  bool upload(const std::string& local, const std::string& remote);
  bool remove(const std::string& path);
  int64_t file_size(const std::string& path);
  std::vector<std::pair<std::string, int64_t>> dus(const std::string& path);
  bool truncate(const std::string& path, const size_t len);
  bool touch(const std::string& path);
  bool rename(const std::string& src, const std::string& dest);
  std::vector<std::pair<std::string, int64_t>> list_info(
      const std::string& path);
  int64_t count(const std::string& path);

 private:
  std::shared_ptr<boxps::PaddleFileMgr> mgr_ = nullptr;
};
#endif
class BoxHelper {
 public:
  explicit BoxHelper(paddle::framework::Dataset* dataset) : dataset_(dataset) {}
  virtual ~BoxHelper() {}

  void SetDate(int year, int month, int day) {
    day_id_ = make_day_id(year, month, day);
    VLOG(0) << "BoxHelpler set year=" << year << ", month="
        << month << ", day=" << day << ", day id=" << day_id_;
  }
  void BeginPass() {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->BeginPass();
#endif
  }
  void EndPass(bool need_save_delta) {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    box_ptr->EndPass(need_save_delta);

    if (box_ptr->Mode() == 1) {
      box_ptr->PopAucRunnerResource();
    }
#endif
  }
#ifdef PADDLE_WITH_BOX_PS
  void LoadAucRunnerData(PadBoxSlotDataset* dataset,
                         boxps::PSAgentBase* agent) {
    auto box_ptr = BoxWrapper::GetInstance();
    // init random pool slots replace
    static bool slot_init = false;
    if (!slot_init) {
      slot_init = true;
      auto slots_set = dataset->GetSlotsIdx(box_ptr->GetEvalSlotSet());
      box_ptr->SetReplacedSlots(slots_set);
    }
    box_ptr->AddReplaceFeasign(agent, box_ptr->GetFeedpassThreadNum());
    auto& records = dataset->GetInputRecord();
    box_ptr->PushAucRunnerResource(records.size());
    box_ptr->GetRandomReplace(&records);
  }
#endif
  void ReadData2Memory() {
    platform::Timer timer;
    VLOG(3) << "Begin ReadData2Memory(), dataset[" << dataset_ << "]";
#ifdef PADDLE_WITH_BOX_PS
    double feed_pass_span = 0.0;
    double read_ins_span = 0.0;

    timer.Start();
    auto box_ptr = BoxWrapper::GetInstance();
    boxps::PSAgentBase* agent = box_ptr->GetAgent();
    VLOG(3) << "Begin call BeginFeedPass in BoxPS";
    box_ptr->BeginFeedPass(day_id_, &agent);
    timer.Pause();

    feed_pass_span = timer.ElapsedSec();

    PadBoxSlotDataset* dataset = dynamic_cast<PadBoxSlotDataset*>(dataset_);
    dataset->SetPSAgent(agent);

    timer.Start();
    // add 0 key
    agent->AddKey(0ul, 0);
    dataset_->LoadIntoMemory();
    timer.Pause();
    read_ins_span = timer.ElapsedSec();

    timer.Start();
    // auc runner
    if (box_ptr->Mode() == 1) {
      LoadAucRunnerData(dataset, agent);
    }
    box_ptr->EndFeedPass(agent);

    timer.Pause();

    VLOG(0) << "passid = " << dataset->GetPassId()
            << ", begin feedpass: " << feed_pass_span
            << "s, download + parse cost: " << read_ins_span
            << "s, end feedpass:" << timer.ElapsedSec() << "s";
#endif
  }

  void LoadIntoMemory() {
    platform::Timer timer;
    VLOG(3) << "Begin LoadIntoMemory(), dataset[" << dataset_ << "]";
    timer.Start();
    dataset_->LoadIntoMemory();
    timer.Pause();
    VLOG(0) << "download + parse cost: " << timer.ElapsedSec() << "s";

    timer.Start();
    FeedPass();
    timer.Pause();
    VLOG(0) << "FeedPass cost: " << timer.ElapsedSec() << " s";
    VLOG(3) << "End LoadIntoMemory(), dataset[" << dataset_ << "]";
  }
  void PreLoadIntoMemory() {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    boxps::PSAgentBase* agent = box_ptr->GetAgent();
    PadBoxSlotDataset* dataset = dynamic_cast<PadBoxSlotDataset*>(dataset_);
    VLOG(0) << "passid = " << dataset->GetPassId()
            << ", Begin PreLoadIntoMemory BeginFeedPass in BoxPS";
    box_ptr->BeginFeedPass(day_id_, &agent);
    dataset->SetPSAgent(agent);
    // add 0 key
    agent->AddKey(0ul, 0);
    dataset_->PreLoadIntoMemory();
#endif
  }
  void WaitFeedPassDone() {
#ifdef PADDLE_WITH_BOX_PS
    platform::Timer timer;
    timer.Start();
    dataset_->WaitPreLoadDone();
    timer.Pause();

    double wait_done_span = timer.ElapsedSec();

    timer.Start();
    PadBoxSlotDataset* dataset = dynamic_cast<PadBoxSlotDataset*>(dataset_);
    boxps::PSAgentBase* agent = dataset->GetPSAgent();
    auto box_ptr = BoxWrapper::GetInstance();
    // auc runner
    if (box_ptr->Mode() == 1) {
      LoadAucRunnerData(dataset, agent);
    }
    box_ptr->EndFeedPass(agent);
    timer.Pause();

    VLOG(0) << "passid = " << dataset->GetPassId()
            << ", WaitFeedPassDone cost: " << wait_done_span
            << "s, read ins cost: " << dataset->GetReadInsTime()
            << "s, merge cost: " << dataset->GetMergeTime()
            << "s, other cost: " << dataset->GetOtherTime()
            << "s, end feedpass:" << timer.ElapsedSec() << "s";
#endif
  }

  void SlotsShuffle(const std::set<std::string>& slots_to_replace) {
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(box_ptr->Mode(), 1);
    box_ptr->FlipPhase();

    PadBoxSlotDataset* dataset = dynamic_cast<PadBoxSlotDataset*>(dataset_);
    CHECK(dataset);

    auto& records = dataset->GetInputRecord();
    auto slot_idx = dataset->GetSlotsIdx(slots_to_replace);

    if (box_ptr->record_replacers_.size() != records.size()) {
      box_ptr->record_replacers_.resize(records.size());
    }
    if (box_ptr->last_slots_idx_.size() > 0) {
      box_ptr->RecordReplaceBack(&records, box_ptr->last_slots_idx_);
    }
    if (slot_idx.size() > 0) {
      box_ptr->RecordReplace(&records, slot_idx);
    }

    box_ptr->last_slots_idx_ = slot_idx;
#endif
  }
#ifdef PADDLE_WITH_BOX_PS
  // notify boxps to feed this pass feasigns from SSD to memory
  static void FeedPassThread(const std::deque<Record>& t,
                             int begin_index,
                             int end_index,
                             boxps::PSAgentBase* p_agent,
                             const std::unordered_set<int>& index_map,
                             int thread_id) {
    p_agent->AddKey(0ul, thread_id);
    for (auto iter = t.begin() + begin_index; iter != t.begin() + end_index;
         iter++) {
      const auto& ins = *iter;
      const auto& feasign_v = ins.uint64_feasigns_;
      for (const auto feasign : feasign_v) {
        /*
        if (index_map.find(feasign.slot()) != index_map.end()) {
          continue;
        }
        */
        p_agent->AddKey(feasign.sign().uint64_feasign_, thread_id);
      }
    }
  }
#endif
  void FeedPass() {
    VLOG(3) << "Begin FeedPass";
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = BoxWrapper::GetInstance();
    auto input_channel_ =
        dynamic_cast<MultiSlotDataset*>(dataset_)->GetInputChannel();
    const std::deque<Record>& pass_data = input_channel_->GetData();

    // get feasigns that FeedPass doesn't need
    const std::unordered_set<std::string>& slot_name_omited_in_feedpass_ =
        box_ptr->GetOmitedSlot();
    std::unordered_set<int> slot_id_omited_in_feedpass_;
    const auto& all_readers = dataset_->GetReaders();
    PADDLE_ENFORCE_GT(all_readers.size(),
                      0,
                      platform::errors::PreconditionNotMet(
                          "Readers number must be greater than 0."));
    const auto& all_slots_name = all_readers[0]->GetAllSlotAlias();
    for (size_t i = 0; i < all_slots_name.size(); ++i) {
      if (slot_name_omited_in_feedpass_.find(all_slots_name[i]) !=
          slot_name_omited_in_feedpass_.end()) {
        slot_id_omited_in_feedpass_.insert(i);
      }
    }
    const size_t tnum = box_ptr->GetFeedpassThreadNum();
    boxps::PSAgentBase* p_agent = box_ptr->GetAgent();
    VLOG(3) << "Begin call BeginFeedPass in BoxPS";
    box_ptr->BeginFeedPass(day_id_, &p_agent);

    std::vector<std::thread> threads;
    size_t len = pass_data.size();
    size_t len_per_thread = len / tnum;
    auto remain = len % tnum;
    size_t begin = 0;
    for (size_t i = 0; i < tnum; i++) {
      threads.push_back(
          std::thread(FeedPassThread,
                      std::ref(pass_data),
                      begin,
                      begin + len_per_thread + (i < remain ? 1 : 0),
                      p_agent,
                      std::ref(slot_id_omited_in_feedpass_),
                      i));
      begin += len_per_thread + (i < remain ? 1 : 0);
    }
    for (size_t i = 0; i < tnum; ++i) {
      threads[i].join();
    }

    if (box_ptr->Mode() == 1) {
      box_ptr->AddReplaceFeasign(p_agent, tnum);
    }
    VLOG(3) << "Begin call EndFeedPass in BoxPS";
    box_ptr->EndFeedPass(p_agent);
#endif
  }

 private:
  Dataset* dataset_;
  int day_id_ = 0;
  bool get_random_replace_done_ = false;
};

}  // end namespace framework
}  // end namespace paddle
