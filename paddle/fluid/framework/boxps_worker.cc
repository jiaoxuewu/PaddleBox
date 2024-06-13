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
#ifdef PADDLE_WITH_BOX_PS
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/lodtensor_printer.h"
#if defined(PADDLE_WITH_CUDA)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#elif defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_XPU)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#endif

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
#include "paddle/fluid/framework/data_type_transform.h"
#include "paddle/fluid/framework/program_utils.h"
#include <sys/stat.h>
#include <fcntl.h>

DECLARE_bool(enable_dump_main_program);
DECLARE_bool(enable_sync_dense_moment);
DECLARE_bool(check_nan_inf);
DECLARE_bool(lineid_have_extend_info);
DECLARE_bool(dump_filed_same_as_aibox);
PADDLE_DEFINE_EXPORTED_bool(padbox_enable_gc, true, "enable paddlebox gc");
PADDLE_DEFINE_EXPORTED_bool(padbox_enable_print_op_debug,
                            false,
                            "enable print op debug ,default false");
PADDLE_DEFINE_EXPORTED_bool(enable_print_dump_field_debug,
                            false,
                            "enable print dump field debug ,default false");
PADDLE_DEFINE_EXPORTED_bool(enable_print_dump_info_debug,
                            false,
                            "enable print dump info debug ,default false");
PADDLE_DEFINE_EXPORTED_bool(
    padbox_enable_sharding_stage,
    true,
    "enable sharding stage step1 only param and grad split, default false");
PADDLE_DEFINE_EXPORTED_string(
    padbox_dump_debug_lineid, "", "config dump debug lineid, default is empty");

namespace paddle {
namespace framework {
BoxPSAsynDenseTable::BoxPSAsynDenseTable(const int device_num)
    : device_num_(device_num) {
  int buffer_size = device_num * 4;  // magic number
  device_grads_.resize(buffer_size);
  buffer_poll_.reset(new PSBufferQueue(buffer_size));
  for (int i = 0; i < buffer_size; i++) {
    buffer_poll_->Send(&device_grads_[i]);
  }
  VLOG(1) << "BoxPSAsynDenseTable init finish ";
}
BoxPSAsynDenseTable::~BoxPSAsynDenseTable() {}

std::set<std::string> BoxPSAsynDenseTable::Init(
    const Scope& root_scope,
    const std::vector<std::string>& param_need_sync,
    const std::vector<std::string>& persistable_vars,
    const std::set<std::string>& async_grad_name) {
  std::set<std::string> async_param_name;
  root_scope_ = const_cast<paddle::framework::Scope*>(&root_scope);
  VLOG(1) << "Begin Init For Aysnc Optimize";
  for (const auto& e : param_need_sync) {
    std::string grad_name = e + "@GRAD";
    if (async_grad_name.find(grad_name) == async_grad_name.end()) continue;
    if (e.find("param") != std::string::npos &&
        e.find("pow_acc") == std::string::npos) {
      VLOG(3) << "async mode choose adam param " << e << " to update";
      async_param_list_.push_back(e);
      async_param_list_.push_back(e + "_moment1_0");
      async_param_list_.push_back(e + "_moment2_0");
      async_param_name.insert(e);
      async_param_name.insert(e + "@GRAD");
    }
    if (e.find("summary") != std::string::npos &&
        e.find("batch_s") != std::string::npos) {
      VLOG(3) << "async mode choose norm param " << e << " to update";
      async_norm_param_list_.push_back(e);
      async_param_name.insert(e);
      async_param_name.insert(e + "@GRAD");
    }
  }
  // adam param
  const size_t adam_param_list_size = async_param_list_.size();
  std::sort(
      async_param_list_.begin(),
      async_param_list_
          .end());  // xx_param.b_0, xx_param_moment1_0, xx_param_moment2_0
  for (size_t i = 0; i < async_param_list_.size(); i += 3) {
    const LoDTensor& root_tensor =
        root_scope.FindVar(async_param_list_[i])->Get<LoDTensor>();
    adam_param_len_ += root_tensor.numel();
  }
  // norm param
  std::sort(async_norm_param_list_.begin(),
            async_norm_param_list_
                .end());  // xxsummary.batch_size, xxsummary.batch_square_sum,
                          // xxsummary.batch_sum
  for (size_t i = 0; i < async_norm_param_list_.size(); i += 1) {
    const LoDTensor& root_tensor =
        root_scope.FindVar(async_norm_param_list_[i])->Get<LoDTensor>();
    total_param_len_ += root_tensor.numel();
    async_param_list_.push_back(async_norm_param_list_[i]);
  }
  total_param_len_ += adam_param_len_;
  original_ps_.resize(async_param_list_.size());

  ps_.mutable_data<float>({total_param_len_, 1}, platform::CPUPlace());
  mom1_.mutable_data<float>({adam_param_len_, 1}, platform::CPUPlace());
  mom2_.mutable_data<float>({adam_param_len_, 1}, platform::CPUPlace());
  for (size_t i = 0; i < device_grads_.size(); ++i) {
    device_grads_[i].mutable_data<float>(
        {static_cast<int64_t>(total_param_len_), 1}, platform::CPUPlace());
  }

  int64_t offset = 0;
  for (size_t i = 0; i < async_param_list_.size(); i++) {
    VLOG(3) << "begin to copy " << async_param_list_[i];
    const LoDTensor& root_tensor =
        root_scope.FindVar(async_param_list_[i])->Get<LoDTensor>();
    auto dim = root_tensor.dims();
    size_t len = root_tensor.numel();
    if (i < adam_param_list_size) {
      if (i % 3 == 0) {
        original_ps_[i]
            .ShareDataWith(ps_.Slice(offset, offset + len))
            .Resize(dim);
      } else if (i % 3 == 1) {
        original_ps_[i]
            .ShareDataWith(mom1_.Slice(offset, offset + len))
            .Resize(dim);
      } else {
        original_ps_[i]
            .ShareDataWith(mom2_.Slice(offset, offset + len))
            .Resize(dim);
        offset += len;
      }
    } else {
      VLOG(3) << "original_ps_ ShareDataWith norml name:"
              << async_param_list_[i] << " , i:" << i << " offset:" << offset;
      original_ps_[i]
          .ShareDataWith(ps_.Slice(offset, offset + len))
          .Resize(dim);
      offset += len;
    }
    TensorCopy(*static_cast<const Tensor*>(&root_tensor),
               platform::CPUPlace(),
               static_cast<Tensor*>(&(original_ps_[i])));
  }
  VLOG(3) << "after original_ps_ ShareDataWith offset:" << offset;

  // Copy global lr for async mode
  for (const auto& e : persistable_vars) {
    if (e.find("learning_rate_") != std::string::npos) {
      PADDLE_ENFORCE_LE(
          base_lr_,
          0,
          platform::errors::PreconditionNotMet(
              "lr have been set, previous value: %f, current var is %s",
              base_lr_,
              e.c_str()));
      VLOG(3) << "begin to copy global learning rate: " << e;
      const LoDTensor& root_tensor = root_scope.FindVar(e)->Get<LoDTensor>();
      const float* gpu_lr = root_tensor.data<float>();
      if (platform::is_gpu_place(root_tensor.place()) ||
          platform::is_xpu_place(root_tensor.place())) {
        SyncCopyD2H(&base_lr_, gpu_lr, 1, root_tensor.place());
      } else {
        base_lr_ = *gpu_lr;
      }
    }
  }
  VLOG(0) << "Aysnc alloc dense table param size: " << async_param_list_.size()
          << ", adam param size: " << adam_param_list_size
          << ", total length:" << total_param_len_
          << ", adam length: " << adam_param_len_ << ", base_lr=" << base_lr_;

  ps_buffer_.reset(new PSBufferQueue(device_num_ * 3));  // magic number
  all_lr_.resize(adam_param_len_);
  auto box_ptr = BoxWrapper::GetInstance();
  std::map<std::string, float> lr_map = box_ptr->GetLRMap();
  int lr_index = 0;
  for (size_t i = 0; i < adam_param_list_size / 3; ++i) {
    float learning_rate = base_lr_;
    if (lr_map.find(async_param_list_[i * 3]) != lr_map.end()) {
      learning_rate = lr_map[async_param_list_[i * 3]];
    }
    for (int j = 0; j < original_ps_[i * 3].numel(); j++) {
      all_lr_[lr_index++] = learning_rate;
    }
  }
  InitThreadGroup();
  update_thread_ = new std::thread(&BoxPSAsynDenseTable::AsyncUpdate, this);
  return async_param_name;
}

void BoxPSAsynDenseTable::Finalize(void) {
  if (update_thread_ == nullptr) {
    return;
  }
  ps_buffer_->Close();
  update_thread_->join();
  buffer_poll_->Close();

  for (size_t i = 0; i < async_param_list_.size(); ++i) {
    VLOG(3) << "begin to copy back" << async_param_list_[i];
    auto* root_tensor =
        root_scope_->Var(async_param_list_[i])->GetMutable<LoDTensor>();
    TensorCopySync(*static_cast<const Tensor*>(&original_ps_[i]),
                   platform::CPUPlace(),
                   root_tensor);
  }

  ps_buffer_ = nullptr;
  buffer_poll_ = nullptr;
  delete update_thread_;
  update_thread_ = nullptr;
}

void BoxPSAsynDenseTable::ThreadUpdate(int thread_id,
                                       const std::vector<LoDTensor*>& grad,
                                       size_t merge_num) {
  float* grad_data = grad[0]->mutable_data<float>(platform::CPUPlace());
  float* param_data = ps_.mutable_data<float>(platform::CPUPlace());
  float* mom1_data = mom1_.mutable_data<float>(platform::CPUPlace());
  float* mom2_data = mom2_.mutable_data<float>(platform::CPUPlace());
  // merge grad
  const size_t start = thread_start_index_[thread_id];
  const size_t end = thread_end_index_[thread_id];
  if (merge_num == 2) {
    LoDTensor* grad_tensor_1 = grad[1];
    float* grad_tensor_1_data =
        grad_tensor_1->mutable_data<float>(platform::CPUPlace());
    for (size_t j = start; j < end; ++j) {
      grad_data[j] = (grad_data[j] + grad_tensor_1_data[j]) / 2;
    }
  } else if (merge_num == 3) {
    LoDTensor* grad_tensor_1 = grad[1];
    LoDTensor* grad_tensor_2 = grad[2];
    float* grad_tensor_1_data =
        grad_tensor_1->mutable_data<float>(platform::CPUPlace());
    float* grad_tensor_2_data =
        grad_tensor_2->mutable_data<float>(platform::CPUPlace());
    for (size_t j = start; j < end; ++j) {
      grad_data[j] =
          (grad_data[j] + grad_tensor_1_data[j] + grad_tensor_2_data[j]) / 3;
    }
  } else if (merge_num == 4) {
    LoDTensor* grad_tensor_1 = grad[1];
    LoDTensor* grad_tensor_2 = grad[2];
    LoDTensor* grad_tensor_3 = grad[3];
    float* grad_tensor_1_data =
        grad_tensor_1->mutable_data<float>(platform::CPUPlace());
    float* grad_tensor_2_data =
        grad_tensor_2->mutable_data<float>(platform::CPUPlace());
    float* grad_tensor_3_data =
        grad_tensor_3->mutable_data<float>(platform::CPUPlace());
    for (size_t j = start; j < end; ++j) {
      grad_data[j] = (grad_data[j] + grad_tensor_1_data[j] +
                      grad_tensor_2_data[j] + grad_tensor_3_data[j]) /
                     4;
    }
  }
  VLOG(3) << "ThreadUpdate[" << thread_id << "] start: " << start
          << ", end: " << end
          << ", adam_param_len_: " << (size_t)adam_param_len_;
  for (size_t j = start; j < end; ++j) {
    if (j < (size_t)adam_param_len_) {  // adam
      mom1_data[j] =
          0.99 * mom1_data[j] + 0.01 * grad_data[j];  // magic beta and episilon
      mom2_data[j] =
          0.9999 * mom2_data[j] + 0.0001 * grad_data[j] * grad_data[j];
      param_data[j] -=
          all_lr_[j] * (mom1_data[j] / (sqrt(mom2_data[j]) + 1e-8));
    } else {  // norm
      param_data[j] = param_data[j] * 0.9999999 + grad_data[j];
    }
  }
  return;
}

void BoxPSAsynDenseTable::AsyncUpdate() {
  VLOG(0) << "Begin AsyncUpdate";
  std::vector<LoDTensor*> grad(4, nullptr);  // max package

  auto box_ptr = BoxWrapper::GetInstance();

  while (ps_buffer_->Receive(&grad[0])) {
    size_t merge_num = ps_buffer_->Size() + 1;
    if (merge_num > 4) {
      merge_num = 4;
    }
    for (size_t i = 1; i < merge_num; ++i) {
      ps_buffer_->Receive(&grad[i]);
    }
    phi::AutoWRLock ps_lock(&ps_lock_);
    std::vector<std::future<void>> wait_futures;
    for (int64_t i = 0; i < thread_num_; ++i) {
      wait_futures.emplace_back(thread_pool->Run(
          [this, i, &grad, merge_num]() { ThreadUpdate(i, grad, merge_num); }));
    }

    for (int64_t i = 0; i < thread_num_; ++i) {
      wait_futures[i].get();
    }
    for (size_t i = 0; i < merge_num; ++i) {
      buffer_poll_->Send(grad[i]);
    }
  }

  VLOG(0) << "Quit AsyncUpdate";
}

// async
void BoxPSAsynDenseTable::PullDense(const platform::Place& place,
                                    Tensor* tensor) {
  // while(ps_buffer_->Size() != 0) {//Size have lock, may have perf problem.
  // And will hang when the lock was removed
  //   ;
  // }
  phi::AutoRDLock ps_lock(&ps_lock_);
  TensorCopy(
      *static_cast<const Tensor*>(&ps_), place, static_cast<Tensor*>(tensor));
}
  
void BoxPSAsynDenseTable::PushDense(const platform::Place& place,
                                    Tensor* tensor) {
  LoDTensor* grad = nullptr;
  buffer_poll_->Receive(&grad);
  TensorCopy(*static_cast<const Tensor*>(tensor),
             platform::CPUPlace(),
             static_cast<Tensor*>(grad));
  ps_buffer_->Send(grad);
}

void BoxPSAsynDenseTable::InitThreadGroup() {
  thread_num_ = 32;
  thread_start_index_.resize(thread_num_, 0);
  thread_end_index_.resize(thread_num_, 0);
  size_t prefix_sum = 0;
  size_t thread_update_avg_len = total_param_len_ / thread_num_;
  int unalloc_len = total_param_len_ % thread_num_;
  for (int i = 0; i < thread_num_; i++) {
    thread_start_index_[i] = prefix_sum;
    if (i < unalloc_len) {
      prefix_sum += thread_update_avg_len + 1;
    } else {
      prefix_sum += thread_update_avg_len;
    }
    thread_end_index_[i] = prefix_sum;
  }
  thread_pool.reset(new paddle::framework::ThreadPool(thread_num_));
}
//======================== BoxPSWorker ======================
// init
void BoxPSWorker::MemoryShareTensor::init(const std::string& name,
                                          const platform::Place& place,
                                          const int64_t& total_len,
                                          Scope* root_scope) {
  char szname[512] = {0};
  snprintf(szname,
           sizeof(szname),
           "paddlebox_boxps_worker_share_%s_%d",
           name.c_str(),
           place.GetDeviceId());
  data_tensor_ = root_scope->Var(szname)->GetMutable<phi::DenseTensor>();
  data_tensor_->mutable_data<float>({total_len, 1}, place);
}
// share
phi::DenseTensor& BoxPSWorker::MemoryShareTensor::share(
    phi::DenseTensor* gpu_tensor, const size_t& len) {
  gpu_tensor->ShareDataWith(data_tensor_->Slice(offset_, offset_ + len));
  offset_ += len;
  return *gpu_tensor;
}
  
static const int DenseKStepNode = 1;
static const int DenseKStepALL = 2;
static const int DenseDataNormal = 3;
void BoxPSWorker::Initialize(const TrainerDesc& desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  node_size_ = boxps::MPICluster::Ins().size();
  device_num_ = GetDeviceCount();
  if (device_num_ == 0) {
    device_num_ = desc.thread_num();
  }
  dump_fields_path_ = desc.dump_fields_path();
  dump_thread_num_ = desc.boxps_param().dump_thread_num();
  if (need_dump_field_ || need_dump_param_) {
    if (dump_thread_num_ <= 1) {
      dump_thread_num_ = 20;
    }
    fds_sizes_.resize(dump_thread_num_);
    for (int i = 0; i  < dump_thread_num_; ++i) {
      fds_sizes_[i] = {-1, 0, 0};
    }
    dump_thread_pool_.reset(new paddle::framework::ThreadPool(dump_thread_num_));
    VLOG(0) << "device id=" << device_id_ << ", dump fields path: " << dump_fields_path_ 
            << ", dump thread num: " << dump_thread_num_;
  }
  VLOG(1) << "boxps_worker init device num: " << device_num_;
}
  
void BoxPSWorker::Finalize() {
  if (sharding_mode_ || device_id_ == 0) {
    for (auto& name : need_copy_vars_) {
      Variable* root_var = root_scope_->FindVar(name);
      if (root_var == nullptr) {
        continue;
      }
      auto root_tensor = root_var->GetMutable<phi::DenseTensor>();
      Variable* var = thread_scope_->FindVar(name);
      auto tensor = var->Get<phi::DenseTensor>();
      TensorCopy(tensor, root_tensor->place(), root_tensor);
    }
    dev_ctx_->Wait();
  }
}
void BoxPSWorker::SetDenseTable(BoxPSAsynDenseTable* dense) {
  dense_table_ = dense;
}
inline bool IsDataNormParam(const std::string& name) {
  if (name.find(".batch_size") != std::string::npos ||
      name.find(".batch_sum") != std::string::npos ||
      name.find(".batch_square_sum") != std::string::npos) {
    return true;
  }
  return false;
}
int BoxPSWorker::CheckNeedParam(VarDesc* var) {
  if (!var->Persistable()) {
    return 0;
  }

  std::string name = var->Name();
  if (sync_mode_ == DenseDataNormal) {
    // data normal param
    if (IsDataNormParam(name)) {
      return 3;
    }
  } else {
    size_t len = name.length();
    const char* ext = name.c_str() + len - 4;
    // .w_0  .b_0
    if (strncmp(ext, ".w_0", 4) == 0 || strncmp(ext, ".b_0", 4) == 0) {
      return 1;
    }
    if (FLAGS_enable_sync_dense_moment) {
      if (len < 14) {
        return 0;
      }
      ext = name.c_str() + len - 14;
      // .w_0_moment1_0  .b_0_moment2_0
      if (strncmp(ext, ".w_0_moment1_0", 14) == 0 ||
          strncmp(ext, ".w_0_moment2_0", 14) == 0 ||
          strncmp(ext, ".b_0_moment1_0", 14) == 0 ||
          strncmp(ext, ".b_0_moment2_0", 14) == 0) {
        return 2;
      }
    }
  }
  return 0;
}

int64_t BoxPSWorker::AllocParamTensor(const ProgramDesc& program,
                                      int64_t* pad_len) {
  auto& block = program.Block(0);
  // init var and copy persistable
  int64_t total_param_len = 0;
  int64_t total_moment_len = 0;
  for (auto& var : block.AllVars()) {
    std::string name = var->Name();
    if (!var->Persistable()) {
      continue;
    }
    int flag = CheckNeedParam(var);
    if (flag == 0) {
      continue;
    }
    const LoDTensor& root_tensor = root_scope_->FindVar(name)->Get<LoDTensor>();
    int64_t numel = root_tensor.numel();
    if (flag == 2) {  // moment
      total_moment_len += numel;
    } else {
      total_param_len += numel;
    }
    VLOG(1) << "param name:" << name << ", length:" << numel;
  }

  *pad_len = 0;
  int64_t all_sync_param_len = total_param_len + total_moment_len;
  if ((node_size_ > 1 && !one_ring_)) {
    if ((all_sync_param_len % device_num_) != 0) {
      *pad_len = (device_num_ - (all_sync_param_len % device_num_));
      all_sync_param_len += *pad_len;
    }
  }
  VLOG(2) << "param length:" << total_param_len
          << ", sync length:" << all_sync_param_len
          << ", sync mode:" << sync_mode_ << ", node size:" << node_size_
          << ", device num:" << device_num_ << ", one ring:" << one_ring_;
  param_sync_.init("total_param_sync", place_, all_sync_param_len, root_scope_);
  return total_param_len;
}

int64_t BoxPSWorker::AllocParamTensorAsync(const ProgramDesc& program) {
  auto& block = program.Block(0);
  // init var and copy persistable
  int64_t total_param_len = 0;
  for (auto& var : block.AllVars()) {
    std::string name = var->Name();
    if (!var->Persistable() ||
        async_param_name_.find(name) == async_param_name_.end()) {
      continue;
    }
    const LoDTensor& root_tensor = root_scope_->FindVar(name)->Get<LoDTensor>();
    int64_t numel = root_tensor.numel();
    total_param_len += numel;
  }

  VLOG(2) << "param length:" << total_param_len
          << "param grad length:" << total_param_len
          << ", device num:" << device_num_;

  CHECK(total_param_len > 0) << "error param total zero";
  CHECK(dense_table_->GetParamTotalLen() == total_param_len);

  param_async_.init("total_param_async", place_, total_param_len, root_scope_);
  grad_async_.init("total_grad_async", place_, total_param_len, root_scope_);
  return total_param_len;
}

int BoxPSWorker::IsParameter(const std::string& name, bool full_match) {
  if (full_match) {
    auto it = params2rootid_.find(name);
    if (it == params2rootid_.end()) {
      return -1;
    }
    if (it->second == nccl_rank_id_) {
      return 1;
    }
    return 0;
  } else {
    // moment, acc
    for (auto it = params2rootid_.begin(); it != params2rootid_.end(); ++it) {
      if (strncmp(name.c_str(), it->first.c_str(), it->first.length()) != 0) {
        continue;
      }
      if (it->second == nccl_rank_id_) {
        return 1;
      }
      return 0;
    }
    return -1;
  }
}

static bool FindVarInMap(const VariableNameMap& op_var_map,
                         const std::multiset<std::string>& var_set) {
  for (auto& o : op_var_map) {
    for (auto& name : o.second) {
      if (var_set.find(name) != var_set.end()) {
        return true;
      }
    }
  }
  return false;
}

static bool IsAvgOp(OpDesc* op_desc) {
  if (op_desc->Type() != "elementwise_add" &&
      op_desc->Type() != "elementwise_mul") {
    return false;
  }
  for (auto& o : op_desc->Outputs()) {
    for (auto& name : o.second) {
      if (name.find("avg_weight") != std::string::npos ||
          name.find("@avg") != std::string::npos) {
        return true;
      }
    }
  }
  return false;
}
void BoxPSWorker::BuildShardingDepends(const ProgramDesc& program) {
  nccl_rank_id_ = place_.GetDeviceId();

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_XPU_KP)
  auto box_wrapper = BoxWrapper::GetInstance();
  nccl_rank_id_ = box_wrapper->GetNCCLRankId(nccl_rank_id_);
#endif

  auto& block = program.Block(0);
  auto all_desc = block.AllOps();

  for (auto& op_desc : all_desc) {
    // broadcast op
    if (op_desc->Type() != "c_broadcast") {
      continue;
    }
    int root_id = op_desc->GetAttrIfExists<int>("root");
    int ring_id = op_desc->GetAttrIfExists<int>("ring_id");
    if (ring_id >= 0 && ring_id != ring_id_) {
      ring_id_ = ring_id;
    }
    for (auto& o : op_desc->Inputs()) {
      for (auto& name : o.second) {
        auto var = block.FindVar(name);
        if (!var->Persistable() || !var->IsParameter()) {
          continue;
        }
        if (params2rootid_.find(name) != params2rootid_.end()) {
          auto it = params2rootid_.find(name);
          if (it->second != root_id) {
            std::cout << "error: param name conflict" << std::endl;
          }
          continue;
        }
        params2rootid_.insert(std::make_pair(name, root_id));
      }
    }
  }
  if (params2rootid_.empty()) {
    return;
  }
  sharding_mode_ = true;
  size_t copy_param_cnt = 0;
  // check find
  for (auto& var : block.AllVars()) {
    if (!var->Persistable()) {
      continue;
    }
    std::string name = var->Name();
    int ret = IsParameter(name, var->IsParameter());
    if (ret < 0 || ret == 1) {
      if (ret == 1) {
        persist_param_vars_.insert(name);
      }
      continue;
    }
    if (var->IsParameter()) {
      // persist parameter，eg: data_norm, learning_rate
      if (IsDataNormParam(name) ||
          name.find("learning_rate") != std::string::npos) {
        ++copy_param_cnt;
      } else {
        unpersist_vars_.insert(name);
      }
    } else {
      // adam ubmq1_h2_param.b_0_moment1_0, avg_weight @avg @w_backup
      remove_vars_.insert(name);
    }
  }

  std::multiset<std::string> all_remove_inputs;
  for (auto& op_desc : all_desc) {
    if (FindVarInMap(op_desc->Inputs(), remove_vars_)) {
      for (auto& o : op_desc->Inputs()) {
        for (auto& name : o.second) {
          all_remove_inputs.insert(name);
        }
      }
      remove_ops_.insert(op_desc);
    } else if (IsAvgOp(op_desc) &&
               (FindVarInMap(op_desc->Outputs(), remove_vars_) ||
                FindVarInMap(op_desc->Inputs(), unpersist_vars_))) {
      remove_ops_.insert(op_desc);
    }
  }

  size_t total_scale_cnt = 0;
  size_t remove_scale_cnt = 0;
  // remove scale op
  for (auto& op_desc : all_desc) {
    if (op_desc->Type() != "scale") {
      continue;
    }
    ++total_scale_cnt;
    // check scale output
    for (auto& name : op_desc->Output("Out")) {
      if (all_remove_inputs.find(name) == all_remove_inputs.end()) {
        continue;
      }
      ++remove_scale_cnt;
      remove_ops_.insert(op_desc);
      break;
    }
  }
  // stage1
  if (FLAGS_padbox_enable_sharding_stage) {
    std::multiset<std::string> broadcast_vars;
    for (auto& op_desc : all_desc) {
      if (op_desc->Type() != "c_broadcast") {
        continue;
      }
      bool find = false;
      for (auto& o : op_desc->Inputs()) {
        for (auto& name : o.second) {
          auto it = broadcast_vars.find(name);
          if (it != broadcast_vars.end()) {
            find = true;
            continue;
          }
          broadcast_vars.insert(name);
        }
      }
      if (find) {
        remove_ops_.insert(op_desc);
      }
    }
  }

  // reset dump param
  if (need_dump_param_ && dump_param_ != nullptr) {
    for (auto& name : *dump_param_) {
      auto var = block.FindVar(name);
      if (var == nullptr) {
        continue;
      }
      std::string new_name = name;
      size_t pos = new_name.find("@");
      if (pos > 0) {
        new_name = name.substr(0, pos);
      }
      if (persist_param_vars_.find(new_name) == persist_param_vars_.end()) {
        continue;
      }
      shard_dump_params_.push_back(name);
    }
    dump_param_ = &shard_dump_params_;
  }
  // reset dump fields
  if (need_dump_field_ && dump_fields_ != nullptr) {
    for (auto& name : *dump_fields_) {
      auto var = block.FindVar(name);
      if (var == nullptr) {
        continue;
      }
      if (remove_vars_.find(name) != remove_vars_.end()) {
        continue;
      }
      shard_dump_fields_.push_back(name);
    }
    dump_fields_ = &shard_dump_fields_;
  }
  VLOG(3) << "device id=" << int(place_.GetDeviceId())
          << ", nccl rank=" << nccl_rank_id_
          << ", total param count=" << params2rootid_.size()
          << ", remove op count=" << remove_ops_.size()
          << ", total scale op=" << total_scale_cnt << ", remove "
          << remove_scale_cnt << ", remove var count=" << remove_vars_.size()
          << ", unpersist var count=" << unpersist_vars_.size()
          << ", dump param count=" << shard_dump_params_.size()
          << ", dump fields count=" << shard_dump_fields_.size();
}
inline bool IsCommunicationOp(const std::string& op_name) {
  if (op_name == "c_broadcast" || op_name == "c_reduce_sum" ||
      op_name == "c_allreduce_sum") {
    return true;
  }
  return false;
}
inline bool IsSyncStreamOp(const std::string& op_name) {
  if (op_name == "c_sync_comm_stream" || op_name == "c_sync_calc_stream") {
    return true;
  }
  return false;
}
void BoxPSWorker::CreateThreadOperators(const ProgramDesc& program) {
  auto& block = program.Block(0);

  size_t op_index = 0;
  auto ops_descs = block.AllOps();
  for (auto& op_desc : ops_descs) {
    // skip remove ops
    if (remove_ops_.find(op_desc) != remove_ops_.end()) {
      continue;
    }
    std::string op_name = op_desc->Type();
    // single stream not need sync
    if (IsSyncStreamOp(op_name)) {
      continue;
    }
    // skip feed fetch op
    if (op_name == "feed" || op_name == "fetch") {
      for (auto& o : op_desc->Inputs()) {
        skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
      }
      for (auto& o : op_desc->Outputs()) {
        skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
      }
    }
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
    // change to device stream
    if (IsCommunicationOp(op_name)) {
      ops_[op_index]->SetAttr("use_calc_stream", true);
    }
    ++op_index;
  }
  // add stream sync point
  bool find = false;
  for (size_t op_id = 0; op_id < ops_.size(); ++op_id) {
    auto& op = ops_[op_id];
    std::string op_name = op->Type();
    if (!IsCommunicationOp(op_name)) {
      if (find) {
        find = false;
        sync_points_.insert(op.get());
      }
      continue;
    }
    if (find) {
      continue;
    }
    find = true;
    sync_points_.insert(op.get());
  }

  // skip dump fields
  if (need_dump_field_ && dump_fields_ != nullptr) {
    skip_vars_.insert(
        skip_vars_.end(), dump_fields_->begin(), dump_fields_->end());
  }
  // skip dump param
  if (need_dump_param_ && dump_param_ != nullptr) {
    skip_vars_.insert(
        skip_vars_.end(), dump_param_->begin(), dump_param_->end());
  }
  // add monitor skip vars
  auto box_ptr = BoxWrapper::GetInstance();
  auto& monitor_vars = box_ptr->GetSkipGCVars();
  if (!monitor_vars.empty()) {
    skip_vars_.insert(
        skip_vars_.end(), monitor_vars.begin(), monitor_vars.end());
  }
  if (FLAGS_padbox_enable_gc) {
    // add op gc vars
    unused_vars_ = GetUnusedVars(block, ops_, skip_vars_, &unpersist_vars_);
  }
  VLOG(3) << "device[" << device_id_ << "] total op count=" << block.OpSize()
          << ", create op count=" << ops_.size()
          << ", skip vars count=" << skip_vars_.size()
          << ", unused vars count=" << unused_vars_.size();
}

void BoxPSWorker::CreateThreadScopeForAsync(const ProgramDesc& program) {
  AllocParamTensorAsync(program);

  thread_scope_ = &(root_scope_->NewScope());

  // make param and param@GRAD in same order
  auto& block = program.Block(0);
  std::vector<VarDesc*> sorted_var = block.AllVars();
  std::sort(sorted_var.begin(),
            sorted_var.end(),
            [](const VarDesc* var1, const VarDesc* var2) {
              std::string var1_name = var1->Name();
              std::string var2_name = var2->Name();
              if (var1_name.find("param") != std::string::npos &&
                  var2_name.find("param") == std::string::npos) {
                return true;
              } else if (var1_name.find("param") == std::string::npos &&
                         var2_name.find("param") != std::string::npos) {
                return false;
              } else {
                return var1->Name() < var2->Name();
              }
            });

  // init var and copy persistable
  int grad_var_num = 0;
  int var_num = 0;
  int persistable_num = 0;
  int64_t total_persistable_len = 0;
  int param_total = 0;

  for (auto& var : sorted_var) {
    std::string name = var->Name();
    ++param_total;
    if (!var->Persistable()) {
      if (async_param_name_.find(name) != async_param_name_.end()) {
        // parm@GRAD can not find in root_scope_  use parm length replace
        VLOG(3) << "device[" << device_id_ << "] grad var name " << name;
        const LoDTensor& root_tensor =
            root_scope_->FindVar(name.substr(0, name.length() - 5))
                ->Get<LoDTensor>();
        LoDTensor* gpu_tensor =
            thread_scope_->Var(name)->GetMutable<LoDTensor>();
        auto dim = root_tensor.dims();
        size_t len = root_tensor.numel();
        grad_async_.share(gpu_tensor, len).Resize(dim);
        grad_var_num += 1;
        skip_vars_.push_back(name);
      } else {
        auto* ptr = thread_scope_->Var(name);
        InitializeVariable(ptr, var->GetType());
      }
    } else {
      Variable* root_var = root_scope_->FindVar(name);
      if (!root_var) {
        VLOG(0) << "not found var name=" << name;
        continue;
      }
      if (root_var->IsType<phi::SelectedRows>()) {
        continue;
      }
      phi::DenseTensor* root_tensor = root_var->GetMutable<phi::DenseTensor>();
      size_t len = root_tensor->numel();
      ++persistable_num;
      total_persistable_len += len;

      LoDTensor* gpu_tensor = thread_scope_->Var(name)->GetMutable<LoDTensor>();
      if (async_param_name_.find(name) != async_param_name_.end()) {
        VLOG(3) << "device[" << device_id_ << "] Persistable var name " << name;
        auto dim = root_tensor->dims();
        param_async_.share(gpu_tensor, len).Resize(dim);
        var_num += 1;
      skip_vars_.push_back(name);
      }
      // only support copy
      TensorCopy(*static_cast<const Tensor*>(root_tensor),
                 place_,
                 static_cast<Tensor*>(gpu_tensor));
    }
  }
  VLOG(0) << "device[" << device_id_ << "] total param count=" << param_total
          << ", persistable=" << persistable_num << "("
          << total_persistable_len / 262144.0 << "MB)"
          << ", param_async_ offset:" << param_async_.offset_
          << ", grad_offset: " << grad_async_.offset_
          << ", var_num: " << var_num << ", grad_var_num: " << grad_var_num;
  CHECK(param_async_.offset_ <= param_async_.numel());
  CHECK(grad_async_.offset_ <= grad_async_.numel());
}

void BoxPSWorker::CreateThreadScopeForNorm(const ProgramDesc& program) {
  int64_t pad_len = 0;
  if (sync_mode_ > 0) {
    AllocParamTensor(program, &pad_len);
  }
  thread_scope_ = &(root_scope_->NewScope());

  auto& block = program.Block(0);
  std::vector<VarDesc*> all_vars = block.AllVars();

  // init var and copy persistable
  int persistable_num = 0;
  int share_var_num = 0;
  int64_t share_persistable_len = 0;
  int64_t total_persistable_len = 0;
  int param_total = 0;
  int copy_persist_num = 0;

  for (auto& var : all_vars) {
    std::string name = var->Name();
    ++param_total;
    if (var->Persistable()) {
      Variable* root_var = root_scope_->FindVar(name);
      if (!root_var) {
        VLOG(0) << "not found var name=" << name;
        continue;
      }
      if (root_var->IsType<phi::SelectedRows>()) {
        continue;
      }
      phi::DenseTensor* root_tensor = root_var->GetMutable<phi::DenseTensor>();
      size_t len = root_tensor->numel();
      ++persistable_num;
      total_persistable_len += len;

      LoDTensor* gpu_tensor = thread_scope_->Var(name)->GetMutable<LoDTensor>();
      if (sync_mode_ > 0) {
        if (CheckNeedParam(var)) {
          auto dim = root_tensor->dims();
          param_sync_.share(gpu_tensor, len).Resize(dim);
          skip_vars_.push_back(name);
        }
      }

      // data norm copy and learning rate
      if (!gpu_tensor->initialized() && place_ == root_tensor->place()) {
        auto dim = root_tensor->dims();
        gpu_tensor->ShareDataWith(*root_tensor).Resize(dim);
        ++share_var_num;
        share_persistable_len += len;
      } else {
        TensorCopy(*static_cast<const Tensor*>(root_tensor),
                   place_,
                   static_cast<Tensor*>(gpu_tensor));
        ++copy_persist_num;
        // add copy back to root scope
        if (device_id_ == 0) {
          need_copy_vars_.push_back(name);
          skip_vars_.push_back(name);
        }
      }
    } else {
      auto* ptr = thread_scope_->Var(name);
      InitializeVariable(ptr, var->GetType());
    }
  }
  if (sync_mode_ > 0) {
    CHECK(param_sync_.offset_ <= (param_sync_.numel() - pad_len));
  }
  VLOG(0) << "device[" << device_id_ << "] total param count=" << param_total
          << ", persistable=[total:" << persistable_num << "("
          << total_persistable_len / 262144.0 << "MB)"
          << ", share:" << share_var_num << "("
          << share_persistable_len / 262144.0 << "MB)"
          << ", copy:" << copy_persist_num << "]";
}

void BoxPSWorker::CreateThreadScopeForSharding(const ProgramDesc& program) {
  int64_t pad_len = 0;
  if (sync_mode_ > 0) {
    AllocParamTensor(program, &pad_len);
  }
  thread_scope_ = &(root_scope_->NewScope());

  auto& block = program.Block(0);
  std::vector<VarDesc*> all_vars = block.AllVars();

  // init var and copy persistable
  int persistable_num = 0;
  int share_var_num = 0;
  int64_t share_persistable_len = 0;
  int64_t total_persistable_len = 0;
  int persist_reset = 0;
  int param_total = 0;
  int unpersist_num = 0;
  int copy_persist_num = 0;
  int64_t real_persist_len = 0;
  int real_persist_num = 0;
  int delete_vars_num = 0;

  for (auto& var : all_vars) {
    std::string name = var->Name();
    all_vars_.push_back(name);
    if (remove_vars_.find(name) != remove_vars_.end()) {
      ++delete_vars_num;
      continue;
    }
    thread_vars_.push_back(name);
    ++param_total;
    if (var->Persistable()) {
      if (unpersist_vars_.find(name) != unpersist_vars_.end()) {
        // unpersist vars(include other thread var and other device var)
        auto* ptr = thread_scope_->Var(name);
        InitializeVariable(ptr, var->GetType());
        // set dims
        auto dims = phi::make_ddim(var->GetShape());
        auto var_dtype =
            paddle::framework::TransToPhiDataType(var->GetDataType());
        ptr->GetMutable<phi::DenseTensor>()->Resize(dims).set_type(var_dtype);
        ++unpersist_num;
        ++persistable_num;
        total_persistable_len += ptr->GetMutable<phi::DenseTensor>()->numel();
        continue;
      }
      Variable* root_var = root_scope_->FindVar(name);
      if (!root_var) {
        VLOG(0) << "not found var name=" << name;
        continue;
      }
      if (root_var->IsType<phi::SelectedRows>()) {
        continue;
      }
      phi::DenseTensor* root_tensor = root_var->GetMutable<phi::DenseTensor>();
      size_t len = root_tensor->numel();
      ++persistable_num;
      total_persistable_len += len;
      real_persist_len += len;
      ++real_persist_num;
      // convert one device to other device c_broadcast param
      if (persist_param_vars_.find(name) != persist_param_vars_.end()) {
        // same device
        if (place_ == root_tensor->place()) {
          ++share_var_num;
          share_persistable_len += len;
          continue;
        }

        auto src_place = root_tensor->place();
        auto holder = root_tensor->MoveMemoryHolder();
        auto dst_ptr = root_tensor->mutable_data(
            place_, root_tensor->dtype(), holder->size());
        
        #if defined(PADDLE_WITH_CUDA)
            auto stream = static_cast<phi::GPUContext*>(dev_ctx_)->stream();
            memory::Copy(
                place_, dst_ptr, src_place, holder->ptr(), holder->size(), stream);
            CHECK(platform::is_gpu_place(root_tensor->place()));
        #elif defined(PADDLE_WITH_XPU)
            // XPUStream stream = static_cast<platform::XPUDeviceContext*>(dev_ctx_)
            //                        ->x_context()
            //                        ->xpu_stream;
            memory::Copy(
                place_, dst_ptr, src_place, holder->ptr(), holder->size());
            CHECK(platform::is_xpu_place(root_tensor->place()));
        #endif
        
        ++persist_reset;
        continue;
      }

      LoDTensor* gpu_tensor = thread_scope_->Var(name)->GetMutable<LoDTensor>();
      if (sync_mode_ > 0) {
        if (CheckNeedParam(var)) {
          auto dim = root_tensor->dims();
          param_sync_.share(gpu_tensor, len).Resize(dim);
          skip_vars_.push_back(name);
        }
      }
      // data norm copy and learning rate
      if (!gpu_tensor->initialized() && place_ == root_tensor->place()) {
        auto dim = root_tensor->dims();
        gpu_tensor->ShareDataWith(*root_tensor).Resize(dim);
        ++share_var_num;
        share_persistable_len += len;
      } else {
        TensorCopy(*static_cast<const Tensor*>(root_tensor),
                   place_,
                   static_cast<Tensor*>(gpu_tensor));
        ++copy_persist_num;
        // device 0 need sync datanorm and learning rate to root scope
        if (device_id_ == 0) {
          need_copy_vars_.push_back(name);
          skip_vars_.push_back(name);
        }
      }
    } else {
      auto* ptr = thread_scope_->Var(name);
      InitializeVariable(ptr, var->GetType());
    }
  }
  if (sync_mode_ > 0) {
    CHECK(param_sync_.offset_ <= (param_sync_.numel() - pad_len));
  }
  VLOG(0) << "device[" << device_id_ << "] total param count=" << param_total
          << ", persistable=[total:" << persistable_num << "("
          << total_persistable_len / 262144.0 << "MB)"
          << ", real:" << real_persist_num << "(" << real_persist_len / 262144.0
          << "MB)"
          << ", share:" << share_var_num << "("
          << share_persistable_len / 262144.0 << "MB)"
          << ", reset:" << persist_reset << ", unpersist:" << unpersist_num
          << ", copy:" << copy_persist_num << "], delete=" << delete_vars_num;
}

void BoxPSWorker::CreateDeviceResource(const ProgramDesc& main_prog) {
  BuildShardingDepends(main_prog);
  if (dense_table_) {
    // async
    CreateThreadScopeForAsync(main_prog);
  } else if (sharding_mode_) {
    // sharding mode
    CreateThreadScopeForSharding(main_prog);
  } else {
    // normal
    CreateThreadScopeForNorm(main_prog);
  }
  CreateThreadOperators(main_prog);

  // debug str
  if (FLAGS_enable_dump_main_program) {
    std::ostringstream str_os;
    for (auto& op : ops_) {
      str_os << op->DebugStringEx(thread_scope_);
      // add gc
      auto it = unused_vars_.find(op.get());
      if (it != unused_vars_.end()) {
        str_os << ", gc names: [";
        for (auto& name : it->second) {
          str_os << name << ",";
        }
        str_os << "]";
      }
      str_os << "\n";
    }
    auto box_ptr = BoxWrapper::GetInstance();
    char filename[512] = {0};
    snprintf(filename,
             sizeof(filename),
             "./device_%d_ops_%d.txt",
             thread_id_,
             box_ptr->Phase());
    WriteToFile(filename, str_os.str());
  }
}
void BoxPSWorker::SyncParam(void) {
  if (param_sync_.numel() == 0 ||
      sync_mode_ == DenseKStepNode && node_size_ == 1) {
    return;
  }

  auto box_ptr = BoxWrapper::GetInstance();
  box_ptr->DenseNcclTimer(device_id_, false, 0x03);

#if defined(PADDLE_WITH_CUDA)
  auto comm = platform::NCCLCommContext::Instance().Get(ring_id_, device_id_);
  auto stream = static_cast<phi::GPUContext*>(dev_ctx_)->stream();
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#elif defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_XPU)
  auto comm = platform::BKCLCommContext::Instance().Get(ring_id_, device_id_);
  XPUStream stream = static_cast<platform::XPUDeviceContext*>(dev_ctx_)
                         ->x_context()
                         ->xpu_stream;
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(stream));
#endif
  box_ptr->DenseNcclTimer(device_id_, true, 0x02);
  int64_t numel = param_sync_.numel();
  float* sendbuff = param_sync_.data<float>();

#if defined(PADDLE_WITH_CUDA)
  if ((node_size_ > 1 && !one_ring_)) {  // KStep Node
    int part_param_len = numel / device_num_;
    float* recv_ptr = &sendbuff[device_id_ * part_param_len];

    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
    PADDLE_ENFORCE_GPU_SUCCESS(
        platform::dynload::ncclReduceScatter(sendbuff,
                                             recv_ptr,
                                             part_param_len,
                                             ncclFloat32,
                                             ncclSum,
                                             comm->comm(),
                                             stream));
    CHECK(box_ptr->SyncDense(
        stream, part_param_len, recv_ptr, recv_ptr, device_id_, false));
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
        recv_ptr, sendbuff, part_param_len, ncclFloat32, comm->comm(), stream));
  } else {  // KStep ALL
    PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
        sendbuff, sendbuff, numel, ncclFloat32, ncclSum, comm->comm(), stream));
  }
  const float scale = 1.0 / (device_num_ * node_size_);
  TensorScaleValue(place_, param_sync_.tensor(), &param_sync_.tensor(), scale);
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#elif defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_XPU)
  PADDLE_ENFORCE_EQ(
      bkcl_all_reduce(comm->comm(),
                      sendbuff,
                      sendbuff,
                      numel,
                      BKCL_FLOAT,
                      BKCL_ADD,
                      stream),
      BKCL_SUCCESS,
      platform::errors::PreconditionNotMet("BKCL all reduce failed"));
  const float scale = 1.0 / (device_num_ * node_size_);
  TensorScaleValue(place_, param_sync_.tensor(), &param_sync_.tensor(), scale);
  PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(stream));
#endif

  box_ptr->DenseNcclTimer(device_id_, true, 0x01);
}
int BoxPSWorker::PackBatchTask(void) {
  device_reader_->AssignFeedVar(*thread_scope_);
  return device_reader_->Next();
}

/**
 * @brief add auc monitor
 */
inline void AddAucMonitor(const Scope* scope, const platform::Place& place) {
  auto box_ptr = BoxWrapper::GetInstance();
  auto& metric_list = box_ptr->GetMetricList();
  for (auto iter = metric_list.begin(); iter != metric_list.end(); iter++) {
    auto* metric_msg = iter->second;
    if (box_ptr->Phase() != metric_msg->MetricPhase()) {
      continue;
    }
    metric_msg->add_data(scope, place);
  }
}
void BoxPSWorker::TrainFiles() {
  VLOG(3) << "begin gpubox_worker TrainFiles";
  platform::Timer timer;
  timer.Start();

  int64_t accum_num = 0;
  int batch_size = 0;
  if (device_reader_ != nullptr) {
    device_reader_->Start();
  }
  int step = 0;
  SetDeviceID(device_id_);

  std::unique_ptr<GarbageCollector> gc = nullptr;
  int64_t max_memory_size = GetEagerDeletionThreshold();
  if (FLAGS_padbox_enable_gc && max_memory_size >= 0 && !unused_vars_.empty()) {
    gc = CreateGarbageCollector(place_, max_memory_size);
  }
  while ((batch_size = PackBatchTask()) > 0) {
    VLOG(2) << "[" << device_id_
            << "]begin running ops, batch size:" << batch_size
            << ", batch id=" << step;
    if (dense_table_) {
      dense_table_->PullDense(place_, &param_async_.tensor());
    }
    for (auto& op : ops_) {
      if (FLAGS_padbox_enable_print_op_debug) {
        VLOG(0) << "thread id=" << thread_id_ << ", "
                << op->DebugStringEx(thread_scope_);
      }
      // add stream sync
      if (sync_points_.find(op.get()) != sync_points_.end()) {
        dev_ctx_->Wait();
      }
      op->Run(*thread_scope_, place_);
      if (gc) {
        DeleteUnusedTensors(*thread_scope_, op.get(), unused_vars_, gc.get());
      }
    }
    if (dense_table_) {
      dense_table_->PushDense(place_, &grad_async_.tensor());
    } else if (sync_mode_ > 0) {
      if (step > param_sync_step_) {
        step = 0;
        SyncParam();
      }
    }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_XPU)
    if (FLAGS_check_nan_inf) {
      // check nan result
      if (framework::details::CheckBatchNanOrInfRet(place_)) {
        framework::details::DumpAllScope(*thread_scope_, place_);
        PADDLE_ENFORCE(false,
                       "ERROR: check INF and NAN, device id=%d, batch id=%d",
                       device_id_,
                       step);
      }
    }
#endif
    AddAucMonitor(thread_scope_, place_);

    accum_num += batch_size;
    if (gc) {
      gc->DirectClearCallback([this]() { thread_scope_->DropKids(); });
    } else {
      thread_scope_->DropKids();
    }
    ++step;
    // std::stringstream ss;
    // ss << "Malloc Cnt: ";
    // for (int i = 0; i < 8; ++i) {
    //   ss << "dev: " << i << " malloc times: "<< platform::get_malloc_cnt(i) << " ";
    // }
    // VLOG(0) << ss.str();
  }
  // sync param step
  if (sync_mode_ > 0) {
    SyncParam();
  }
  dev_ctx_->Wait();
  thread_scope_->DropKids();

  timer.Pause();
  auto box_ptr = BoxWrapper::GetInstance();
  box_ptr->PrintSyncTimer(device_id_, timer.ElapsedSec());
}
void BoxPSWorker::TrainFilesWithProfiler() {
  VLOG(3) << "begin section_worker TrainFiles with profiler";

  int64_t step_cnt = 0;
  int64_t accum_num = 0;
  int batch_size = 0;

  platform::Timer reader_timer;
  platform::Timer cal_timer;
  platform::Timer trans_timer;
  platform::Timer sync_timer;
  platform::Timer main_timer;
  platform::Timer outer_timer;
  platform::Timer dump_timer;

  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    op_name.push_back(op->Type());
  }
  op_total_time.resize(ops_.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  device_reader_->Start();

  SetDeviceID(device_id_);

  std::unique_ptr<GarbageCollector> gc = nullptr;
  int64_t max_memory_size = GetEagerDeletionThreshold();
  if (FLAGS_padbox_enable_gc && max_memory_size >= 0 && !unused_vars_.empty()) {
    gc = CreateGarbageCollector(place_, max_memory_size);
  }

  outer_timer.Start();
  while (true) {
    main_timer.Resume();
    reader_timer.Resume();
#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
    TRACE_SCOPE_START("PackBatchTask", dev_ctx_->Wait());
#endif
    batch_size = PackBatchTask();
#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
    TRACE_SCOPE_END("PackBatchTask", dev_ctx_->Wait());
#endif
    reader_timer.Pause();
    if (batch_size <= 0) {
      break;
    }
    VLOG(2) << "[" << device_id_
            << "]begin running ops, batch size:" << batch_size
            << ", batch id=" << step_cnt;

    cal_timer.Resume();
    int op_id = 0;
    dev_ctx_->Wait();
    // std::vector<std::string> op_names;
#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
    TRACE_SCOPE_START("ops run",);
#endif
    for (auto& op : ops_) {
#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
      RUNTIME_TRACE_SCOPE_START((op->Type()+" run").c_str(),);
#endif
      timeline.Start();
      op->Run(*thread_scope_, place_);
      dev_ctx_->Wait();
      timeline.Pause();
      op_total_time[op_id++] += timeline.ElapsedUS();

#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
      RUNTIME_TRACE_SCOPE_END((op->Type()+" run").c_str(),);
#endif

      if (gc) {
        DeleteUnusedTensors(*thread_scope_, op.get(), unused_vars_, gc.get());
      }
    }
    dev_ctx_->Wait();
#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
    TRACE_SCOPE_END("ops run",);
#endif
    cal_timer.Pause();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_XPU)
    if (FLAGS_check_nan_inf) {
      // check nan result
      if (framework::details::CheckBatchNanOrInfRet(place_)) {
        framework::details::DumpAllScope(*thread_scope_, place_);
        PADDLE_ENFORCE(false,
                       "ERROR: check INF and NAN, device id=%d, batch id=%d",
                       device_id_,
                       step_cnt);
      }
    }
#endif

    AddAucMonitor(thread_scope_, place_);

    if (need_dump_field_) {
      dump_timer.Resume();
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
      dump_timer.Pause();
    }
    if (need_dump_param_ && (sharding_mode_ || device_id_ == 0)) {
      dump_timer.Resume();
      DumpParam(*thread_scope_, step_cnt);
      dump_timer.Pause();
    }
    if (gc) {
      gc->DirectClearCallback([this]() { thread_scope_->DropKids(); });
    } else {
      thread_scope_->DropKids();
    }
    ++step_cnt;
    accum_num += batch_size;
    main_timer.Pause();
  }
  if (need_dump_field_ || need_dump_param_) {
    dump_timer.Resume();
    FlushDump();
    dump_timer.Pause();
  }

  thread_scope_->DropKids();
  dev_ctx_->Wait();
  outer_timer.Pause();

  LOG(ERROR) << "log_for_profile"
             << " card:" << device_id_ << " thread:" << thread_id_
             << " step_count:" << step_cnt << " batch_count:" << accum_num
             << " read_time:" << reader_timer.ElapsedUS()
             << " trans_time:" << trans_timer.ElapsedUS()
             << " cal_time:" << cal_timer.ElapsedUS()
             << " sync_time:" << sync_timer.ElapsedUS()
             << " main_time:" << main_timer.ElapsedUS()
             << " outer_time:" << outer_timer.ElapsedUS()
             << " dump_timer:" << dump_timer.ElapsedUS();
  for (size_t i = 0; i < ops_.size(); ++i) {
    LOG(ERROR) << "card:" << device_id_ << ", op: " << op_name[i]
               << ", mean time: " << op_total_time[i] / accum_num
               << "us, sum:" << op_total_time[i] / 1000000.0 << "sec";
  }
  auto box_ptr = BoxWrapper::GetInstance();
  box_ptr->PrintSyncTimer(device_id_, outer_timer.ElapsedSec());
}

//================================== paddlebox dump
//============================================
template <class... ARGS>
inline void format_string_append(
    std::string* str,
    const char* fmt,
    ARGS&&... args) {  // use VA_ARGS may be better ?
  int len = snprintf(NULL, 0, fmt, args...);
  PADDLE_ENFORCE(len >= 0, "format args length error");
  size_t oldlen = str->length();
  str->resize(oldlen + len + 1);
  PADDLE_ENFORCE(snprintf(&(*str)[oldlen], (size_t)len + 1, fmt, args...) ==
                 len);
  str->resize(oldlen + len);
}

static const size_t max_fmt_buff_size = 40;

template <typename T, typename C>
inline void PrintLodTensorFmtType(const Tensor* tensor,
                                  const int64_t& start,
                                  const int64_t& end,
                                  const char* fmt,
                                  std::string* str) {
  int64_t num = end - start;
  if (num <= 0) {
    return;
  }

  size_t oldlen = str->length();
  // resize string
  size_t new_len = oldlen + num * max_fmt_buff_size;
  str->resize(new_len);

  const C* ptr = reinterpret_cast<const C*>(tensor->data<T>());
  for (int64_t i = start; i < end; ++i) {
    int ret = snprintf(&(*str)[oldlen], max_fmt_buff_size, fmt, ptr[i]);
    PADDLE_ENFORCE(ret > 0, "args or buff size error");
    oldlen = oldlen + ret;
  }
  // resize real string
  str->resize(oldlen);
  PADDLE_ENFORCE(new_len >= oldlen, "new len %lu must be larger than oldlen %lu", new_len, oldlen);
}
inline void PrintLodTensor(const Tensor* tensor,
                           const int64_t& start,
                           const int64_t& end,
                           std::string* out) {
  auto dtype = framework::TransToProtoVarType(tensor->dtype());
  if (dtype == proto::VarType::FP32) {
    PrintLodTensorFmtType<float, float>(tensor, start, end, ":%.9g", out);
  } else if (dtype == proto::VarType::INT64) {
    PrintLodTensorFmtType<int64_t, uint64_t>(tensor, start, end, ":%lu", out);
  } else if (dtype == proto::VarType::FP64) {
    PrintLodTensorFmtType<double, double>(tensor, start, end, ":%.9g", out);
  } else if (dtype == proto::VarType::INT32) {
    PrintLodTensorFmtType<int, int>(tensor, start, end, ":%d", out);
  } else if (dtype == proto::VarType::INT16) {
    PrintLodTensorFmtType<int16_t, int16_t>(tensor, start, end, ":%d", out);
  } else {
    out->append("unsupported type");
  }
}

inline bool GetTensorBound(const LoDTensor& tensor,
                           int index,
                           std::pair<int64_t, int64_t>* bound) {
  if (!tensor.IsInitialized()) {
    return false;
  }
  auto& dims = tensor.dims();
  if (dims.size() == 0 || dims[0] <= 0) {
    return false;
  }

  int64_t dim_len = 1;
  if (tensor.lod().size() != 0 && dims.size() == 2) {
    auto& lod = tensor.lod()[0];
    if (lod.size() <= static_cast<size_t>(index + 1)) {
      return false;
    }
    bound->first = lod[index] * dims[1];
    bound->second = lod[index + 1] * dims[1];
  } else {
    if (dims[0] <= static_cast<int64_t>(index)) {
      return false;
    }
    for (int i = 1; i < dims.size(); ++i) {
      dim_len *= dims[i];
    }
    bound->first = index * dim_len;
    bound->second = (index + 1) * dim_len;
  }
  return (dim_len > 0);
}
bool CheckValidTensor(const LoDTensor* tensor, size_t batch_size) {
  auto& dims = tensor->dims();
  if (dims.size() == 0) {
    return false;
  }
  if (tensor->lod().size() != 0) {
    auto& lod = tensor->lod()[0];
    if (lod.size() != batch_size + 1) {
      return false;
    }
  } else {
    if (dims[0] != static_cast<int>(batch_size)) {
      return false;
    }
  }
  return true;
}
static const size_t MAX_FILE_LEN = 1UL << 31;
static const size_t MAX_BUFF_LEN = 4 * 1024 * 1024;
void BoxPSWorker::OpenDump(const int &tid) {
  fd_info_t &fd_info = fds_sizes_[tid];
  if (fd_info.fd >= 0 && fd_info.len < MAX_FILE_LEN) {
    // already opened, just return
    return;
  }
  // close file and open new one
  if (fd_info.fd >= 0) {
    ::close(fd_info.fd);
  }
  std::string filename = string::format_string(
    "%s/part-%02d-%05d-%05d", dump_fields_path_.c_str(), device_id_, tid, fd_info.fileid);
  ++fd_info.fileid;
  
  int fd = ::open(filename.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_APPEND, 0777);
  PADDLE_ENFORCE(fd >= 0, "open %s failed", filename.c_str());
  fd_info.len = 0;
  fd_info.fd = fd;
}
void BoxPSWorker::WriteDump(const int &tid, const std::string& buf) {
  fd_info_t &fd_info = fds_sizes_[tid];
  int ret = ::write(fd_info.fd, buf.c_str(), buf.size());
  fd_info.len += ret;
}
void BoxPSWorker::FlushDump(void) {
  for (auto &fd_info : fds_sizes_) {
    if (fd_info.fd < 0) {
      continue;
    }
    if (fd_info.len > 0) {
      ::fsync(fd_info.fd);
    }
    ::close(fd_info.fd);
    fd_info.fd = -1;
    fd_info.len = 0;
    fd_info.fileid = 0;
  }
}
void BoxPSWorker::DumpParam(const Scope& scope, const int batch_id) {
  platform::Timer timeline;
  timeline.Resume();

  size_t field_num = dump_param_->size();
  // thread process fields
  std::vector<framework::LoDTensor> cpu_tensors(field_num);
  for (size_t i = 0; i < field_num; ++i) {
    auto& name = (*dump_param_)[i];
    auto& cpu_tensor = cpu_tensors[i];
    Variable* var = scope.FindVar(name);
    if (var == nullptr || !var->IsInitialized()) {
      cpu_tensor.clear();
      continue;
    }
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (!tensor.IsInitialized()) {
      VLOG(1) << "Note: param[" << name
              << "] is not initialized, so it was skipped.";
      cpu_tensor.clear();
      continue;
    }
    if (platform::is_gpu_place(tensor.place())) {
      TensorCopy(tensor, platform::CUDAPinnedPlace(), &cpu_tensor);
    } else if (platform::is_xpu_place(tensor.place())) {
      TensorCopySync(tensor, platform::CPUPlace(), &cpu_tensor);
    } else {
      cpu_tensor.ShareDataWith(tensor);
    }
  }
  dev_ctx_->Wait();
  timeline.Pause();

  double copy_time = timeline.ElapsedSec();

  timeline.Resume();
  parallel_run_range(
      field_num, [this, &scope, batch_id, &cpu_tensors](
        const int &tid, const size_t& start, const size_t &end) {
        thread_local std::string s;
        for (size_t i = start; i < end; ++i) {
          auto& cpu_tensor = cpu_tensors[i];
          if (!cpu_tensor.IsInitialized()) {
            continue;
          }
          OpenDump(tid);
          int64_t len = cpu_tensor.numel();
          auto& name = (*dump_param_)[i];
          format_string_append(&s, "(%d,%s,%ld)", batch_id, name.c_str(), len);
          PrintLodTensor(&cpu_tensor, 0, len, &s);
          s.append("\n");
          // write to channel
          WriteDump(tid, s);
          s.clear();
        }
      }, dump_thread_pool_.get());
  timeline.Pause();

  if (FLAGS_enable_print_dump_info_debug) {
    VLOG(0) << "[" << device_id_ << "]dump param field_num=" << field_num
            << ", total span=" << timeline.ElapsedSec()
            << ", copy span=" << copy_time;
  }
}

void BoxPSWorker::DumpField(const Scope& scope,
                            int dump_mode,
                            int dump_interval) {
  // dump_mode: 0: no random, 1: random with insid hash, 2: random with random
  // number
  size_t batch_size = device_reader_->GetCurBatchSize();
  size_t field_num = dump_fields_->size();

  std::vector<framework::LoDTensor> cpu_tensors(field_num);
  std::set<std::string> used_slot_set;

#if (defined PADDLE_WITH_XPU_KP) && (defined PADDLE_WITH_BOX_PS)
  auto real_reader = dynamic_cast<SlotPaddleBoxDataFeed *>(device_reader_);
  PADDLE_ENFORCE_NOT_NULL(
        real_reader, platform::errors::NotFound("In XPU only support SlotPaddleBoxDataFeed"));
  std::vector<std::string> used_slot_names;
  real_reader->GetUsedSlotIndex(nullptr, &used_slot_names);
  for (auto & slot : used_slot_names) {
    used_slot_set.insert(slot);
  }
#endif

  platform::Timer timeline;
  timeline.Resume();
  // copy fields
  for (size_t i = 0; i < field_num; ++i) {
    auto& field = (*dump_fields_)[i];
    Variable* var = scope.FindVar(field);
    auto &cpu_tensor = cpu_tensors[i];
    if (var == nullptr || !var->IsInitialized()) {
      VLOG(3) << "Note: field[" << field
              << "] cannot be find in scope, so it was skipped.";
      cpu_tensor.clear();
      continue;
    }
    const LoDTensor& tensor = var->Get<LoDTensor>();
    if (!tensor.IsInitialized()) {
      VLOG(3) << "Note: field[" << field
              << "] is not initialized, so it was skipped.";
      cpu_tensor.clear();
      continue;
    }
    if (!CheckValidTensor(&tensor, batch_size)) {
      cpu_tensor.clear();
      auto& dims = tensor.dims();
      VLOG(1) << "Note: field[" << field
              << "] cannot pass check, "
                 "so it was skipped. Maybe the dimension is wrong batch size=" 
              << batch_size << ", dims: [" << dims << "]";
      continue;
    }
    if (platform::is_gpu_place(tensor.place())) {
      TensorCopy(tensor, platform::CUDAPinnedPlace(), &cpu_tensor);
    } else if (platform::is_xpu_place(tensor.place())) {
      TensorCopySync(tensor, platform::CPUPlace(), &cpu_tensor);
    } else {
      cpu_tensor.ShareDataWith(tensor);
    }

#ifdef PADDLE_WITH_XPU_KP
    auto fid2sign_map_ptr = paddle::framework::BoxWrapper::GetInstance()->GetFid2SginMap();
    if (used_slot_set.find(field) != used_slot_set.end() \
        && fid2sign_map_ptr != nullptr && fid2sign_map_ptr->size() > 0) {
      auto t_dtype = framework::TransToProtoVarType(cpu_tensors[i].dtype());
      if (t_dtype == proto::VarType::INT64) {
        size_t numel = cpu_tensors[i].numel();
        int64_t * slot_data = cpu_tensors[i].data<int64_t>();
        for (size_t j = 0; j < numel; ++j) {
          uint64_t fid = static_cast<uint64_t>(slot_data[j]);
          PADDLE_ENFORCE_LT(fid, fid2sign_map_ptr->size());
          uint64_t sign = (*fid2sign_map_ptr)[fid];
          PADDLE_ENFORCE(sign > 0 || (sign == 0 && fid == 0),
              platform::errors::PreconditionNotMet(
              "sign can only be 0 when fid is 0, fid:%llu, sign:%llu",
              (unsigned long long)(fid), (unsigned long long)sign));
          slot_data[j] = static_cast<int64_t>(sign);
        }
      }
    }
#endif

  }
  dev_ctx_->Wait();
  // wait stream
  timeline.Pause();

  if (FLAGS_enable_print_dump_field_debug) {
    VLOG(0) << "device id=" << device_id_ << ", field_num=" << field_num 
            << ", batch_size=" << batch_size;
  }
  double copy_time = timeline.ElapsedSec();

  timeline.Resume();
  std::atomic<size_t> line_cnt{0};
  std::atomic<size_t> num_cnt{0};
  // dump data
  parallel_run_range(
      batch_size,
      [this,
       &cpu_tensors,
       field_num,
       dump_mode,
       dump_interval,
       &line_cnt,
       &num_cnt](const int &tid, const size_t& start, const size_t &end) {
        if (FLAGS_enable_print_dump_field_debug) {
          VLOG(0) << "[" << device_id_ << "] begin tid=" << tid <<", field_num=" << field_num;
        }

        thread_local std::default_random_engine engine(0);
        thread_local std::uniform_int_distribution<size_t> dist(0U, INT_MAX);
        thread_local std::pair<int64_t, int64_t> bound;
        thread_local std::string s;
        s.reserve(1024);
        size_t r = 0;
        size_t pos = 0;
        size_t num = 0;

        for (size_t i = start; i < end; ++i) {
          const std::string& lineid = device_reader_->GetLineId(i);
          if (dump_mode == 1) {
            r = XXH64(lineid.data(), lineid.length(), 0);
          } else if (dump_mode == 2) {
            r = dist(engine);
          }
          if (r % dump_interval != 0) {
            continue;
          }
          if (!FLAGS_padbox_dump_debug_lineid.empty() && 
            strncmp(lineid.c_str(), FLAGS_padbox_dump_debug_lineid.c_str(), 32) != 0) {
            continue;
          }
          ++line_cnt;
          OpenDump(tid);    
          if (FLAGS_lineid_have_extend_info) {
            pos = lineid.find(" ");
            if (pos != std::string::npos) {
              s.append(&lineid[0], pos);
            } else {
              s.append(lineid);
            }
          } else {
            s.append(lineid);
          }
          for (size_t k = 0; k < field_num; ++k) {
            auto& tensor = cpu_tensors[k];
            auto& field = (*dump_fields_)[k];
            if (!GetTensorBound(tensor, i, &bound)) {
              continue;
            }
            s.append("\t", 1);
            num += (bound.second - bound.first);
            if (FLAGS_dump_filed_same_as_aibox) {
              size_t ext_pos = field.find(".");
              if (ext_pos != std::string::npos) {
                s.append(&field[0], ext_pos);
              } else {
                s.append(field);
              }
            } else {
              format_string_append(
                  &s, "%s:%ld", field.c_str(), bound.second - bound.first);
            }
            if (FLAGS_enable_print_dump_field_debug) {
              VLOG(0) << "[" << device_id_ << "]tid=" << tid << ", lineid:[" << lineid 
                      << "], name=" << field << ", dims: [" << tensor.dims() << "], len=" << s.length()
                      << ", bound=[" << bound.first << "," << bound.second << "]";
            }
            PrintLodTensor(&tensor, bound.first, bound.second, &s);
            if (s.length() > MAX_BUFF_LEN) {
              WriteDump(tid, s);
              s.clear();
            }
          }

          // append extends tag info
          if (pos > 0) {
            s.append("\t", 1);
            s.append(&lineid[pos + 1], lineid.length() - pos - 1);
          }
          s.append("\n");
          // write to channel
          WriteDump(tid, s);
          s.clear();
          // debug info
          if (FLAGS_enable_print_dump_field_debug) {
            VLOG(0) << "[" << device_id_ << "] tid=" << tid
                    << ", lineid:[" << lineid << "], field_num=" << field_num;
          }
        }
        num_cnt += num;
      }, dump_thread_pool_.get());
  timeline.Pause();
  // add dump warning
  if (line_cnt > 0 && num_cnt == 0) {
    VLOG(0)
        << "WARNING: dump ins count [" << line_cnt << "] fields num ["
        << field_num
        << "] dump values is empty, please check configure fields is vailid!!!";
  }
  if (FLAGS_enable_print_dump_info_debug) {
    VLOG(0) << "[" << device_id_ << "]dump ins count=" << line_cnt
            << ", field=" << field_num << ", num=" << num_cnt
            << ", total span=" << timeline.ElapsedSec()
            << ", copy span=" << copy_time;
  }
}
}  // namespace framework
}  // namespace paddle
#endif
