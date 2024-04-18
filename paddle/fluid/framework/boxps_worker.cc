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

DECLARE_bool(enable_sync_dense_moment);
DECLARE_bool(check_nan_inf);
PADDLE_DEFINE_EXPORTED_bool(padbox_enable_gc, false, "enable paddlebox gc");
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
  VLOG(1) << "boxps_worker init device num: " << device_num_;
}

void BoxPSWorker::SetDenseTable(BoxPSAsynDenseTable* dense) {
  dense_table_ = dense;
}

int BoxPSWorker::CheckNeedParam(VarDesc* var) {
  if (!var->Persistable()) {
    return 0;
  }

  std::string name = var->Name();
  if (sync_mode_ == DenseDataNormal) {
    // data normal param
    if (name.find(".batch_size") != std::string::npos ||
        name.find(".batch_sum") != std::string::npos ||
        name.find(".batch_square_sum") != std::string::npos) {
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

int64_t BoxPSWorker::AllocParamTensor(int64_t* pad_len) {
  auto& block = program_->Block(0);
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
  param_sync_.mutable_data<float>({all_sync_param_len, 1}, place_);
  return total_param_len;
}

int64_t BoxPSWorker::AllocParamTensorAsync() {
  auto& block = program_->Block(0);
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

  param_async_.mutable_data<float>({total_param_len, 1}, place_);
  grad_async_.mutable_data<float>({total_param_len, 1}, place_);
  return total_param_len;
}

void BoxPSWorker::CreateDeviceResource(const ProgramDesc& main_prog) {
  program_.reset(new ProgramDesc(main_prog));
  auto& block = program_->Block(0);
  for (auto& op_desc : block.AllOps()) {
    // skip feed fetch op
    if (op_desc->Type() == "feed" || op_desc->Type() == "fetch") {
      for (auto& o : op_desc->Inputs()) {
        skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
      }
      for (auto& o : op_desc->Outputs()) {
        skip_vars_.insert(skip_vars_.end(), o.second.begin(), o.second.end());
      }
    }
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
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

  int64_t pad_len = 0;
  if (sync_mode_ > 0) {
    AllocParamTensor(&pad_len);
  } else if (dense_table_) {
    AllocParamTensorAsync();
  }

  thread_scope_ = &(root_scope_->NewScope());

  int64_t offset = 0;
  int64_t grad_offset = 0;
  // make param and param@GRAD in same order
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
  int share_var_num = 0;
  int64_t share_persistable_len = 0;
  int64_t total_persistable_len = 0;
  for (auto& var : sorted_var) {
    std::string name = var->Name();
    if (!var->Persistable()) {
      if (dense_table_ &&
          async_param_name_.find(name) != async_param_name_.end()) {
        // parm@GRAD can not find in root_scope_  use parm length replace
        VLOG(3) << "device[" << device_id_ << "] grad var name " << name;
        const LoDTensor& root_tensor =
            root_scope_->FindVar(name.substr(0, name.length() - 5))
                ->Get<LoDTensor>();
        LoDTensor* gpu_tensor =
            thread_scope_->Var(name)->GetMutable<LoDTensor>();
        auto dim = root_tensor.dims();
        size_t len = root_tensor.numel();
        gpu_tensor
            ->ShareDataWith(grad_async_.Slice(grad_offset, grad_offset + len))
            .Resize(dim);
        grad_offset += len;
        grad_var_num += 1;
        skip_vars_.push_back(name);
      } else {
        auto* ptr = thread_scope_->Var(name);
        InitializeVariable(ptr, var->GetType());
      }
    } else {
      const LoDTensor& root_tensor =
          root_scope_->FindVar(name)->Get<LoDTensor>();
      size_t len = root_tensor.numel();
      ++persistable_num;
      total_persistable_len += len;
      // add gc skip vars
      skip_vars_.push_back(name);

      LoDTensor* gpu_tensor = thread_scope_->Var(name)->GetMutable<LoDTensor>();
      if (sync_mode_ > 0) {
        if (CheckNeedParam(var)) {
          auto dim = root_tensor.dims();
          gpu_tensor->ShareDataWith(param_sync_.Slice(offset, offset + len))
              .Resize(dim);
          offset += len;
        }
      } else if (dense_table_) {
        if (async_param_name_.find(name) != async_param_name_.end()) {
          VLOG(3) << "device[" << device_id_ << "] Persistable var name "
                  << name;
          auto dim = root_tensor.dims();
          gpu_tensor->ShareDataWith(param_async_.Slice(offset, offset + len))
              .Resize(dim);
          offset += len;
          var_num += 1;
        }
      }
      if (!gpu_tensor->initialized() && place_ == root_tensor.place()) {
        auto dim = root_tensor.dims();
        gpu_tensor->ShareDataWith(root_tensor).Resize(dim);
        ++share_var_num;
        share_persistable_len += len;
      } else {
        TensorCopy(*static_cast<const Tensor*>(&root_tensor),
                   place_,
                   static_cast<Tensor*>(gpu_tensor));
      }
    }
  }
  if (sync_mode_ > 0) {
    CHECK(offset <= (param_sync_.numel() - pad_len));
  } else if (dense_table_) {
    VLOG(3) << "device[" << device_id_
            << "]CreateDeviceResource param_async_ offset:" << offset
            << " grad_offset: " << grad_offset << " var_num: " << var_num
            << " grad_var_num: " << grad_var_num;
    CHECK(offset <= param_async_.numel());
    CHECK(grad_offset <= grad_async_.numel());
  }
  if (share_var_num > 0) {
    VLOG(0) << "device[" << device_id_ << "] persistable total num ["
            << persistable_num << "," << total_persistable_len << ","
            << total_persistable_len / 262144.0
            << "MB], share persistable num [" << share_var_num << ","
            << share_persistable_len << "," << share_persistable_len / 262144.0
            << "MB]";
  }
  if (FLAGS_padbox_enable_gc) {
    // add op gc vars
    unused_vars_ = GetUnusedVars2(block, ops_, skip_vars_);
    //  for (auto &var : unused_vars_) {
    //    VLOG(0) << "op name=" << var.first->Type() << ", gc names: " <<
    //    paddle::string::join_strings(var.second, ",");
    //  }
    if (device_id_ == 0) {
      VLOG(0) << "total op count=" << ops_.size()
              << ", skip vars count=" << skip_vars_.size()
              << ", unused vars op count=" << unused_vars_.size();
    }
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
  auto comm = platform::NCCLCommContext::Instance().Get(0, device_id_);
  auto stream = static_cast<phi::GPUContext*>(dev_ctx_)->stream();
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
#elif defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_XPU)
  auto comm = platform::BKCLCommContext::Instance().Get(0, device_id_);
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
  TensorScaleValue(place_, param_sync_, &param_sync_, scale);
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
  TensorScaleValue(place_, param_sync_, &param_sync_, scale);
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
      dense_table_->PullDense(place_, &param_async_);
    }

    for (auto& op : ops_) {
      op->Run(*thread_scope_, place_);
      if (gc) {
        DeleteUnusedTensors(*thread_scope_, op.get(), unused_vars_, gc.get());
      }
    }

    if (dense_table_) {
      dense_table_->PushDense(place_, &grad_async_);
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
      DumpFieldBoxPS(*thread_scope_, dump_mode_, dump_interval_);
      dump_timer.Pause();
    }
    if (need_dump_param_ && device_id_ == 0) {
      dump_timer.Resume();
      DumpParamBoxPS(*thread_scope_, step_cnt);
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
    writer_.Flush();
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
}  // namespace framework
}  // namespace paddle
#endif
