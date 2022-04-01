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
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
DECLARE_bool(enable_binding_train_cpu);
namespace paddle {
namespace framework {

void BoxPSTrainer::Initialize(const TrainerDesc& trainer_desc,
                              Dataset* dataset) {
  thread_num_ = trainer_desc.thread_num();
  VLOG(3) << "pipeline num: " << thread_num_;

  SetDataset(dataset);
  ParseDumpConfig(trainer_desc);
  // get filelist from trainer_desc here
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();

  param_config_ = trainer_desc.boxps_param();
  async_mode_ = param_config_.async_mode();
  if (async_mode_) {
    dense_table_.reset(new BoxPSAsynDenseTable(thread_num_));
    VLOG(3) << "async mode ";
  }
  dump_thread_num_ = param_config_.dump_thread_num();
  if (need_dump_field_ && dump_thread_num_ <= 1) {
    dump_thread_num_ = 20;
  }

  workers_.resize(thread_num_);
  param_need_sync_.reset(new std::vector<std::string>);

  int sync_dense_mode = param_config_.sync_dense_mode();
  int sync_weight_step = param_config_.sync_weight_step();
  bool sync_one_ring = param_config_.sync_one_ring();
  for (int i = 0; i < thread_num_; ++i) {
    platform::Place place = platform::CUDAPlace(i);
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    auto this_worker =
        std::dynamic_pointer_cast<paddle::framework::BoxPSWorker>(workers_[i]);
    this_worker->SetDeviceIndex(i);
    this_worker->SetThreadIndex(i);
    this_worker->SetDataFeed(readers[i]);
    this_worker->SetReaderPlace(place);
    this_worker->SetNeedDumpField(need_dump_field_);
    this_worker->SetNeedDumpParam(need_dump_param_);
    this_worker->SetDumpFieldVector(dump_fields_);
    this_worker->SetDumpParamVector(dump_param_);
    this_worker->SetPlace(place);
    this_worker->Initialize(trainer_desc);
    this_worker->InitRandomDumpConfig(trainer_desc);
    this_worker->SetParamSyncStep(sync_weight_step);
    this_worker->SetDenseSyncMode(sync_dense_mode);
    this_worker->SetOneRing(sync_one_ring);
  }
  param_need_sync_.reset(
      new std::vector<std::string>(param_config_.param_need_sync().begin(),
                                   param_config_.param_need_sync().end()));
  VLOG(3) << "param_need_sync_ have: ";
  for (const std::string& name : *param_need_sync_) {
    VLOG(3) << name;
  }
  // set debug here
  SetDebug(trainer_desc.debug());
}

void BoxPSTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_) {
    InitDumpEnv();
  }
  VLOG(3) << "init other env done.";
}

std::string BoxPSTrainer::GetDumpPath(int tid) {
  return string::format_string("%s/part-%05d", dump_fields_path_.c_str(), tid);
}

void BoxPSTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  // Only set dump channel on the last section
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->SetChannelWriter(queue_.get());
  }
  // TODO(hutuxian): should make it as a config
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.push_back(
        std::thread(std::bind(&TrainerBase::DumpWork, this, i)));
  }
  VLOG(0) << "init dump write file thread num=" << dump_thread_num_;
}

void BoxPSTrainer::CopyParameters(const Scope& root_scope, int device_id) {
  Scope* thread_scope = GetWorkerScope(device_id);
  for (const std::string& name : *param_need_sync_) {
    const LoDTensor& root_tensor = root_scope.FindVar(name)->Get<LoDTensor>();

    // TODO(hutxian): check a new var of the same name is created in
    LoDTensor* gpu_tensor = thread_scope->Var(name)->GetMutable<LoDTensor>();
    platform::Place place = platform::CUDAPlace(device_id);
    TensorCopy(*static_cast<const Tensor*>(&root_tensor), place,
               static_cast<Tensor*>(gpu_tensor));
  }
}

void BoxPSTrainer::DumpParameters(void) {
  Scope* thread_scope = GetWorkerScope(0);
  for (const auto& var : persistable_vars_) {
    auto* root_tensor = root_scope_->Var(var)->GetMutable<LoDTensor>();
    // TODO(hutuxian): Add a final all-reduce?
    const auto& thread_tensor = thread_scope->FindVar(var)->Get<LoDTensor>();
    TensorCopySync(thread_tensor, platform::CPUPlace(), root_tensor);
  }
}

void BoxPSTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                  const platform::Place& place) {
  PADDLE_ENFORCE(root_scope_, "Null root_scope pointer");
  for (auto& var : main_program.Block(0).AllVars()) {
    if (var->Persistable()) {
      persistable_vars_.push_back(var->Name());
    }
  }

  std::set<std::string> async_param_name;
  if (async_mode_) {
    async_param_name = dense_table_->Init(*root_scope_, *param_need_sync_.get(),
                                          persistable_vars_);
  }
  for (int i = 0; i < thread_num_; ++i) {
    auto this_worker =
        std::dynamic_pointer_cast<paddle::framework::BoxPSWorker>(workers_[i]);
    this_worker->SetRootScope(root_scope_);
    if (async_mode_) {
      this_worker->SetDenseTable(dense_table_.get());
      this_worker->SetAsyncParamName(async_param_name);
    }
    this_worker->CreateDeviceResource(main_program);
    //    CopyParameters(*root_scope_, i);
  }
}
inline std::vector<std::shared_ptr<paddle::framework::ThreadPool>>&
GetThreadPool(int thread_num) {
  static std::vector<std::shared_ptr<paddle::framework::ThreadPool>>
      thread_pools;
  if (!thread_pools.empty()) {
    return thread_pools;
  }
  thread_pools.resize(thread_num);
  for (int i = 0; i < thread_num; ++i) {
    thread_pools[i].reset(new paddle::framework::ThreadPool(1));
  }
  if (!FLAGS_enable_binding_train_cpu) {
    return thread_pools;
  }
  std::vector<int>& train_cores = boxps::get_train_cores();
  if (train_cores.size() < static_cast<size_t>(thread_num)) {
    return thread_pools;
  }
  std::vector<int> ncores;
  for (int i = 0; i < thread_num; ++i) {
    ncores.push_back(train_cores[i]);
    if (train_cores.size() / 2 == static_cast<size_t>(thread_num)) {
      ncores.push_back(train_cores[i + thread_num]);
    }
    thread_pools[i]->SetCPUAffinity(ncores, false);
    ncores.clear();
  }
  return thread_pools;
}
void BoxPSTrainer::Run() {
  VLOG(3) << "Going to run";
  auto pool = GetThreadPool(thread_num_);
  wait_futures_.clear();
  CHECK(static_cast<int>(pool.size()) == thread_num_);
  for (int i = 0; i < thread_num_; ++i) {
    if (!debug_) {
      wait_futures_.emplace_back(
          pool[i]->Run([this, i]() { workers_[i]->TrainFiles(); }));
    } else {
      wait_futures_.emplace_back(
          pool[i]->Run([this, i]() { workers_[i]->TrainFilesWithProfiler(); }));
    }
  }
}

void BoxPSTrainer::Finalize() {
  for (auto& th : wait_futures_) {
    th.get();
  }
  if (async_mode_) {
    // must be after train thread, otherwise the ps_buffer_ will be closed first
    dense_table_->Finalize();
  }
  if (need_dump_field_) {
    FinalizeDumpEnv();
  }
  DumpParameters();
  root_scope_->DropKids();
}

Scope* BoxPSTrainer::GetWorkerScope(int thread_id) {
  return workers_[thread_id]->GetThreadScope();
}

}  // end namespace framework
}  // end namespace paddle
#endif
