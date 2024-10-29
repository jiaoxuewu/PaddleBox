/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#pragma once

#include <ThreadPool.h>

#include <fstream>
#include <memory>
#include <mutex>  // NOLINT
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef PADDLE_WITH_GLOO
#include <gloo/broadcast.h>

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/threadpool.h"
DECLARE_int32(padbox_dataset_shuffle_thread_num);
DECLARE_int32(padbox_dataset_merge_thread_num);
DECLARE_int32(padbox_max_shuffle_wait_count);
DECLARE_bool(enable_shuffle_by_searchid);
DECLARE_bool(padbox_dataset_disable_shuffle);
DECLARE_bool(padbox_dataset_disable_polling);
DECLARE_bool(padbox_dataset_disable_random_update);
namespace boxps {
class PSAgentBase;
}
namespace paddle {
namespace framework {

// Dataset is a abstract class, which defines user interfaces
// Example Usage:
//    Dataset* dataset = DatasetFactory::CreateDataset("InMemoryDataset")
//    dataset->SetFileList(std::vector<std::string>{"a.txt", "b.txt"})
//    dataset->SetThreadNum(1)
//    dataset->CreateReaders();
//    dataset->SetDataFeedDesc(your_data_feed_desc);
//    dataset->LoadIntoMemory();
//    dataset->SetTrainerNum(2);
//    dataset->GlobalShuffle();
class Dataset {
 public:
  Dataset() {}
  virtual ~Dataset() {}
  // do sample
  virtual void TDMSample(const std::string tree_name,
                         const std::string tree_path,
                         const std::vector<uint16_t> tdm_layer_counts,
                         const uint16_t start_sample_layer,
                         const bool with_hierachy,
                         const uint16_t seed_,
                         const uint16_t sample_slot) {}
  // set file list
  virtual void SetFileList(const std::vector<std::string>& filelist) = 0;
  virtual void SetIndexFileList(const std::vector<std::string>& filelist) {}
  // set readers' num
  virtual void SetThreadNum(int thread_num) = 0;
  // set workers' num
  virtual void SetTrainerNum(int trainer_num) = 0;
  // set fleet send batch size
  virtual void SetFleetSendBatchSize(int64_t size) = 0;
  virtual void ReleaseMemoryFun() = 0;
  // set fs name and ugi
  virtual void SetHdfsConfig(const std::string& fs_name,
                             const std::string& fs_ugi) = 0;
  // set customized download command, such as using afs api
  virtual void SetDownloadCmd(const std::string& download_cmd) = 0;
  // set data fedd desc, which contains:
  //   data feed name, batch size, slots
  virtual void SetDataFeedDesc(const std::string& data_feed_desc_str) = 0;
  // set channel num
  virtual void SetChannelNum(int channel_num) = 0;
  // set parse ins id
  virtual void SetParseInsId(bool parse_ins_id) = 0;
  virtual void SetParseContent(bool parse_content) = 0;
  virtual void SetParseLogKey(bool parse_logkey) = 0;
  virtual void SetEnablePvMerge(bool enable_pv_merge) = 0;
  virtual bool EnablePvMerge() = 0;
  virtual void SetMergeBySid(bool is_merge) = 0;
  virtual void SetShuffleByUid(bool enable_shuffle_uid) = 0;
  // set merge by ins id
  virtual void SetMergeByInsId(int merge_size) = 0;
  virtual void SetGenerateUniqueFeasign(bool gen_uni_feasigns) = 0;
  // set fea eval mode
  virtual void SetFeaEval(bool fea_eval, int record_candidate_size) = 0;
  // get file list
  virtual const std::vector<std::string>& GetFileList() = 0;
  // get thread num
  virtual int GetThreadNum() = 0;
  // get worker num
  virtual int GetTrainerNum() = 0;
  // get fleet send batch size
  virtual int64_t GetFleetSendBatchSize() = 0;
  // get hdfs config
  virtual std::pair<std::string, std::string> GetHdfsConfig() = 0;
  // get download cmd
  virtual std::string GetDownloadCmd() = 0;
  // get data fedd desc
  virtual const paddle::framework::DataFeedDesc& GetDataFeedDesc() = 0;
  // get channel num
  virtual int GetChannelNum() = 0;
  // get readers, the reader num depend both on thread num
  // and filelist size
  virtual std::vector<paddle::framework::DataFeed*> GetReaders() = 0;
  // create input channel and output channel
  virtual void CreateChannel() = 0;
  // register message handler between workers
  virtual void RegisterClientToClientMsgHandler() = 0;
  // load all data into memory
  virtual void LoadIntoMemory() = 0;
  // load all data into memory in async mode
  virtual void PreLoadIntoMemory() = 0;
  // wait async load done
  virtual void WaitPreLoadDone() = 0;
  // release all memory data
  virtual void ReleaseMemory() = 0;
  // local shuffle data
  virtual void LocalShuffle() = 0;
  // global shuffle data
  virtual void GlobalShuffle(int thread_num = -1) = 0;
  virtual void SlotsShuffle(const std::set<std::string>& slots_to_replace) = 0;
  // create readers
  virtual void CreateReaders() = 0;
  // destroy readers
  virtual void DestroyReaders() = 0;
  // get memory data size
  virtual int64_t GetMemoryDataSize() = 0;
  // get memory data size in input_pv_channel_
  virtual int64_t GetPvDataSize() = 0;
  // get shuffle data size
  virtual int64_t GetShuffleDataSize() = 0;
  // merge by ins id
  virtual void MergeByInsId() = 0;
  // merge pv instance
  virtual void PreprocessInstance() = 0;
  // divide pv instance
  virtual void PostprocessInstance() = 0;
  // only for untest
  virtual void SetCurrentPhase(int current_phase) = 0;
  virtual void GenerateLocalTablesUnlock(int table_id,
                                         int feadim,
                                         int read_thread_num,
                                         int consume_thread_num,
                                         int shard_num) = 0;
  virtual void ClearLocalTables() = 0;
  // create preload readers
  virtual void CreatePreLoadReaders() = 0;
  // destroy preload readers after prelaod done
  virtual void DestroyPreLoadReaders() = 0;
  // set preload thread num
  virtual void SetPreLoadThreadNum(int thread_num) = 0;
  // seperate train thread and dataset thread
  virtual void DynamicAdjustChannelNum(int channel_num,
                                       bool discard_remaining_ins = false) = 0;
  virtual void DynamicAdjustReadersNum(int thread_num) = 0;
  // set fleet send sleep seconds
  virtual void SetFleetSendSleepSeconds(int seconds) = 0;

  virtual std::vector<std::string> GetSlots() = 0;

  virtual void SetGpuGraphMode(int is_graph_mode) = 0;
  virtual int GetGpuGraphMode() = 0;

  // down load to disk mode
  virtual void SetDiablePolling(bool disable) = 0;
  virtual void SetDisableShuffle(bool disable) = 0;
  virtual void PreLoadIntoDisk(const std::string& path, const int file_num) = 0;
  virtual void WaitLoadDiskDone(void) = 0;
  virtual void SetLoadArchiveFile(bool archive) = 0;

 protected:
  virtual int ReceiveFromClient(int msg_type,
                                int client_id,
                                const std::string& msg) = 0;
};

// DatasetImpl is the implementation of Dataset,
// it holds memory data if user calls load_into_memory
template <typename T>
class DatasetImpl : public Dataset {
 public:
  DatasetImpl();
  virtual ~DatasetImpl() {
    if (release_thread_ != nullptr) {
      release_thread_->join();
    }
  }
  virtual void SetFileList(const std::vector<std::string>& filelist);
  virtual void ReleaseMemoryFun();
  virtual void SetThreadNum(int thread_num);
  virtual void SetTrainerNum(int trainer_num);
  virtual void SetFleetSendBatchSize(int64_t size);
  virtual void SetHdfsConfig(const std::string& fs_name,
                             const std::string& fs_ugi);
  virtual void SetDownloadCmd(const std::string& download_cmd);
  virtual void SetDataFeedDesc(const std::string& data_feed_desc_str);
  virtual void SetChannelNum(int channel_num);
  virtual void SetParseInsId(bool parse_ins_id);
  virtual void SetParseContent(bool parse_content);
  virtual void SetParseLogKey(bool parse_logkey);
  virtual void SetEnablePvMerge(bool enable_pv_merge);
  virtual void SetMergeBySid(bool is_merge);
  virtual void SetShuffleByUid(bool enable_shuffle_uid);

  virtual void SetMergeByInsId(int merge_size);
  virtual void SetGenerateUniqueFeasign(bool gen_uni_feasigns);
  virtual void SetFeaEval(bool fea_eval, int record_candidate_size);
  virtual const std::vector<std::string>& GetFileList() { return filelist_; }
  virtual int GetThreadNum() { return thread_num_; }
  virtual int GetTrainerNum() { return trainer_num_; }
  virtual Channel<T> GetInputChannel() { return input_channel_; }
  virtual void SetInputChannel(const Channel<T>& input_channel) {
    input_channel_ = input_channel;
  }
  virtual int64_t GetFleetSendBatchSize() { return fleet_send_batch_size_; }
  virtual std::pair<std::string, std::string> GetHdfsConfig() {
    return std::make_pair(fs_name_, fs_ugi_);
  }
  virtual void SetGpuGraphMode(int is_graph_mode);
  virtual int GetGpuGraphMode();
  virtual std::string GetDownloadCmd();
  virtual const paddle::framework::DataFeedDesc& GetDataFeedDesc() {
    return data_feed_desc_;
  }
  virtual int GetChannelNum() { return channel_num_; }
  virtual bool EnablePvMerge() { return enable_pv_merge_; }
  virtual std::vector<paddle::framework::DataFeed*> GetReaders();
  virtual void CreateChannel();
  virtual void RegisterClientToClientMsgHandler();
  virtual void LoadIntoMemory();
  virtual void PreLoadIntoMemory();
  virtual void WaitPreLoadDone();
  virtual void ReleaseMemory();
  virtual void LocalShuffle();
  virtual void GlobalShuffle(int thread_num = -1) {}
  virtual void SlotsShuffle(const std::set<std::string>& slots_to_replace) {}
  virtual const std::vector<T>& GetSlotsOriginalData() {
    return slots_shuffle_original_data_;
  }
  virtual void CreateReaders();
  virtual void DestroyReaders();
  virtual int64_t GetMemoryDataSize();
  virtual int64_t GetPvDataSize();
  virtual int64_t GetShuffleDataSize();
  virtual void MergeByInsId() {}
  virtual void PreprocessInstance() {}
  virtual void PostprocessInstance() {}
  virtual void SetCurrentPhase(int current_phase) {}
  virtual void GenerateLocalTablesUnlock(int table_id,
                                         int feadim,
                                         int read_thread_num,
                                         int consume_thread_num,
                                         int shard_num) {}
  virtual void ClearLocalTables() {}
  virtual void CreatePreLoadReaders();
  virtual void DestroyPreLoadReaders();
  virtual void SetPreLoadThreadNum(int thread_num);
  virtual void DynamicAdjustChannelNum(int channel_num,
                                       bool discard_remaining_ins = false);
  virtual void DynamicAdjustReadersNum(int thread_num);
  virtual void SetFleetSendSleepSeconds(int seconds);
  virtual std::vector<std::string> GetSlots();
  /* for enable_heterps_
  virtual void EnableHeterps(bool enable_heterps) {
    enable_heterps_ = enable_heterps;
  }
  */

  std::vector<paddle::framework::Channel<T>>& GetMultiOutputChannel() {
    return multi_output_channel_;
  }

  std::vector<paddle::framework::Channel<T>>& GetCurOutputChannel() {
    if (cur_channel_ == 0) {
      return multi_output_channel_;
    } else {
      return multi_consume_channel_;
    }
  }
  std::vector<uint64_t>& GetGpuGraphTotalKeys() {
    return gpu_graph_total_keys_;
  }
  Channel<T>& GetInputChannelRef() { return input_channel_; }
  // aucrunner need
  virtual std::vector<T>& GetInputRecord() { return input_records_; }
  // disable shuffle
  virtual void SetDiablePolling(bool disable) {}
  virtual void SetDisableShuffle(bool disable) {}
  virtual void PreLoadIntoDisk(const std::string& path, const int file_num) {}
  virtual void WaitLoadDiskDone(void) {}
  virtual void SetLoadArchiveFile(bool archive) {}
 protected:
  virtual int ReceiveFromClient(int msg_type,
                                int client_id,
                                const std::string& msg) {
    // TODO(yaoxuefeng) for SlotRecordDataset
    return -1;
  }
  std::vector<std::shared_ptr<paddle::framework::DataFeed>> readers_;
  std::vector<std::shared_ptr<paddle::framework::DataFeed>> preload_readers_;
  paddle::framework::Channel<T> input_channel_;
  paddle::framework::Channel<PvInstance> input_pv_channel_;
  std::vector<paddle::framework::Channel<PvInstance>> multi_pv_output_;
  std::vector<paddle::framework::Channel<PvInstance>> multi_pv_consume_;

  int channel_num_;
  std::vector<paddle::framework::Channel<T>> multi_output_channel_;
  std::vector<paddle::framework::Channel<T>> multi_consume_channel_;
  std::vector<std::unordered_set<uint64_t>> local_tables_;
  // when read ins, we put ins from one channel to the other,
  // and when finish reading, we set cur_channel = 1 - cur_channel,
  // so if cur_channel=0, all data are in output_channel, else consume_channel
  int cur_channel_;
  std::vector<T> slots_shuffle_original_data_;
  RecordCandidateList slots_shuffle_rclist_;
  int thread_num_;
  int pull_sparse_to_local_thread_num_;
  paddle::framework::DataFeedDesc data_feed_desc_;
  int trainer_num_;
  std::vector<std::string> filelist_;
  size_t file_idx_;
  uint64_t total_fea_num_;
  std::mutex mutex_for_pick_file_;
  std::mutex mutex_for_fea_num_;
  std::string fs_name_;
  std::string fs_ugi_;
  int64_t fleet_send_batch_size_;
  int64_t fleet_send_sleep_seconds_;
  std::vector<std::thread> preload_threads_;
  std::thread* release_thread_ = nullptr;
  bool merge_by_insid_;
  bool parse_ins_id_;
  bool parse_content_;
  bool parse_logkey_;
  bool merge_by_sid_;
  bool shuffle_by_uid_;
  bool parse_uid_;
  bool enable_pv_merge_;  // True means to merge pv
  int current_phase_;     // 1 join, 0 update
  size_t merge_size_;
  bool slots_shuffle_fea_eval_ = false;
  bool gen_uni_feasigns_ = false;
  int preload_thread_num_;
  std::mutex global_index_mutex_;
  int64_t global_index_ = 0;
  std::vector<std::shared_ptr<ThreadPool>> consume_task_pool_;
  std::vector<T> input_records_;  // only for paddleboxdatafeed
  std::vector<std::string> use_slots_;
  bool enable_heterps_ = false;
  int gpu_graph_mode_ = 0;
  // std::vector<std::vector<int64_t>> gpu_graph_device_keys_;
  std::vector<std::vector<std::vector<uint64_t>>> graph_all_type_total_keys_;
  std::vector<uint64_t> gpu_graph_total_keys_;
};

// use std::vector<MultiSlotType> or Record as data type
class MultiSlotDataset : public DatasetImpl<Record> {
 public:
  MultiSlotDataset() {}
  virtual void TDMSample(const std::string tree_name,
                         const std::string tree_path,
                         const std::vector<uint16_t> tdm_layer_counts,
                         const uint16_t start_sample_layer,
                         const bool with_hierachy,
                         const uint16_t seed_,
                         const uint16_t sample_slot);
  virtual void MergeByInsId();
  virtual void PreprocessInstance();
  virtual void PostprocessInstance();
  virtual void SetCurrentPhase(int current_phase);
  virtual void GenerateLocalTablesUnlock(int table_id,
                                         int feadim,
                                         int read_thread_num,
                                         int consume_thread_num,
                                         int shard_num);
  virtual void ClearLocalTables() {
    for (auto& t : local_tables_) {
      t.clear();
      std::unordered_set<uint64_t>().swap(t);
    }
    std::vector<std::unordered_set<uint64_t>>().swap(local_tables_);
  }
  virtual void PreprocessChannel(
      const std::set<std::string>& slots_to_replace,
      std::unordered_set<uint16_t>& index_slot);  // NOLINT
  virtual void SlotsShuffle(const std::set<std::string>& slots_to_replace);
  virtual void GetRandomData(
      const std::unordered_set<uint16_t>& slots_to_replace,
      std::vector<Record>* result);
  virtual ~MultiSlotDataset() {}
  virtual void GlobalShuffle(int thread_num = -1);
  virtual void DynamicAdjustReadersNum(int thread_num);
  virtual void PrepareTrain();

 protected:
  virtual int ReceiveFromClient(int msg_type,
                                int client_id,
                                const std::string& msg);
};
class SlotRecordDataset : public DatasetImpl<SlotRecord> {
 public:
  SlotRecordDataset() { SlotRecordPool(); }
  virtual ~SlotRecordDataset() {}
  // create input channel
  virtual void CreateChannel();
  // create readers
  virtual void CreateReaders();
  // release memory
  virtual void ReleaseMemory();
  virtual void GlobalShuffle(int thread_num = -1);
  virtual void DynamicAdjustChannelNum(int channel_num,
                                       bool discard_remaining_ins);
  virtual void PrepareTrain();
  virtual void DynamicAdjustReadersNum(int thread_num);

 protected:
  bool enable_heterps_ = true;
};

#ifdef PADDLE_WITH_BOX_PS
class PadBoxSlotDataset : public DatasetImpl<SlotRecord> {
 public:
  PadBoxSlotDataset();
  virtual ~PadBoxSlotDataset();
  // seperate train thread and dataset thread
  virtual void DynamicAdjustChannelNum(int channel_num,
                                       bool discard_remaining_ins = false) {
    // not need to
  }
  // dynamic adjust reader num
  virtual void DynamicAdjustReadersNum(int thread_num);
  // set file list
  virtual void SetFileList(const std::vector<std::string>& filelist);
  // create input channel and output channel
  virtual void CreateChannel();
  // load all data into memory
  virtual void LoadIntoMemory();
  // release all memory data
  virtual void ReleaseMemory();
  // create readers
  virtual void CreateReaders();
  // destroy readers
  virtual void DestroyReaders();
  // merge pv instance
  virtual void PreprocessInstance();
  // restore
  virtual void PostprocessInstance();
  // prepare train do something
  virtual void PrepareTrain(void);
  virtual int64_t GetMemoryDataSize() {
    if (input_records_.empty()) {
      return total_ins_num_;
    }
    return input_records_.size();
  }
  virtual int64_t GetPvDataSize() { return input_pv_ins_.size(); }
  virtual int64_t GetShuffleDataSize() { return input_records_.size(); }
  // merge ins from multiple sources and unroll
  virtual void UnrollInstance();
  virtual void ReceiveSuffleData(const int client_id, const char* msg, int len);

  // pre load
  virtual void LoadIndexIntoMemory() {}
  virtual void PreLoadIntoMemory();
  virtual void WaitPreLoadDone();

  virtual void SetDiablePolling(bool disable) { disable_polling_ = disable; }
  virtual void SetDisableShuffle(bool disable) { disable_shuffle_ = disable; }
  virtual void PreLoadIntoDisk(const std::string& path, const int file_num);
  virtual void WaitLoadDiskDone(void);
  virtual void SetLoadArchiveFile(bool archive) { is_archive_file_ = archive; }


 protected:
  // shuffle data
  virtual void ShuffleData(int thread_num = -1);

 public:
  void SetPSAgent(boxps::PSAgentBase* agent) { p_agent_ = agent; }
  boxps::PSAgentBase* GetPSAgent(void) { return p_agent_; }
  double GetReadInsTime(void) { return max_read_ins_span_; }
  double GetOtherTime(void) { return other_timer_.ElapsedSec(); }
  double GetMergeTime(void) { return max_merge_ins_span_; }
  uint16_t GetPassId(void) { return pass_id_; }
  // aucrunner
  std::set<uint16_t> GetSlotsIdx(const std::set<std::string>& str_slots) {
    std::set<uint16_t> slots_idx;
    uint16_t idx = 0;
    auto multi_slot_desc = data_feed_desc_.multi_slot_desc();
    for (int i = 0; i < multi_slot_desc.slots_size(); ++i) {
      auto slot = multi_slot_desc.slots(i);
      if (!slot.is_used() || slot.type().at(0) != 'u') {
        continue;
      }
      if (str_slots.find(slot.name()) != str_slots.end()) {
        slots_idx.insert(idx);
      }
      ++idx;
    }

    return slots_idx;
  }

 protected:
  void MergeInsKeys(const Channel<SlotRecord>& in);
  void CheckThreadPool(void);
  void CheckDownThreadPool(void);
  void DumpIntoDisk(const Channel<SlotRecord>& in, const std::string& path,
                    const int pass_num);
  std::function<uint64_t(const SlotRecord&)> general_shuffle_func(void);

 protected:
  Channel<SlotRecord> shuffle_channel_ = nullptr;
  std::vector<int> mpi_flags_;
  std::atomic<int> finished_counter_{0};
  int mpi_size_ = 1;
  int mpi_rank_ = 0;
  std::vector<SlotPvInstance> input_pv_ins_;
  int shuffle_thread_num_ = FLAGS_padbox_dataset_shuffle_thread_num;
  std::atomic<int> shuffle_counter_{0};
  void* data_consumer_ = nullptr;
  std::atomic<int> receiver_cnt_{0};
  boxps::PSAgentBase* p_agent_ = nullptr;
  paddle::framework::ThreadPool* thread_pool_ = nullptr;
  std::vector<std::future<void>> wait_futures_;
  double max_read_ins_span_ = 0;
  double min_read_ins_span_ = 0;
  platform::Timer other_timer_;
  double max_merge_ins_span_ = 0;
  double min_merge_ins_span_ = 0;
  std::atomic<int> read_ins_ref_{0};
  std::atomic<int> merge_ins_ref_{0};
  std::mutex merge_mutex_;
  std::vector<int> used_fea_index_;
  int merge_thread_num_ = FLAGS_padbox_dataset_merge_thread_num;
  paddle::framework::ThreadPool* merge_pool_ = nullptr;
  paddle::framework::ThreadPool* shuffle_pool_ = nullptr;
  uint16_t pass_id_ = 0;
  double max_shuffle_span_ = 0;
  double min_shuffle_span_ = 0;
  bool disable_shuffle_ = FLAGS_padbox_dataset_disable_shuffle;
  bool disable_polling_ = FLAGS_padbox_dataset_disable_polling;
  bool disable_random_update_ = FLAGS_padbox_dataset_disable_random_update;
  std::vector<std::shared_ptr<BinaryArchiveWriter>> binary_files_;
  bool is_archive_file_ = false;
  std::atomic<int64_t> total_ins_num_{0};
  paddle::framework::ThreadPool* down_pool_ = nullptr;
  paddle::framework::ThreadPool* dump_pool_ = nullptr;
  SlotObjPool* slot_pool_ = nullptr;
};

class InputTableDataset : public PadBoxSlotDataset {
 public:
  virtual void SetIndexFileList(const std::vector<std::string>& filelist) {
    index_filelist_ = filelist;
  }
  virtual void LoadIndexIntoMemory();

 private:
  std::vector<std::string> index_filelist_;
};
#endif
}  // end namespace framework
}  // end namespace paddle
