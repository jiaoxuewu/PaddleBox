/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#include "paddle/fluid/framework/data_feed.h"

#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#ifdef _LINUX
#include <stdio_ext.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif
#include "io/fs.h"
#include "paddle/fluid/platform/monitor.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#endif

USE_INT_STAT(STAT_total_feasign_num_in_mem);
DECLARE_bool(enable_ins_parser_file);

#ifdef PADDLE_WITH_BOX_PS
#include <dlfcn.h>
extern "C" {
typedef paddle::framework::ISlotParser* (*MyPadBoxGetObject)(void);
typedef void (*MyPadBoxFreeObject)(paddle::framework::ISlotParser*);
}
#endif
namespace paddle {
namespace framework {

DLManager& global_dlmanager_pool() {
  static DLManager manager;
  return manager;
}

class BufferedLineFileReader {
  typedef std::function<bool()> SampleFunc;
  static const int MAX_FILE_BUFF_SIZE = 4 * 1024 * 1024;
  class FILEReader {
   public:
    explicit FILEReader(FILE* fp) : fp_(fp) {}
    int read(char* buf, int len) { return fread(buf, sizeof(char), len, fp_); }

   private:
    FILE* fp_;
  };

 public:
  typedef std::function<bool(const std::string&)> LineFunc;

 private:
  template <typename T>
  int read_lines(T* reader, LineFunc func, int skip_lines) {
    int lines = 0;
    size_t ret = 0;
    char* ptr = NULL;
    char* eol = NULL;
    total_len_ = 0;
    error_line_ = 0;

    SampleFunc spfunc = get_sample_func();
    std::string x;
    while (!is_error() && (ret = reader->read(buff_, MAX_FILE_BUFF_SIZE)) > 0) {
      total_len_ += ret;
      ptr = buff_;
      eol = reinterpret_cast<char*>(memchr(ptr, '\n', ret));
      while (eol != NULL) {
        int size = static_cast<int>((eol - ptr) + 1);
        x.append(ptr, size - 1);
        ++lines;
        if (lines > skip_lines && spfunc()) {
          if (!func(x)) {
            ++error_line_;
          }
        }

        x.clear();
        ptr += size;
        ret -= size;
        eol = reinterpret_cast<char*>(memchr(ptr, '\n', ret));
      }
      if (ret > 0) {
        x.append(ptr, ret);
      }
    }
    if (!is_error() && !x.empty()) {
      ++lines;
      if (lines > skip_lines && spfunc()) {
        if (!func(x)) {
          ++error_line_;
        }
      }
    }
    return lines;
  }

 public:
  BufferedLineFileReader()
      : random_engine_(std::random_device()()),
        uniform_distribution_(0.0f, 1.0f) {
    total_len_ = 0;
    sample_line_ = 0;
    buff_ =
        reinterpret_cast<char*>(calloc(MAX_FILE_BUFF_SIZE + 1, sizeof(char)));
  }
  ~BufferedLineFileReader() { free(buff_); }

#ifdef PADDLE_WITH_BOX_PS
  int read_api(boxps::PaddleDataReader* reader, LineFunc func, int skip_lines) {
    return read_lines<boxps::PaddleDataReader>(reader, func, skip_lines);
  }
#endif
  int read_file(FILE* fp, LineFunc func, int skip_lines) {
    FILEReader reader(fp);
    return read_lines<FILEReader>(&reader, func, skip_lines);
  }
  uint64_t file_size(void) { return total_len_; }
  void set_sample_rate(float r) { sample_rate_ = r; }
  size_t get_sample_line() { return sample_line_; }
  bool is_error(void) { return (error_line_ > 10); }

 private:
  SampleFunc get_sample_func() {
    if (std::abs(sample_rate_ - 1.0f) < 1e-5f) {
      return [this](void) { return true; };
    }
    return [this](void) {
      return (uniform_distribution_(random_engine_) < sample_rate_);
    };
  }

 private:
  char* buff_ = nullptr;
  uint64_t total_len_ = 0;

  std::default_random_engine random_engine_;
  std::uniform_real_distribution<float> uniform_distribution_;
  float sample_rate_ = 1.0f;
  size_t sample_line_ = 0;
  size_t error_line_ = 0;
};
void RecordCandidateList::ReSize(size_t length) {
  mutex_.lock();
  capacity_ = length;
  CHECK(capacity_ > 0);  // NOLINT
  candidate_list_.clear();
  candidate_list_.resize(capacity_);
  full_ = false;
  cur_size_ = 0;
  total_size_ = 0;
  mutex_.unlock();
}

void RecordCandidateList::ReInit() {
  mutex_.lock();
  full_ = false;
  cur_size_ = 0;
  total_size_ = 0;
  mutex_.unlock();
}

void RecordCandidateList::AddAndGet(const Record& record,
                                    RecordCandidate* result) {
  mutex_.lock();
  size_t index = 0;
  ++total_size_;
  auto fleet_ptr = FleetWrapper::GetInstance();
  if (!full_) {
    candidate_list_[cur_size_++] = record;
    full_ = (cur_size_ == capacity_);
  } else {
    CHECK(cur_size_ == capacity_);
    index = fleet_ptr->LocalRandomEngine()() % total_size_;
    if (index < capacity_) {
      candidate_list_[index] = record;
    }
  }
  index = fleet_ptr->LocalRandomEngine()() % cur_size_;
  *result = candidate_list_[index];
  mutex_.unlock();
}

void DataFeed::AddFeedVar(Variable* var, const std::string& name) {
  CheckInit();
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (name == use_slots_[i]) {
      if (var == nullptr) {
        feed_vec_[i] = nullptr;
      } else {
        feed_vec_[i] = var->GetMutable<LoDTensor>();
      }
    }
  }
}

bool DataFeed::SetFileList(const std::vector<std::string>& files) {
  std::unique_lock<std::mutex> lock(*mutex_for_pick_file_);
  CheckInit();
  // Do not set finish_set_filelist_ flag,
  // since a user may set file many times after init reader
  filelist_.assign(files.begin(), files.end());

  finish_set_filelist_ = true;
  return true;
}

void DataFeed::SetBatchSize(int batch_size) {
  PADDLE_ENFORCE_GT(batch_size,
                    0,
                    platform::errors::InvalidArgument(
                        "Batch size %d is illegal.", batch_size));
  default_batch_size_ = batch_size;
}

bool DataFeed::PickOneFile(std::string* filename) {
  PADDLE_ENFORCE_NOT_NULL(
      mutex_for_pick_file_,
      platform::errors::PreconditionNotMet(
          "You should call SetFileListMutex before PickOneFile"));
  PADDLE_ENFORCE_NOT_NULL(
      file_idx_,
      platform::errors::PreconditionNotMet(
          "You should call SetFileListIndex before PickOneFile"));
  std::unique_lock<std::mutex> lock(*mutex_for_pick_file_);
  VLOG(4) << "filelist_ size: " << filelist_.size();
  if (*file_idx_ == filelist_.size()) {
    VLOG(3) << "DataFeed::PickOneFile no more file to pick";
    return false;
  }
  VLOG(3) << "file_idx_=" << *file_idx_;
  *filename = filelist_[(*file_idx_)++];
  return true;
}

void DataFeed::CheckInit() {
  PADDLE_ENFORCE_EQ(
      finish_init_,
      true,
      platform::errors::PreconditionNotMet("DataFeed initialization failed."));
}

void DataFeed::CheckSetFileList() {
  PADDLE_ENFORCE_EQ(
      finish_set_filelist_,
      true,
      platform::errors::PreconditionNotMet("DataFeed set filelist failed."));
}

void DataFeed::CheckStart() {
  PADDLE_ENFORCE_EQ(finish_start_,
                    true,
                    platform::errors::PreconditionNotMet(
                        "Datafeed has not started running yet."));
}

void DataFeed::AssignFeedVar(const Scope& scope) {
  CheckInit();
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    feed_vec_[i] = scope.FindVar(use_slots_[i])->GetMutable<LoDTensor>();
  }
}

void DataFeed::CopyToFeedTensor(void* dst, const void* src, size_t size) {
  if (platform::is_cpu_place(this->place_)) {
    memcpy(dst, src, size);
  } else {
#ifdef PADDLE_WITH_CUDA
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
#elif defined(PADDLE_WITH_HIP)
    hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
#elif defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU)
//    xpu_memcpy(dst, src, size, XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    platform::MemcpySyncH2D(dst, src, size, this->place_);
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "Not supported GPU/ROCM, please compile with option WITH_GPU=ON or "
        "WITH_ROCM=ON."));
#endif
  }
}

template <typename T>
void PrivateQueueDataFeed<T>::SetQueueSize(int queue_size) {
  PADDLE_ENFORCE_GT(
      queue_size,
      0,
      platform::errors::InvalidArgument(
          "Queue size %d is illegal in PrivateQueueDataFeed.", queue_size));
  queue_size_ = queue_size;
  queue_ = paddle::framework::MakeChannel<T>();
  queue_->SetCapacity(queue_size);
}

template <typename T>
bool PrivateQueueDataFeed<T>::Start() {
  VLOG(4) << "entering PrivateQueueDataFeed<T>::Start()";
  CheckSetFileList();
  read_thread_ = std::thread(&PrivateQueueDataFeed::ReadThread, this);
  read_thread_.detach();

  finish_start_ = true;
  return true;
}

template <typename T>
void PrivateQueueDataFeed<T>::ReadThread() {
#ifdef _LINUX
  VLOG(4) << "entering PrivateQueueDataFeed<T>::ReadThread()";
  std::string filename;
  while (PickOneFile(&filename)) {
    int err_no = 0;
    fp_ = fs_open_read(filename, &err_no, pipe_command_, true);
    __fsetlocking(&*fp_, FSETLOCKING_BYCALLER);
    T instance;
    while (ParseOneInstanceFromPipe(&instance)) {
      queue_->Put(instance);
    }
  }
  queue_->Close();
#endif
}

template <typename T>
int PrivateQueueDataFeed<T>::Next() {
#ifdef _LINUX
  CheckStart();
  int index = 0;
  T ins_vec;
  while (index < default_batch_size_) {
    T instance;
    if (!queue_->Get(instance)) {
      break;
    }
    AddInstanceToInsVec(&ins_vec, instance, index++);
  }
  batch_size_ = index;
  if (batch_size_ != 0) {
    PutToFeedVec(ins_vec);
  }
  return batch_size_;
#else
  return 0;
#endif
}

// explicit instantiation
template class PrivateQueueDataFeed<std::vector<MultiSlotType>>;

template <typename T>
InMemoryDataFeed<T>::InMemoryDataFeed() {
  this->file_idx_ = nullptr;
  this->mutex_for_pick_file_ = nullptr;
  this->fp_ = nullptr;
  this->thread_id_ = 0;
  this->thread_num_ = 1;
  this->parse_ins_id_ = false;
  this->parse_uid_ = false;
  this->parse_content_ = false;
  this->parse_logkey_ = false;
  this->enable_pv_merge_ = false;
  this->current_phase_ = 1;  // 1:join ;0:update
  this->input_channel_ = nullptr;
  this->output_channel_ = nullptr;
  this->consume_channel_ = nullptr;
}

template <typename T>
bool InMemoryDataFeed<T>::Start() {
#ifdef _LINUX
  VLOG(4) << "entering InMemoryDataFeed<T>::Start()";
  this->CheckSetFileList();
  if (output_channel_->Size() == 0 && input_channel_->Size() != 0) {
    std::vector<T> data;
    input_channel_->Read(data);
    output_channel_->Write(std::move(data));
  }
#endif
  if (batch_offsets_.size() > 0) {
    VLOG(3) << "batch_size offsets: " << batch_offsets_.size();
    enable_heterps_ = true;
    this->offset_index_ = 0;
  }
  this->finish_start_ = true;
  return true;
}

template <typename T>
int InMemoryDataFeed<T>::Next() {
#ifdef _LINUX
  this->CheckStart();
  if (!enable_heterps_) {
    CHECK(output_channel_ != nullptr);
    CHECK(consume_channel_ != nullptr);
    VLOG(3) << "output_channel_ size=" << output_channel_->Size()
            << ", consume_channel_ size=" << consume_channel_->Size()
            << ", thread_id=" << thread_id_;
    int index = 0;
    T instance;
    std::vector<T> ins_vec;
    ins_vec.reserve(this->default_batch_size_);
    while (index < this->default_batch_size_) {
      if (output_channel_->Size() == 0) {
        break;
      }
      output_channel_->Get(instance);
      ins_vec.push_back(instance);
      ++index;
      consume_channel_->Put(std::move(instance));
    }
    this->batch_size_ = index;
    VLOG(3) << "batch_size_=" << this->batch_size_
            << ", thread_id=" << thread_id_;
    if (this->batch_size_ != 0) {
      PutToFeedVec(ins_vec);
    } else {
      VLOG(3) << "finish reading, output_channel_ size="
              << output_channel_->Size()
              << ", consume_channel_ size=" << consume_channel_->Size()
              << ", thread_id=" << thread_id_;
    }
  } else {
    VLOG(3) << "enable heter next: " << offset_index_
            << " batch_offsets: " << batch_offsets_.size();
    if (offset_index_ >= batch_offsets_.size()) {
      VLOG(3) << "offset_index: " << offset_index_
              << " batch_offsets: " << batch_offsets_.size();
      return 0;
    }
    auto& batch = batch_offsets_[offset_index_++];
    this->batch_size_ = batch.second;
    VLOG(3) << "batch_size_=" << this->batch_size_
            << ", thread_id=" << thread_id_;
    if (this->batch_size_ != 0) {
      PutToFeedVec(&records_[batch.first], this->batch_size_);
    } else {
      VLOG(3) << "finish reading for heterps, batch size zero, thread_id="
              << thread_id_;
    }
    VLOG(3) << "enable heter next: " << offset_index_
            << " batch_offsets: " << batch_offsets_.size()
            << " baych_size: " << this->batch_size_;
  }
  return this->batch_size_;
#else
  return 0;
#endif
}

template <typename T>
void InMemoryDataFeed<T>::SetInputChannel(void* channel) {
  input_channel_ = static_cast<paddle::framework::ChannelObject<T>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetOutputChannel(void* channel) {
  output_channel_ = static_cast<paddle::framework::ChannelObject<T>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetConsumeChannel(void* channel) {
  consume_channel_ = static_cast<paddle::framework::ChannelObject<T>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetInputPvChannel(void* channel) {
  input_pv_channel_ =
      static_cast<paddle::framework::ChannelObject<PvInstance>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetOutputPvChannel(void* channel) {
  output_pv_channel_ =
      static_cast<paddle::framework::ChannelObject<PvInstance>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetConsumePvChannel(void* channel) {
  consume_pv_channel_ =
      static_cast<paddle::framework::ChannelObject<PvInstance>*>(channel);
}

template <typename T>
void InMemoryDataFeed<T>::SetThreadId(int thread_id) {
  thread_id_ = thread_id;
}

template <typename T>
void InMemoryDataFeed<T>::SetThreadNum(int thread_num) {
  thread_num_ = thread_num;
}

template <typename T>
void InMemoryDataFeed<T>::SetParseContent(bool parse_content) {
  parse_content_ = parse_content;
}

template <typename T>
void InMemoryDataFeed<T>::SetParseLogKey(bool parse_logkey) {
  parse_logkey_ = parse_logkey;
}

template <typename T>
void InMemoryDataFeed<T>::SetEnablePvMerge(bool enable_pv_merge) {
  enable_pv_merge_ = enable_pv_merge;
}

template <typename T>
void InMemoryDataFeed<T>::SetCurrentPhase(int current_phase) {
  current_phase_ = current_phase;
}

template <typename T>
void InMemoryDataFeed<T>::SetParseInsId(bool parse_ins_id) {
  parse_ins_id_ = parse_ins_id;
}

template <typename T>
void InMemoryDataFeed<T>::SetParseUid(bool parse_uid) {
  parse_uid_ = parse_uid;
}

template <typename T>
void InMemoryDataFeed<T>::LoadIntoMemory() {
#ifdef _LINUX
  if (!so_parser_name_.empty()) {
    LoadIntoMemoryFromSo();
    return;
  }
  VLOG(3) << "LoadIntoMemory() begin, thread_id=" << thread_id_;
  std::string filename;
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
#ifdef PADDLE_WITH_BOX_PS
    if (BoxWrapper::GetInstance()->UseAfsApi()) {
      this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(
          filename, this->pipe_command_);
    } else {
#endif
      int err_no = 0;
      this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_, true);
#ifdef PADDLE_WITH_BOX_PS
    }
#endif
    CHECK(this->fp_ != nullptr);
    __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
    paddle::framework::ChannelWriter<T> writer(input_channel_);
    T instance;
    platform::Timer timeline;
    timeline.Start();
    while (ParseOneInstanceFromPipe(&instance)) {
      writer << std::move(instance);
      instance = T();
    }
    STAT_ADD(STAT_total_feasign_num_in_mem, fea_num_);
    {
      std::lock_guard<std::mutex> flock(*mutex_for_fea_num_);
      *total_fea_num_ += fea_num_;
      fea_num_ = 0;
    }
    writer.Flush();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemory() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_;
  }
  VLOG(3) << "LoadIntoMemory() end, thread_id=" << thread_id_;
#endif
}

template <typename T>
void InMemoryDataFeed<T>::LoadIntoMemoryFromSo() {
#if (defined _LINUX) && (defined PADDLE_WITH_HETERPS) && \
    (defined PADDLE_WITH_PSLIB)
  VLOG(3) << "LoadIntoMemoryFromSo() begin, thread_id=" << thread_id_;
  int buf_len = 1024 * 1024 * 10;
  char* buf = reinterpret_cast<char*>(malloc(buf_len + 10));
  auto ps_gpu_ptr = PSGPUWrapper::GetInstance();

  paddle::framework::CustomParser* parser =
      global_dlmanager_pool().Load(so_parser_name_, slot_conf_);

  std::string filename;
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    platform::Timer timeline;
    timeline.Start();
    if (ps_gpu_ptr->UseAfsApi()) {
      auto afs_reader = ps_gpu_ptr->OpenReader(filename);
      int read_len = 0;
      char* cursor = buf;
      int remain = 0;
      while ((read_len = afs_reader->read(cursor, buf_len - remain)) > 0) {
        std::vector<T> instances;
        read_len += remain;
        remain = ParseInstanceFromSo(read_len, buf, &instances, parser);
        input_channel_->Write(std::move(instances));
        instances = std::vector<T>();
        if (remain) {
          memmove(buf, buf + read_len - remain, remain);
        }
        cursor = buf + remain;
      }
    } else {
      VLOG(0) << "Should Call InitAfsApi First";
    }

    timeline.Pause();
    VLOG(3) << "LoadIntoMemoryFromSo() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_;
  }
  free(buf);
  VLOG(3) << "LoadIntoMemoryFromSo() end, thread_id=" << thread_id_;
#endif
}

// explicit instantiation
template class InMemoryDataFeed<Record>;

void MultiSlotDataFeed::Init(
    const paddle::framework::DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;

  PADDLE_ENFORCE_EQ(
      data_feed_desc.has_multi_slot_desc(),
      true,
      platform::errors::PreconditionNotMet(
          "Multi_slot_desc has not been set in MultiSlotDataFeed."));
  paddle::framework::MultiSlotDesc multi_slot_desc =
      data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch_size());
  // temporarily set queue size = batch size * 100
  SetQueueSize(data_feed_desc.batch_size() * 100);
  size_t all_slot_num = multi_slot_desc.slots_size();
  all_slots_.resize(all_slot_num);
  all_slots_type_.resize(all_slot_num);
  use_slots_index_.resize(all_slot_num);
  total_dims_without_inductive_.resize(all_slot_num);
  inductive_shape_index_.resize(all_slot_num);
  use_slots_.clear();
  use_slots_is_dense_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();
    all_slots_type_[i] = slot.type();
    use_slots_index_[i] = slot.is_used() ? use_slots_.size() : -1;
    total_dims_without_inductive_[i] = 1;
    inductive_shape_index_[i] = -1;
    if (slot.is_used()) {
      use_slots_.push_back(all_slots_[i]);
      use_slots_is_dense_.push_back(slot.is_dense());
      std::vector<int> local_shape;
      if (slot.is_dense()) {
        for (int j = 0; j < slot.shape_size(); ++j) {
          if (slot.shape(j) > 0) {
            total_dims_without_inductive_[i] *= slot.shape(j);
          }
          if (slot.shape(j) == -1) {
            inductive_shape_index_[i] = j;
          }
        }
      }
      for (int j = 0; j < slot.shape_size(); ++j) {
        local_shape.push_back(slot.shape(j));
      }
      use_slots_shape_.push_back(local_shape);
    }
  }
  feed_vec_.resize(use_slots_.size());
  pipe_command_ = data_feed_desc.pipe_command();
  finish_init_ = true;
}

void MultiSlotDataFeed::ReadThread() {
#ifdef _LINUX
  VLOG(4) << "entering MultiSlotDataFeed::ReadThread()";
  std::string filename;
  while (PickOneFile(&filename)) {
    int err_no = 0;
    fp_ = fs_open_read(filename, &err_no, pipe_command_, true);
    CHECK(fp_ != nullptr);
    __fsetlocking(&*fp_, FSETLOCKING_BYCALLER);
    std::vector<MultiSlotType> instance;
    int ins_num = 0;
    while (ParseOneInstanceFromPipe(&instance)) {
      ins_num++;
      queue_->Put(instance);
    }
    VLOG(3) << "filename: " << filename << " inst num: " << ins_num;
  }
  queue_->Close();
#endif
}

bool MultiSlotDataFeed::CheckFile(const char* filename) {
#ifdef _LINUX
  CheckInit();  // get info of slots
  std::ifstream fin(filename);
  if (!fin.good()) {
    VLOG(1) << "error: open file<" << filename << "> fail";
    return false;
  }
  std::string line;
  int instance_cout = 0;
  std::string all_slots_alias = "";
  for (const auto& alias : all_slots_) {
    all_slots_alias += alias + " ";
  }
  std::string use_slots_alias = "";
  for (const auto& alias : use_slots_) {
    use_slots_alias += alias + " ";
  }
  VLOG(3) << "total slots num: " << all_slots_.size();
  VLOG(3) << "total slots alias: " << all_slots_alias;
  VLOG(3) << "used slots num: " << use_slots_.size();
  VLOG(3) << "used slots alias: " << use_slots_alias;
  while (getline(fin, line)) {
    ++instance_cout;
    const char* str = line.c_str();
    char* endptr = const_cast<char*>(str);
    int len = line.length();
    for (size_t i = 0; i < all_slots_.size(); ++i) {
      auto num = strtol(endptr, &endptr, 10);
      if (num < 0) {
        VLOG(0) << "error: the number of ids is a negative number: " << num;
        VLOG(0) << "please check line<" << instance_cout << "> in file<"
                << filename << ">";
        VLOG(0) << "Error occurred when parsing " << i
                << " th slot with total slots number: " << all_slots_.size();
        return false;
      } else if (num == 0) {
        VLOG(0)
            << "error: the number of ids can not be zero, you need "
               "padding it in data generator; or if there is something wrong"
               " with the data, please check if the data contains unresolvable "
               "characters.";
        VLOG(0) << "please check line<" << instance_cout << "> in file<"
                << filename << ">";
        VLOG(0) << "Error occurred when parsing " << i
                << " th slot with total slots number: " << all_slots_.size();
        return false;
      } else if (errno == ERANGE || num > INT_MAX) {
        VLOG(0) << "error: the number of ids greater than INT_MAX";
        VLOG(0) << "please check line<" << instance_cout << "> in file<"
                << filename << ">";
        VLOG(0) << "Error occurred when parsing " << i
                << " th slot with total slots number: " << all_slots_.size();
        return false;
      }
      if (all_slots_type_[i] == "float") {
        for (int j = 0; j < num; ++j) {
          strtof(endptr, &endptr);
          if (errno == ERANGE) {
            VLOG(0) << "error: the value is out of the range of "
                       "representable values for float";
            VLOG(0) << "please check line<" << instance_cout << "> in file<"
                    << filename << ">";
            VLOG(0) << "Error occurred when parsing " << i
                    << " th slot with total slots number: "
                    << all_slots_.size();
            VLOG(0) << "and in this slot: " << j
                    << " th id with total id number: " << num;
            return false;
          }
          if (j + 1 != num && endptr - str == len) {
            VLOG(0) << "error: there is a wrong with the number of ids.";
            VLOG(0) << "Error occurred when parsing " << i
                    << " th slot with total slots number: "
                    << all_slots_.size();
            VLOG(0) << "and in this slot: " << j
                    << " th id with total id number: " << num;
            VLOG(0) << "please check line<" << instance_cout << "> in file<"
                    << filename << ">";
            return false;
          }
        }
      } else if (all_slots_type_[i] == "uint64") {
        for (int j = 0; j < num; ++j) {
          strtoull(endptr, &endptr, 10);
          if (errno == ERANGE) {
            VLOG(0) << "error: the value is out of the range of "
                       "representable values for uint64_t";
            VLOG(0) << "Error occurred when parsing " << i
                    << " th slot with total slots number: "
                    << all_slots_.size();
            VLOG(0) << "and in this slot: " << j
                    << " th id with total id number: " << num;
            VLOG(0) << "please check line<" << instance_cout << "> in file<"
                    << filename << ">";
            return false;
          }
          if (j + 1 != num && endptr - str == len) {
            VLOG(0) << "error: there is a wrong with the number of ids.";
            VLOG(0) << "Error occurred when parsing " << i
                    << " th slot with total slots number: "
                    << all_slots_.size();
            VLOG(0) << "and in this slot: " << j
                    << " th id with total id number: " << num;
            VLOG(0) << "please check line<" << instance_cout << "> in file<"
                    << filename << ">";
            return false;
          }
        }
      } else {
        VLOG(0) << "error: this type<" << all_slots_type_[i]
                << "> is not supported";
        return false;
      }
    }
    // It may be added '\t' character to the end of the output of reduce
    // task when processes data by Hadoop(when the output of the reduce
    // task of Hadoop has only one field, it will add a '\t' at the end
    // of the line by default, and you can use this option to avoid it:
    // `-D mapred.textoutputformat.ignoreseparator=true`), which does
    // not affect the correctness of the data. Therefore, it should be
    // judged that the data is not normal when the end of each line of
    // data contains characters which are not spaces.
    while (endptr - str != len) {
      if (!isspace(*(endptr++))) {
        VLOG(0)
            << "error: there is some extra characters at the end of the line.";
        VLOG(0) << "please check line<" << instance_cout << "> in file<"
                << filename << ">";
        return false;
      }
    }
  }
  VLOG(3) << "instances cout: " << instance_cout;
  VLOG(3) << "The file format is correct";
#endif
  return true;
}

bool MultiSlotDataFeed::ParseOneInstanceFromPipe(
    std::vector<MultiSlotType>* instance) {
#ifdef _LINUX
  thread_local string::LineFileReader reader;

  if (!reader.getline(&*(fp_.get()))) {
    return false;
  } else {
    int use_slots_num = use_slots_.size();
    instance->resize(use_slots_num);
    const char* str = reader.get();
    std::string line = std::string(str);

    char* endptr = const_cast<char*>(str);
    int pos = 0;
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      int num = strtol(&str[pos], &endptr, 10);

      if (num <= 0) {
        std::stringstream ss;
        ss << "\n\nGot unexpected input, maybe something wrong with it.\n";
        ss << "\n----------------------\n";
        ss << "The Origin Input Data:\n";
        ss << "----------------------\n";

        ss << line << "\n";

        ss << "\n----------------------\n";
        ss << "Some Possible Errors:\n";
        ss << "----------------------\n";
        ss << "1. The number of ids can not be zero, you need padding.\n";
        ss << "2. The input data contains unresolvable characters.\n";
        ss << "3. We detect the slot " << i << "'s feasign number is " << num
           << " which is illegal.\n";
        ss << "\n";

        PADDLE_THROW(platform::errors::InvalidArgument(ss.str()));
      }

      if (idx != -1) {
        (*instance)[idx].Init(all_slots_type_[i]);
        if ((*instance)[idx].GetType()[0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            (*instance)[idx].AddValue(feasign);
          }
        } else if ((*instance)[idx].GetType()[0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            (*instance)[idx].AddValue(feasign);
          }
        }
        pos = endptr - str;
      } else {
        for (int j = 0; j <= num; ++j) {
          // pos = line.find_first_of(' ', pos + 1);
          while (line[pos + 1] != ' ') {
            pos++;
          }
        }
      }
    }
    return true;
  }
#else
  return true;
#endif
}

bool MultiSlotDataFeed::ParseOneInstance(std::vector<MultiSlotType>* instance) {
#ifdef _LINUX
  std::string line;
  if (getline(file_, line)) {
    int use_slots_num = use_slots_.size();
    instance->resize(use_slots_num);
    // parse line
    const char* str = line.c_str();
    char* endptr = const_cast<char*>(str);
    int pos = 0;
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      int num = strtol(&str[pos], &endptr, 10);
      PADDLE_ENFORCE_NE(
          num,
          0,
          platform::errors::InvalidArgument(
              "The number of ids can not be zero, you need padding "
              "it in data generator; or if there is something wrong with "
              "the data, please check if the data contains unresolvable "
              "characters.\nplease check this error line: %s, \n Specifically, "
              "something wrong happened(the length of this slot's feasign is 0)"
              "when we parse the %d th slots."
              "Maybe something wrong around this slot"
              "\nWe detect the feasign number of this slot is %d, "
              "which is illegal.",
              str,
              i,
              num));

      if (idx != -1) {
        (*instance)[idx].Init(all_slots_type_[i]);
        if ((*instance)[idx].GetType()[0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            (*instance)[idx].AddValue(feasign);
          }
        } else if ((*instance)[idx].GetType()[0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            (*instance)[idx].AddValue(feasign);
          }
        }
        pos = endptr - str;
      } else {
        for (int j = 0; j <= num; ++j) {
          pos = line.find_first_of(' ', pos + 1);
        }
      }
    }
  } else {
    return false;
  }
#endif
  return false;
}

void MultiSlotDataFeed::AddInstanceToInsVec(
    std::vector<MultiSlotType>* ins_vec,
    const std::vector<MultiSlotType>& instance,
    int index) {
#ifdef _LINUX
  if (index == 0) {
    ins_vec->resize(instance.size());
    for (size_t i = 0; i < instance.size(); ++i) {
      (*ins_vec)[i].Init(instance[i].GetType());
      (*ins_vec)[i].InitOffset();
    }
  }

  for (size_t i = 0; i < instance.size(); ++i) {
    (*ins_vec)[i].AddIns(instance[i]);
  }
#endif
}

void MultiSlotDataFeed::PutToFeedVec(
    const std::vector<MultiSlotType>& ins_vec) {
#ifdef _LINUX
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (feed_vec_[i] == nullptr) {
      continue;
    }
    VLOG(4) << "MultiSlotDataFeed::PutToFeedVec i: " << i;
    const auto& type = ins_vec[i].GetType();
    const auto& offset = ins_vec[i].GetOffset();
    int total_instance = static_cast<int>(offset.back());
    VLOG(4) << "total_instance: " << total_instance;
    // platform::CPUPlace()
    VLOG(4) << "this->place_: " << this->place_;
    if (type[0] == 'f') {  // float
      const auto& feasign = ins_vec[i].GetFloatData();
      float* tensor_ptr =
          feed_vec_[i]->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, &feasign[0], total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      const auto& feasign = ins_vec[i].GetUint64Data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      CopyToFeedTensor(
          tensor_ptr, &feasign[0], total_instance * sizeof(int64_t));
    }

    if (!use_slots_is_dense_[i]) {
      LoD data_lod{offset};
      feed_vec_[i]->set_lod(data_lod);
    }
    if (use_slots_is_dense_[i]) {
      if (inductive_shape_index_[i] != -1) {
        use_slots_shape_[i][inductive_shape_index_[i]] =
            total_instance / total_dims_without_inductive_[i];
      }
      feed_vec_[i]->Resize(phi::make_ddim(use_slots_shape_[i]));
    }
  }
#endif
}

void MultiSlotInMemoryDataFeed::Init(
    const paddle::framework::DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;

  PADDLE_ENFORCE_EQ(
      data_feed_desc.has_multi_slot_desc(),
      true,
      platform::errors::PreconditionNotMet(
          "Multi_slot_desc has not been set in MultiSlotInMemoryDataFeed."));
  paddle::framework::MultiSlotDesc multi_slot_desc =
      data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch_size());
  size_t all_slot_num = multi_slot_desc.slots_size();
  all_slots_.resize(all_slot_num);
  all_slots_type_.resize(all_slot_num);
  use_slots_index_.resize(all_slot_num);
  total_dims_without_inductive_.resize(all_slot_num);
  inductive_shape_index_.resize(all_slot_num);
  use_slots_.clear();
  use_slots_is_dense_.clear();
  slot_conf_.resize(all_slot_num);
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();
    all_slots_type_[i] = slot.type();
    use_slots_index_[i] = slot.is_used() ? use_slots_.size() : -1;

    slot_conf_[i].name = slot.name();
    slot_conf_[i].type = slot.type();
    slot_conf_[i].use_slots_index = use_slots_index_[i];

    total_dims_without_inductive_[i] = 1;
    inductive_shape_index_[i] = -1;
    if (slot.is_used()) {
      use_slots_.push_back(all_slots_[i]);
      use_slots_is_dense_.push_back(slot.is_dense());
      slot_conf_[i].use_slots_is_dense = slot.is_dense();
      std::vector<int> local_shape;
      if (slot.is_dense()) {
        for (int j = 0; j < slot.shape_size(); ++j) {
          if (slot.shape(j) > 0) {
            total_dims_without_inductive_[i] *= slot.shape(j);
          }
          if (slot.shape(j) == -1) {
            inductive_shape_index_[i] = j;
          }
        }
      }
      for (int j = 0; j < slot.shape_size(); ++j) {
        local_shape.push_back(slot.shape(j));
      }
      use_slots_shape_.push_back(local_shape);
    }
  }
  uid_slot_ = multi_slot_desc.uid_slot();
  feed_vec_.resize(use_slots_.size());
  const int kEstimatedFeasignNumPerSlot = 5;  // Magic Number
  for (size_t i = 0; i < all_slot_num; i++) {
    batch_float_feasigns_.push_back(std::vector<float>());
    batch_uint64_feasigns_.push_back(std::vector<uint64_t>());
    batch_float_feasigns_[i].reserve(default_batch_size_ *
                                     kEstimatedFeasignNumPerSlot);
    batch_uint64_feasigns_[i].reserve(default_batch_size_ *
                                      kEstimatedFeasignNumPerSlot);
    offset_.push_back(std::vector<size_t>());
    offset_[i].reserve(default_batch_size_ +
                       1);  // Each lod info will prepend a zero
  }
  visit_.resize(all_slot_num, false);
  pipe_command_ = data_feed_desc.pipe_command();
  so_parser_name_ = data_feed_desc.so_parser_name();
  finish_init_ = true;
  input_type_ = data_feed_desc.input_type();
}

void MultiSlotInMemoryDataFeed::GetMsgFromLogKey(const std::string& log_key,
                                                 uint64_t* search_id,
                                                 uint32_t* cmatch,
                                                 uint32_t* rank) {
  std::string searchid_str = log_key.substr(16, 16);
  *search_id = (uint64_t)strtoull(searchid_str.c_str(), NULL, 16);

  std::string cmatch_str = log_key.substr(11, 3);
  *cmatch = (uint32_t)strtoul(cmatch_str.c_str(), NULL, 16);

  std::string rank_str = log_key.substr(14, 2);
  *rank = (uint32_t)strtoul(rank_str.c_str(), NULL, 16);
}

int MultiSlotInMemoryDataFeed::ParseInstanceFromSo(
    int len,
    const char* str,
    std::vector<Record>* instances,
    CustomParser* parser) {
  // VLOG(0) << "parser: " << parser;
  return parser->ParseInstance(len, str, instances);
}

bool MultiSlotInMemoryDataFeed::ParseOneInstanceFromPipe(Record* instance) {
#ifdef _LINUX
  thread_local string::LineFileReader reader;

  if (!reader.getline(&*(fp_.get()))) {
    return false;
  } else {
    const char* str = reader.get();
    std::string line = std::string(str);
    // VLOG(3) << line;
    char* endptr = const_cast<char*>(str);
    int pos = 0;
    if (parse_ins_id_) {
      int num = strtol(&str[pos], &endptr, 10);
      CHECK(num == 1);  // NOLINT
      pos = endptr - str + 1;
      size_t len = 0;
      while (str[pos + len] != ' ') {
        ++len;
      }
      instance->ins_id_ = std::string(str + pos, len);
      pos += len + 1;
      VLOG(3) << "ins_id " << instance->ins_id_;
    }
    if (parse_content_) {
      int num = strtol(&str[pos], &endptr, 10);
      CHECK(num == 1);  // NOLINT
      pos = endptr - str + 1;
      size_t len = 0;
      while (str[pos + len] != ' ') {
        ++len;
      }
      instance->content_ = std::string(str + pos, len);
      pos += len + 1;
      VLOG(3) << "content " << instance->content_;
    }
    if (parse_logkey_) {
      int num = strtol(&str[pos], &endptr, 10);
      CHECK(num == 1);  // NOLINT
      pos = endptr - str + 1;
      size_t len = 0;
      while (str[pos + len] != ' ') {
        ++len;
      }
      // parse_logkey
      std::string log_key = std::string(str + pos, len);
      uint64_t search_id;
      uint32_t cmatch;
      uint32_t rank;
      GetMsgFromLogKey(log_key, &search_id, &cmatch, &rank);

      instance->ins_id_ = log_key;
      instance->search_id = search_id;
      instance->cmatch = cmatch;
      instance->rank = rank;
      pos += len + 1;
    }
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      int num = strtol(&str[pos], &endptr, 10);
      PADDLE_ENFORCE_NE(
          num,
          0,
          platform::errors::InvalidArgument(
              "The number of ids can not be zero, you need padding "
              "it in data generator; or if there is something wrong with "
              "the data, please check if the data contains unresolvable "
              "characters.\nplease check this error line: %s, \n Specifically, "
              "something wrong happened(the length of this slot's feasign is 0)"
              "when we parse the %d th slots."
              "Maybe something wrong around this slot"
              "\nWe detect the feasign number of this slot is %d, "
              "which is illegal.",
              str,
              i,
              num));
#ifdef PADDLE_WITH_PSLIB
      if (parse_uid_ && all_slots_[i] == uid_slot_) {
        PADDLE_ENFORCE(num == 1 && all_slots_type_[i][0] == 'u',
                       platform::errors::PreconditionNotMet(
                           "The uid has to be uint64 and single.\n"
                           "please check this error line: %s",
                           str));

        char* uidptr = endptr;
        uint64_t feasign = (uint64_t)strtoull(uidptr, &uidptr, 10);
        instance->uid_ = feasign;
      }
#endif
      if (idx != -1) {
        if (all_slots_type_[i][0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            // if float feasign is equal to zero, ignore it
            // except when slot is dense
            if (fabs(feasign) < 1e-6 && !use_slots_is_dense_[i]) {
              continue;
            }
            FeatureFeasign f;
            f.float_feasign_ = feasign;
            instance->float_feasigns_.push_back(FeatureItem(f, idx));
          }
        } else if (all_slots_type_[i][0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            // if uint64 feasign is equal to zero, ignore it
            // except when slot is dense
            if (feasign == 0 && !use_slots_is_dense_[i]) {
              continue;
            }
            FeatureFeasign f;
            f.uint64_feasign_ = feasign;
            instance->uint64_feasigns_.push_back(FeatureItem(f, idx));
          }
        }
        pos = endptr - str;
      } else {
        for (int j = 0; j <= num; ++j) {
          // pos = line.find_first_of(' ', pos + 1);
          while (line[pos + 1] != ' ') {
            pos++;
          }
        }
      }
    }
    instance->float_feasigns_.shrink_to_fit();
    instance->uint64_feasigns_.shrink_to_fit();
    fea_num_ += instance->uint64_feasigns_.size();
    return true;
  }
#else
  return false;
#endif
}

bool MultiSlotInMemoryDataFeed::ParseOneInstance(Record* instance) {
#ifdef _LINUX
  std::string line;
  if (getline(file_, line)) {
    VLOG(3) << line;
    // parse line
    const char* str = line.c_str();
    char* endptr = const_cast<char*>(str);
    int pos = 0;
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      int num = strtol(&str[pos], &endptr, 10);
      PADDLE_ENFORCE_NE(
          num,
          0,
          platform::errors::InvalidArgument(
              "The number of ids can not be zero, you need padding "
              "it in data generator; or if there is something wrong with "
              "the data, please check if the data contains unresolvable "
              "characters.\nplease check this error line: %s, \n Specifically, "
              "something wrong happened(the length of this slot's feasign is 0)"
              "when we parse the %d th slots."
              "Maybe something wrong around this slot"
              "\nWe detect the feasign number of this slot is %d, "
              "which is illegal.",
              str,
              i,
              num));

      if (idx != -1) {
        if (all_slots_type_[i][0] == 'f') {  // float
          for (int j = 0; j < num; ++j) {
            float feasign = strtof(endptr, &endptr);
            if (fabs(feasign) < 1e-6) {
              continue;
            }
            FeatureFeasign f;
            f.float_feasign_ = feasign;
            instance->float_feasigns_.push_back(FeatureItem(f, idx));
          }
        } else if (all_slots_type_[i][0] == 'u') {  // uint64
          for (int j = 0; j < num; ++j) {
            uint64_t feasign = (uint64_t)strtoull(endptr, &endptr, 10);
            if (feasign == 0) {
              continue;
            }
            FeatureFeasign f;
            f.uint64_feasign_ = feasign;
            instance->uint64_feasigns_.push_back(FeatureItem(f, idx));
          }
        }
        pos = endptr - str;
      } else {
        for (int j = 0; j <= num; ++j) {
          pos = line.find_first_of(' ', pos + 1);
        }
      }
    }
    instance->float_feasigns_.shrink_to_fit();
    instance->uint64_feasigns_.shrink_to_fit();
    return true;
  } else {
    return false;
  }
#endif
  return false;
}

void MultiSlotInMemoryDataFeed::PutToFeedVec(const Record* ins_vec, int num) {
#ifdef _LINUX
  for (size_t i = 0; i < batch_float_feasigns_.size(); ++i) {
    batch_float_feasigns_[i].clear();
    batch_uint64_feasigns_[i].clear();
    offset_[i].clear();
    offset_[i].push_back(0);
  }
  ins_content_vec_.clear();
  ins_content_vec_.reserve(num);
  ins_id_vec_.clear();
  ins_id_vec_.reserve(num);
  for (int i = 0; i < num; ++i) {
    auto& r = ins_vec[i];
    ins_id_vec_.push_back(r.ins_id_);
    ins_content_vec_.push_back(r.content_);
    for (auto& item : r.float_feasigns_) {
      batch_float_feasigns_[item.slot()].push_back(item.sign().float_feasign_);
      visit_[item.slot()] = true;
    }
    for (auto& item : r.uint64_feasigns_) {
      batch_uint64_feasigns_[item.slot()].push_back(
          item.sign().uint64_feasign_);
      visit_[item.slot()] = true;
    }
    for (size_t j = 0; j < use_slots_.size(); ++j) {
      const auto& type = all_slots_type_[j];
      if (visit_[j]) {
        visit_[j] = false;
      } else {
        // fill slot value with default value 0
        if (type[0] == 'f') {  // float
          batch_float_feasigns_[j].push_back(0.0);
        } else if (type[0] == 'u') {  // uint64
          batch_uint64_feasigns_[j].push_back(0);
        }
      }
      // get offset of this ins in this slot
      if (type[0] == 'f') {  // float
        offset_[j].push_back(batch_float_feasigns_[j].size());
      } else if (type[0] == 'u') {  // uint64
        offset_[j].push_back(batch_uint64_feasigns_[j].size());
      }
    }
  }

  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (feed_vec_[i] == nullptr) {
      continue;
    }
    int total_instance = offset_[i].back();
    const auto& type = all_slots_type_[i];
    if (type[0] == 'f') {  // float
      float* feasign = batch_float_feasigns_[i].data();
      float* tensor_ptr =
          feed_vec_[i]->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      uint64_t* feasign = batch_uint64_feasigns_[i].data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(int64_t));
    }
    auto& slot_offset = offset_[i];
    if (this->input_type_ == 0) {
      LoD data_lod{slot_offset};
      feed_vec_[i]->set_lod(data_lod);
    } else if (this->input_type_ == 1) {
      if (!use_slots_is_dense_[i]) {
        std::vector<size_t> tmp_offset;
        PADDLE_ENFORCE_EQ(slot_offset.size(),
                          2,
                          platform::errors::InvalidArgument(
                              "In batch reader, the sparse tensor lod size "
                              "must be 2, but received %d.",
                              slot_offset.size()));
        const auto& max_size = slot_offset[1];
        tmp_offset.reserve(max_size + 1);
        for (unsigned int k = 0; k <= max_size; k++) {
          tmp_offset.emplace_back(k);
        }
        slot_offset = tmp_offset;
        LoD data_lod{slot_offset};
        feed_vec_[i]->set_lod(data_lod);
      }
    }
    if (use_slots_is_dense_[i]) {
      if (inductive_shape_index_[i] != -1) {
        use_slots_shape_[i][inductive_shape_index_[i]] =
            total_instance / total_dims_without_inductive_[i];
      }
      feed_vec_[i]->Resize(phi::make_ddim(use_slots_shape_[i]));
    }
  }
#endif
}

void MultiSlotInMemoryDataFeed::PutToFeedVec(
    const std::vector<Record>& ins_vec) {
#ifdef _LINUX
  for (size_t i = 0; i < batch_float_feasigns_.size(); ++i) {
    batch_float_feasigns_[i].clear();
    batch_uint64_feasigns_[i].clear();
    offset_[i].clear();
    offset_[i].push_back(0);
  }
  ins_content_vec_.clear();
  ins_content_vec_.reserve(ins_vec.size());
  ins_id_vec_.clear();
  ins_id_vec_.reserve(ins_vec.size());
  for (size_t i = 0; i < ins_vec.size(); ++i) {
    auto& r = ins_vec[i];
    ins_id_vec_.push_back(r.ins_id_);
    ins_content_vec_.push_back(r.content_);
    for (auto& item : r.float_feasigns_) {
      batch_float_feasigns_[item.slot()].push_back(item.sign().float_feasign_);
      visit_[item.slot()] = true;
    }
    for (auto& item : r.uint64_feasigns_) {
      batch_uint64_feasigns_[item.slot()].push_back(
          item.sign().uint64_feasign_);
      visit_[item.slot()] = true;
    }
    for (size_t j = 0; j < use_slots_.size(); ++j) {
      const auto& type = all_slots_type_[j];
      if (visit_[j]) {
        visit_[j] = false;
      } else {
        // fill slot value with default value 0
        if (type[0] == 'f') {  // float
          batch_float_feasigns_[j].push_back(0.0);
        } else if (type[0] == 'u') {  // uint64
          batch_uint64_feasigns_[j].push_back(0);
        }
      }
      // get offset of this ins in this slot
      if (type[0] == 'f') {  // float
        offset_[j].push_back(batch_float_feasigns_[j].size());
      } else if (type[0] == 'u') {  // uint64
        offset_[j].push_back(batch_uint64_feasigns_[j].size());
      }
    }
  }

  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (feed_vec_[i] == nullptr) {
      continue;
    }
    int total_instance = offset_[i].back();
    const auto& type = all_slots_type_[i];
    if (type[0] == 'f') {  // float
      float* feasign = batch_float_feasigns_[i].data();
      float* tensor_ptr =
          feed_vec_[i]->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      uint64_t* feasign = batch_uint64_feasigns_[i].data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(int64_t));
    }
    auto& slot_offset = offset_[i];
    if (this->input_type_ == 0) {
      if (!use_slots_is_dense_[i]) {
        LoD data_lod{slot_offset};
        feed_vec_[i]->set_lod(data_lod);
      }
    } else if (this->input_type_ == 1) {
      if (!use_slots_is_dense_[i]) {
        std::vector<size_t> tmp_offset;
        PADDLE_ENFORCE_EQ(slot_offset.size(),
                          2,
                          platform::errors::InvalidArgument(
                              "In batch reader, the sparse tensor lod size "
                              "must be 2, but received %d.",
                              slot_offset.size()));
        const auto& max_size = slot_offset[1];
        tmp_offset.reserve(max_size + 1);
        for (unsigned int k = 0; k <= max_size; k++) {
          tmp_offset.emplace_back(k);
        }
        slot_offset = tmp_offset;
        LoD data_lod{slot_offset};
        feed_vec_[i]->set_lod(data_lod);
      }
    }
    if (use_slots_is_dense_[i]) {
      if (inductive_shape_index_[i] != -1) {
        use_slots_shape_[i][inductive_shape_index_[i]] =
            total_instance / total_dims_without_inductive_[i];
      }
      feed_vec_[i]->Resize(phi::make_ddim(use_slots_shape_[i]));
    }
  }
#endif
}

#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)) && !defined(_WIN32)
template <typename T>
void PrivateInstantDataFeed<T>::PutToFeedVec() {
  for (size_t i = 0; i < use_slots_.size(); ++i) {
    const auto& type = ins_vec_[i].GetType();
    const auto& offset = ins_vec_[i].GetOffset();
    int total_instance = static_cast<int>(offset.back());

    if (type[0] == 'f') {  // float
      const auto& feasign = ins_vec_[i].GetFloatData();
      float* tensor_ptr =
          feed_vec_[i]->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, &feasign[0], total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      const auto& feasign = ins_vec_[i].GetUint64Data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      CopyToFeedTensor(
          tensor_ptr, &feasign[0], total_instance * sizeof(int64_t));
    }

    LoD data_lod{offset};
    feed_vec_[i]->set_lod(data_lod);
    if (use_slots_is_dense_[i]) {
      int64_t total_dims = 1;
      for (const auto e : use_slots_shape_[i]) {
        total_dims *= e;
      }
      PADDLE_ENFORCE_EQ(
          total_dims,
          total_instance,
          platform::errors::InvalidArgument(
              "The actual data size of slot[%s] doesn't match its declaration. "
              "The actual data size of slot is %lld"
              ", and its declaration is %lld.",
              use_slots_[i].c_str(),
              total_dims,
              total_instance));
      feed_vec_[i]->Resize(phi::make_ddim(use_slots_shape_[i]));
    }
  }
}

template <typename T>
int PrivateInstantDataFeed<T>::Next() {
  if (ParseOneMiniBatch()) {
    PutToFeedVec();
    return ins_vec_[0].GetBatchSize();
  }
  Postprocess();

  std::string filename;
  if (!PickOneFile(&filename)) {
    return -1;
  }
  if (!Preprocess(filename)) {
    return -1;
  }

  PADDLE_ENFORCE_EQ(
      true,
      ParseOneMiniBatch(),
      platform::errors::InvalidArgument("Fail to parse mini-batch data."));
  PutToFeedVec();
  return ins_vec_[0].GetBatchSize();
}

template <typename T>
void PrivateInstantDataFeed<T>::Init(const DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;

  PADDLE_ENFORCE_EQ(
      data_feed_desc.has_multi_slot_desc(),
      true,
      platform::errors::PreconditionNotMet(
          "Multi_slot_desc has not been set in PrivateInstantDataFeed."));
  paddle::framework::MultiSlotDesc multi_slot_desc =
      data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch_size());
  size_t all_slot_num = multi_slot_desc.slots_size();
  all_slots_.resize(all_slot_num);
  all_slots_type_.resize(all_slot_num);
  use_slots_index_.resize(all_slot_num);
  multi_inductive_shape_index_.resize(all_slot_num);
  use_slots_.clear();
  use_slots_is_dense_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();
    all_slots_type_[i] = slot.type();
    use_slots_index_[i] = slot.is_used() ? use_slots_.size() : -1;
    if (slot.is_used()) {
      use_slots_.push_back(all_slots_[i]);
      use_slots_is_dense_.push_back(slot.is_dense());
      std::vector<int> local_shape;
      if (slot.is_dense()) {
        for (int j = 0; j < slot.shape_size(); ++j) {
          if (slot.shape(j) == -1) {
            multi_inductive_shape_index_[i].push_back(j);
          }
        }
      }
      for (int j = 0; j < slot.shape_size(); ++j) {
        local_shape.push_back(slot.shape(j));
      }
      use_slots_shape_.push_back(local_shape);
    }
  }
  feed_vec_.resize(use_slots_.size());
  ins_vec_.resize(use_slots_.size());

  finish_init_ = true;
}

template class PrivateInstantDataFeed<std::vector<MultiSlotType>>;

bool MultiSlotFileInstantDataFeed::Preprocess(const std::string& filename) {
  fd_ = open(filename.c_str(), O_RDONLY);
  PADDLE_ENFORCE_NE(
      fd_,
      -1,
      platform::errors::Unavailable(
          "Fail to open file: %s in MultiSlotFileInstantDataFeed.",
          filename.c_str()));

  struct stat sb;
  fstat(fd_, &sb);
  end_ = static_cast<size_t>(sb.st_size);

  buffer_ =
      reinterpret_cast<char*>(mmap(NULL, end_, PROT_READ, MAP_PRIVATE, fd_, 0));
  PADDLE_ENFORCE_NE(
      buffer_,
      MAP_FAILED,
      platform::errors::Unavailable(
          "Memory map failed when create shared memory, error number is %s.",
          strerror(errno)));

  offset_ = 0;
  return true;
}

bool MultiSlotFileInstantDataFeed::Postprocess() {
  if (buffer_ != nullptr) {
    munmap(buffer_, end_);
    buffer_ = nullptr;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
    end_ = 0;
    offset_ = 0;
  }
  return true;
}

bool MultiSlotFileInstantDataFeed::ParseOneMiniBatch() {
  if (offset_ == end_) {
    return false;
  }

  batch_size_ = 0;
  while (batch_size_ < default_batch_size_ && offset_ < end_) {
    for (size_t i = 0; i < use_slots_index_.size(); ++i) {
      int idx = use_slots_index_[i];
      char type = all_slots_type_[i][0];

      uint16_t num = *reinterpret_cast<uint16_t*>(buffer_ + offset_);
      PADDLE_ENFORCE_NE(
          num,
          0,
          platform::errors::InvalidArgument(
              "The number of ids can not be zero, you need padding "
              "it in data generator; or if there is something wrong with "
              "the data, please check if the data contains unresolvable "
              "characters."));
      offset_ += sizeof(uint16_t);

      if (idx != -1) {
        int inductive_size = multi_inductive_shape_index_[i].size();
        if (UNLIKELY(batch_size_ == 0)) {
          ins_vec_[idx].Init(all_slots_type_[i], default_batch_size_ * num);
          ins_vec_[idx].InitOffset(default_batch_size_);
          uint64_t* inductive_shape =
              reinterpret_cast<uint64_t*>(buffer_ + offset_);
          for (int inductive_id = 0; inductive_id < inductive_size;
               ++inductive_id) {
            use_slots_shape_[i][multi_inductive_shape_index_[i][inductive_id]] =
                static_cast<int>(*(inductive_shape + inductive_id));
          }
        }
        num -= inductive_size;
        offset_ += sizeof(uint64_t) * inductive_size;

        if (type == 'f') {
          ins_vec_[idx].AppendValues(
              reinterpret_cast<float*>(buffer_ + offset_), num);
          offset_ += num * sizeof(float);
        } else if (type == 'u') {
          ins_vec_[idx].AppendValues(
              reinterpret_cast<uint64_t*>(buffer_ + offset_), num);
          offset_ += num * sizeof(uint64_t);
        }
      } else {
        if (type == 'f') {
          offset_ += num * sizeof(float);
        } else if (type == 'u') {
          offset_ += num * sizeof(uint64_t);
        }
      }
    }
    ++batch_size_;
    // OPTIMIZE: It is better to insert check codes between instances for format
    // checking
  }

  PADDLE_ENFORCE(batch_size_ == default_batch_size_ || offset_ == end_,
                 platform::errors::InvalidArgument(
                     "The batch size id not equal to default batch size, or "
                     "the offset is not equal to end index."
                     "The batch size is %d, default batcch size is %d, offset "
                     "is %d, end index is %d.",
                     batch_size_,
                     default_batch_size_,
                     offset_,
                     end_));
  return true;
}
#endif

bool PaddleBoxDataFeed::Start() {
#ifdef _LINUX
  int phase = GetCurrentPhase();  // join: 1, update: 0
  this->CheckSetFileList();
  if (enable_pv_merge_ && phase == 1) {
    // join phase : input_pv_channel to output_pv_channel
    if (output_pv_channel_->Size() == 0 && input_pv_channel_->Size() != 0) {
      std::vector<PvInstance> data;
      input_pv_channel_->Read(data);
      output_pv_channel_->Write(std::move(data));
    }
  } else {
    // input_channel to output
    if (output_channel_->Size() == 0 && input_channel_->Size() != 0) {
      std::vector<Record> data;
      input_channel_->Read(data);
      output_channel_->Write(std::move(data));
    }
  }
#endif
  this->finish_start_ = true;
  return true;
}

int PaddleBoxDataFeed::Next() {
#ifdef _LINUX
  int phase = GetCurrentPhase();  // join: 1, update: 0
  this->CheckStart();
  if (enable_pv_merge_ && phase == 1) {
    // join phase : output_pv_channel to consume_pv_channel
    CHECK(output_pv_channel_ != nullptr);
    CHECK(consume_pv_channel_ != nullptr);
    VLOG(3) << "output_pv_channel_ size=" << output_pv_channel_->Size()
            << ", consume_pv_channel_ size=" << consume_pv_channel_->Size()
            << ", thread_id=" << thread_id_;
    int index = 0;
    PvInstance pv_instance;
    std::vector<PvInstance> pv_vec;
    pv_vec.reserve(this->pv_batch_size_);
    while (index < this->pv_batch_size_) {
      if (output_pv_channel_->Size() == 0) {
        break;
      }
      output_pv_channel_->Get(pv_instance);
      pv_vec.push_back(pv_instance);
      ++index;
      consume_pv_channel_->Put(std::move(pv_instance));
    }
    this->batch_size_ = index;
    VLOG(3) << "pv_batch_size_=" << this->batch_size_
            << ", thread_id=" << thread_id_;
    if (this->batch_size_ != 0) {
      PutToFeedVec(pv_vec);
    } else {
      VLOG(3) << "finish reading, output_pv_channel_ size="
              << output_pv_channel_->Size()
              << ", consume_pv_channel_ size=" << consume_pv_channel_->Size()
              << ", thread_id=" << thread_id_;
    }
    return this->batch_size_;
  } else {
    this->batch_size_ = MultiSlotInMemoryDataFeed::Next();
    return this->batch_size_;
  }
#else
  return 0;
#endif
}

void PaddleBoxDataFeed::Init(const DataFeedDesc& data_feed_desc) {
  MultiSlotInMemoryDataFeed::Init(data_feed_desc);
  rank_offset_name_ = data_feed_desc.rank_offset();
  pv_batch_size_ = data_feed_desc.pv_batch_size();
}

void PaddleBoxDataFeed::GetRankOffset(const std::vector<PvInstance>& pv_vec,
                                      int ins_number) {
  int index = 0;
  int max_rank = 3;  // the value is setting
  int row = ins_number;
  int col = max_rank * 2 + 1;
  int pv_num = pv_vec.size();

  std::vector<int> rank_offset_mat(row * col, -1);
  rank_offset_mat.shrink_to_fit();

  for (int i = 0; i < pv_num; i++) {
    auto pv_ins = pv_vec[i];
    int ad_num = pv_ins->ads.size();
    int index_start = index;
    for (int j = 0; j < ad_num; ++j) {
      auto ins = pv_ins->ads[j];
      int rank = -1;
      if ((ins->cmatch == 222 || ins->cmatch == 223) &&
          ins->rank <= static_cast<uint32_t>(max_rank) && ins->rank != 0) {
        rank = ins->rank;
      }

      rank_offset_mat[index * col] = rank;
      if (rank > 0) {
        for (int k = 0; k < ad_num; ++k) {
          auto cur_ins = pv_ins->ads[k];
          int fast_rank = -1;
          if ((cur_ins->cmatch == 222 || cur_ins->cmatch == 223) &&
              cur_ins->rank <= static_cast<uint32_t>(max_rank) &&
              cur_ins->rank != 0) {
            fast_rank = cur_ins->rank;
          }

          if (fast_rank > 0) {
            int m = fast_rank - 1;
            rank_offset_mat[index * col + 2 * m + 1] = cur_ins->rank;
            rank_offset_mat[index * col + 2 * m + 2] = index_start + k;
          }
        }
      }
      index += 1;
    }
  }

  int* rank_offset = rank_offset_mat.data();
  int* tensor_ptr = rank_offset_->mutable_data<int>({row, col}, this->place_);
  CopyToFeedTensor(tensor_ptr, rank_offset, row * col * sizeof(int));
}

void PaddleBoxDataFeed::AssignFeedVar(const Scope& scope) {
  MultiSlotInMemoryDataFeed::AssignFeedVar(scope);
  // set rank offset memory
  int phase = GetCurrentPhase();  // join: 1, update: 0
  if (enable_pv_merge_ && phase == 1) {
    rank_offset_ = scope.FindVar(rank_offset_name_)->GetMutable<LoDTensor>();
  }
}

void PaddleBoxDataFeed::PutToFeedVec(const std::vector<PvInstance>& pv_vec) {
#ifdef _LINUX
  int ins_number = 0;
  std::vector<Record*> ins_vec;
  for (auto& pv : pv_vec) {
    ins_number += pv->ads.size();
    for (auto ins : pv->ads) {
      ins_vec.push_back(ins);
    }
  }
  GetRankOffset(pv_vec, ins_number);
  PutToFeedVec(ins_vec);
#endif
}

int PaddleBoxDataFeed::GetCurrentPhase() {
#ifdef PADDLE_WITH_BOX_PS
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  if (box_ptr->Mode() == 1) {  // For AucRunner
    return 1;
  } else {
    return box_ptr->Phase();
  }
#else
  LOG(WARNING) << "It should be complied with BOX_PS...";
  return current_phase_;
#endif
}

void PaddleBoxDataFeed::PutToFeedVec(const std::vector<Record*>& ins_vec) {
#ifdef _LINUX
  for (size_t i = 0; i < batch_float_feasigns_.size(); ++i) {
    batch_float_feasigns_[i].clear();
    batch_uint64_feasigns_[i].clear();
    offset_[i].clear();
    offset_[i].push_back(0);
  }
  ins_content_vec_.clear();
  ins_content_vec_.reserve(ins_vec.size());
  ins_id_vec_.clear();
  ins_id_vec_.reserve(ins_vec.size());
  for (size_t i = 0; i < ins_vec.size(); ++i) {
    auto r = ins_vec[i];
    ins_id_vec_.push_back(r->ins_id_);
    ins_content_vec_.push_back(r->content_);
    for (auto& item : r->float_feasigns_) {
      batch_float_feasigns_[item.slot()].push_back(item.sign().float_feasign_);
      visit_[item.slot()] = true;
    }
    for (auto& item : r->uint64_feasigns_) {
      batch_uint64_feasigns_[item.slot()].push_back(
          item.sign().uint64_feasign_);
      visit_[item.slot()] = true;
    }
    for (size_t j = 0; j < use_slots_.size(); ++j) {
      const auto& type = all_slots_type_[j];
      if (visit_[j]) {
        visit_[j] = false;
      } else {
        // fill slot value with default value 0
        if (type[0] == 'f') {  // float
          batch_float_feasigns_[j].push_back(0.0);
        } else if (type[0] == 'u') {  // uint64
          batch_uint64_feasigns_[j].push_back(0);
        }
      }
      // get offset of this ins in this slot
      if (type[0] == 'f') {  // float
        offset_[j].push_back(batch_float_feasigns_[j].size());
      } else if (type[0] == 'u') {  // uint64
        offset_[j].push_back(batch_uint64_feasigns_[j].size());
      }
    }
  }

  for (size_t i = 0; i < use_slots_.size(); ++i) {
    if (feed_vec_[i] == nullptr) {
      continue;
    }
    int total_instance = offset_[i].back();
    const auto& type = all_slots_type_[i];
    if (type[0] == 'f') {  // float
      float* feasign = batch_float_feasigns_[i].data();
      float* tensor_ptr =
          feed_vec_[i]->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(float));
    } else if (type[0] == 'u') {  // uint64
      // no uint64_t type in paddlepaddle
      uint64_t* feasign = batch_uint64_feasigns_[i].data();
      int64_t* tensor_ptr = feed_vec_[i]->mutable_data<int64_t>(
          {total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(int64_t));
    }
    auto& slot_offset = offset_[i];
    LoD data_lod{slot_offset};
    feed_vec_[i]->set_lod(data_lod);
    if (use_slots_is_dense_[i]) {
      if (inductive_shape_index_[i] != -1) {
        use_slots_shape_[i][inductive_shape_index_[i]] =
            total_instance / total_dims_without_inductive_[i];
      }
      feed_vec_[i]->Resize(phi::make_ddim(use_slots_shape_[i]));
    }
  }
#endif
}

template class InMemoryDataFeed<SlotRecord>;
void SlotRecordInMemoryDataFeed::Init(const DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;
  PADDLE_ENFORCE(data_feed_desc.has_multi_slot_desc(),
                 platform::errors::PreconditionNotMet(
                     "Multi_slot_desc has not been set in data_feed_desc"));
  paddle::framework::MultiSlotDesc multi_slot_desc =
      data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch_size());
  size_t all_slot_num = multi_slot_desc.slots_size();

  all_slots_.resize(all_slot_num);
  all_slots_info_.resize(all_slot_num);
  used_slots_info_.resize(all_slot_num);
  use_slot_size_ = 0;
  use_slots_.clear();

  float_total_dims_size_ = 0;
  float_total_dims_without_inductives_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();

    AllSlotInfo& all_slot = all_slots_info_[i];
    all_slot.slot = slot.name();
    all_slot.type = slot.type();
    all_slot.used_idx = slot.is_used() ? use_slot_size_ : -1;
    all_slot.slot_value_idx = -1;

    if (slot.is_used()) {
      UsedSlotInfo& info = used_slots_info_[use_slot_size_];
      info.idx = i;
      info.slot = slot.name();
      info.type = slot.type();
      info.dense = slot.is_dense();
      info.total_dims_without_inductive = 1;
      info.inductive_shape_index = -1;

      // record float value and uint64_t value pos
      if (info.type[0] == 'u') {
        info.slot_value_idx = uint64_use_slot_size_;
        all_slot.slot_value_idx = uint64_use_slot_size_;
        ++uint64_use_slot_size_;
      } else if (info.type[0] == 'f') {
        info.slot_value_idx = float_use_slot_size_;
        all_slot.slot_value_idx = float_use_slot_size_;
        ++float_use_slot_size_;
      }

      use_slots_.push_back(slot.name());

      if (slot.is_dense()) {
        for (int j = 0; j < slot.shape_size(); ++j) {
          if (slot.shape(j) > 0) {
            info.total_dims_without_inductive *= slot.shape(j);
          }
          if (slot.shape(j) == -1) {
            info.inductive_shape_index = j;
          }
        }
      }
      if (info.type[0] == 'f') {
        float_total_dims_without_inductives_.push_back(
            info.total_dims_without_inductive);
        float_total_dims_size_ += info.total_dims_without_inductive;
      }
      info.local_shape.clear();
      for (int j = 0; j < slot.shape_size(); ++j) {
        info.local_shape.push_back(slot.shape(j));
      }
      ++use_slot_size_;
    }
  }
  used_slots_info_.resize(use_slot_size_);

  feed_vec_.resize(used_slots_info_.size());
  const int kEstimatedFeasignNumPerSlot = 5;  // Magic Number
  for (size_t i = 0; i < all_slot_num; i++) {
    batch_float_feasigns_.push_back(std::vector<float>());
    batch_uint64_feasigns_.push_back(std::vector<uint64_t>());
    batch_float_feasigns_[i].reserve(default_batch_size_ *
                                     kEstimatedFeasignNumPerSlot);
    batch_uint64_feasigns_[i].reserve(default_batch_size_ *
                                      kEstimatedFeasignNumPerSlot);
    offset_.push_back(std::vector<size_t>());
    offset_[i].reserve(default_batch_size_ +
                       1);  // Each lod info will prepend a zero
  }
  visit_.resize(all_slot_num, false);
  pipe_command_ = data_feed_desc.pipe_command();
  finish_init_ = true;
  input_type_ = data_feed_desc.input_type();
  size_t pos = pipe_command_.find(".so");
  if (pos != std::string::npos) {
    pos = pipe_command_.rfind('|');
    if (pos == std::string::npos) {
      so_parser_name_ = pipe_command_;
      pipe_command_.clear();
    } else {
      so_parser_name_ = pipe_command_.substr(pos + 1);
      pipe_command_ = pipe_command_.substr(0, pos);
    }
    so_parser_name_ = paddle::string::erase_spaces(so_parser_name_);
  } else {
    so_parser_name_.clear();
  }
#if defined(PADDLE_WITH_GPU_GRAPH) && defined(PADDLE_WITH_HETERPS)
  gpu_graph_data_generator_.SetConfig(data_feed_desc);
#endif
}

void SlotRecordInMemoryDataFeed::LoadIntoMemory() {
  VLOG(3) << "SlotRecord LoadIntoMemory() begin, thread_id=" << thread_id_;
  if (!so_parser_name_.empty()) {
    LoadIntoMemoryByLib();
  } else {
    LoadIntoMemoryByCommand();
  }
}
void SlotRecordInMemoryDataFeed::LoadIntoMemoryByLib(void) {
  if (true) {
    // user defined file format analysis
    LoadIntoMemoryByFile();
  } else {
    LoadIntoMemoryByLine();
  }
}

void SlotRecordInMemoryDataFeed::LoadIntoMemoryByFile(void) {
#if (defined _LINUX) && (defined PADDLE_WITH_HETERPS) && \
    (defined PADDLE_WITH_PSLIB)
  paddle::framework::CustomParser* parser =
      global_dlmanager_pool().Load(so_parser_name_, all_slots_info_);
  CHECK(parser != nullptr);
  // get slotrecord object
  auto pull_record_func = [this](std::vector<SlotRecord>& record_vec,
                                 int max_fetch_num,
                                 int offset) {
    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (max_fetch_num > 0) {
        SlotRecordPool().get(&record_vec[0], offset);
      } else {  // free all
        max_fetch_num = static_cast<int>(record_vec.size());
        if (max_fetch_num > offset) {
          SlotRecordPool().put(&record_vec[offset], (max_fetch_num - offset));
        }
      }
    } else if (max_fetch_num > 0) {
      SlotRecordPool().get(&record_vec, max_fetch_num);
    } else {
      SlotRecordPool().put(&record_vec);
    }
  };

  std::string filename;
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    platform::Timer timeline;
    timeline.Start();

    int lines = 0;
    bool is_ok = true;
    auto ps_gpu_ptr = PSGPUWrapper::GetInstance();
    do {
      if (ps_gpu_ptr->UseAfsApi()) {
        auto afs_reader = ps_gpu_ptr->OpenReader(filename);
        is_ok = parser->ParseFileInstance(
            [this, afs_reader](char* buf, int len) {
              return afs_reader->read(buf, len);
            },
            pull_record_func,
            lines);
      } else {
        int err_no = 0;
        this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_, true);

        CHECK(this->fp_ != nullptr);
        __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
        is_ok = parser->ParseFileInstance(
            [this](char* buf, int len) {
              return fread(buf, sizeof(char), len, this->fp_.get());
            },
            pull_record_func,
            lines);

        if (!is_ok) {
          LOG(WARNING) << "parser error, filename=" << filename
                       << ", lines=" << lines;
        }
      }
    } while (!is_ok);
    timeline.Pause();
    VLOG(3) << "LoadIntoMemoryByLib() read all file, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_ << ", lines=" << lines;
  }
#endif
}

void SlotRecordInMemoryDataFeed::LoadIntoMemoryByLine(void) {
#ifdef _LINUX
  paddle::framework::CustomParser* parser =
      global_dlmanager_pool().Load(so_parser_name_, all_slots_info_);
  std::string filename;
  BufferedLineFileReader line_reader;
  line_reader.set_sample_rate(sample_rate_);
  BufferedLineFileReader::LineFunc line_func = nullptr;

  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    std::vector<SlotRecord> record_vec;
    platform::Timer timeline;
    timeline.Start();
    int offset = 0;
    int old_offset = 0;

    SlotRecordPool().get(&record_vec, OBJPOOL_BLOCK_SIZE);
    // get slotrecord object function
    auto record_func = [this, &offset, &record_vec, &old_offset](
                           std::vector<SlotRecord>& vec, int num) {
      vec.resize(num);
      if (offset + num > OBJPOOL_BLOCK_SIZE) {
        input_channel_->WriteMove(offset, &record_vec[0]);
        SlotRecordPool().get(&record_vec[0], offset);
        record_vec.resize(OBJPOOL_BLOCK_SIZE);
        offset = 0;
        old_offset = 0;
      }
      for (int i = 0; i < num; ++i) {
        auto& ins = record_vec[offset + i];
        ins->reset();
        vec[i] = ins;
      }
      offset = offset + num;
    };

    line_func = [this,
                 &parser,
                 &record_vec,
                 &offset,
                 &filename,
                 &record_func,
                 &old_offset](const std::string& line) {
      old_offset = offset;
      if (!parser->ParseOneInstance(line, record_func)) {
        offset = old_offset;
        LOG(WARNING) << "read file:[" << filename << "] item error, line:["
                     << line << "]";
        return false;
      }
      if (offset >= OBJPOOL_BLOCK_SIZE) {
        input_channel_->Write(std::move(record_vec));
        record_vec.clear();
        SlotRecordPool().get(&record_vec, OBJPOOL_BLOCK_SIZE);
        offset = 0;
      }
      return true;
    };

    int lines = 0;

    do {
      int err_no = 0;
      this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_, true);
      CHECK(this->fp_ != nullptr);
      __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
      lines = line_reader.read_file(this->fp_.get(), line_func, lines);
    } while (line_reader.is_error());

    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (offset < OBJPOOL_BLOCK_SIZE) {
        SlotRecordPool().put(&record_vec[offset],
                             (OBJPOOL_BLOCK_SIZE - offset));
      }
    } else {
      SlotRecordPool().put(&record_vec);
    }
    record_vec.clear();
    record_vec.shrink_to_fit();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemoryByLib() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_ << ", lines=" << lines
            << ", sample lines=" << line_reader.get_sample_line()
            << ", filesize=" << line_reader.file_size() / 1024.0 / 1024.0
            << "MB";
  }

  VLOG(3) << "LoadIntoMemoryByLib() end, thread_id=" << thread_id_
          << ", total size: " << line_reader.file_size();
#endif
}

void SlotRecordInMemoryDataFeed::LoadIntoMemoryByCommand(void) {
#ifdef _LINUX
  std::string filename;
  BufferedLineFileReader line_reader;
  line_reader.set_sample_rate(sample_rate_);

  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    int lines = 0;
    std::vector<SlotRecord> record_vec;
    platform::Timer timeline;
    timeline.Start();
    SlotRecordPool().get(&record_vec, OBJPOOL_BLOCK_SIZE);
    int offset = 0;

    do {
      int err_no = 0;
      this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_, true);
      CHECK(this->fp_ != nullptr);
      __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);

      lines = line_reader.read_file(
          this->fp_.get(),
          [this, &record_vec, &offset, &filename](const std::string& line) {
            if (ParseOneInstance(line, &record_vec[offset])) {
              ++offset;
            } else {
              LOG(WARNING) << "read file:[" << filename
                           << "] item error, line:[" << line << "]";
              return false;
            }
            if (offset >= OBJPOOL_BLOCK_SIZE) {
              input_channel_->Write(std::move(record_vec));
              record_vec.clear();
              SlotRecordPool().get(&record_vec, OBJPOOL_BLOCK_SIZE);
              offset = 0;
            }
            return true;
          },
          lines);
    } while (line_reader.is_error());
    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (offset < OBJPOOL_BLOCK_SIZE) {
        SlotRecordPool().put(&record_vec[offset],
                             (OBJPOOL_BLOCK_SIZE - offset));
      }
    } else {
      SlotRecordPool().put(&record_vec);
    }
    record_vec.clear();
    record_vec.shrink_to_fit();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemory() read all lines, file=" << filename
            << ", lines=" << lines
            << ", sample lines=" << line_reader.get_sample_line()
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_;
  }
  VLOG(3) << "LoadIntoMemory() end, thread_id=" << thread_id_
          << ", total size: " << line_reader.file_size();
#endif
}

static void parser_log_key(const std::string& log_key,
                           uint64_t* search_id,
                           uint32_t* cmatch,
                           uint32_t* rank) {
  std::string searchid_str = log_key.substr(16, 16);
  *search_id = static_cast<uint64_t>(strtoull(searchid_str.c_str(), NULL, 16));
  std::string cmatch_str = log_key.substr(11, 3);
  *cmatch = static_cast<uint32_t>(strtoul(cmatch_str.c_str(), NULL, 16));
  std::string rank_str = log_key.substr(14, 2);
  *rank = static_cast<uint32_t>(strtoul(rank_str.c_str(), NULL, 16));
}

bool SlotRecordInMemoryDataFeed::ParseOneInstance(const std::string& line,
                                                  SlotRecord* ins) {
  SlotRecord& rec = (*ins);
  // parse line
  const char* str = line.c_str();
  char* endptr = const_cast<char*>(str);
  int pos = 0;

  thread_local std::vector<std::vector<float>> slot_float_feasigns;
  thread_local std::vector<std::vector<uint64_t>> slot_uint64_feasigns;
  slot_float_feasigns.resize(float_use_slot_size_);
  slot_uint64_feasigns.resize(uint64_use_slot_size_);

  if (parse_ins_id_) {
    int num = strtol(&str[pos], &endptr, 10);
    CHECK(num == 1);  // NOLINT
    pos = endptr - str + 1;
    size_t len = 0;
    while (str[pos + len] != ' ') {
      ++len;
    }
    rec->ins_id_ = std::string(str + pos, len);
    pos += len + 1;
  }
  if (parse_logkey_) {
    int num = strtol(&str[pos], &endptr, 10);
    CHECK(num == 1);  // NOLINT
    pos = endptr - str + 1;
    size_t len = 0;
    while (str[pos + len] != ' ') {
      ++len;
    }
    // parse_logkey
    std::string log_key = std::string(str + pos, len);
    uint64_t search_id;
    uint32_t cmatch;
    uint32_t rank;
    parser_log_key(log_key, &search_id, &cmatch, &rank);

    rec->ins_id_ = log_key;
    rec->search_id = search_id;
    rec->cmatch = cmatch;
    rec->rank = rank;
    pos += len + 1;
  }

  int float_total_slot_num = 0;
  int uint64_total_slot_num = 0;

  for (size_t i = 0; i < all_slots_info_.size(); ++i) {
    auto& info = all_slots_info_[i];
    int num = strtol(&str[pos], &endptr, 10);
    PADDLE_ENFORCE(num,
                   "The number of ids can not be zero, you need padding "
                   "it in data generator; or if there is something wrong with "
                   "the data, please check if the data contains unresolvable "
                   "characters.\nplease check this error line: %s",
                   str);
    if (info.used_idx != -1) {
      if (info.type[0] == 'f') {  // float
        auto& slot_fea = slot_float_feasigns[info.slot_value_idx];
        slot_fea.clear();
        for (int j = 0; j < num; ++j) {
          float feasign = strtof(endptr, &endptr);
          if (fabs(feasign) < 1e-6 && !used_slots_info_[info.used_idx].dense) {
            continue;
          }
          slot_fea.push_back(feasign);
          ++float_total_slot_num;
        }
      } else if (info.type[0] == 'u') {  // uint64
        auto& slot_fea = slot_uint64_feasigns[info.slot_value_idx];
        slot_fea.clear();
        for (int j = 0; j < num; ++j) {
          uint64_t feasign =
              static_cast<uint64_t>(strtoull(endptr, &endptr, 10));
          slot_fea.push_back(feasign);
          ++uint64_total_slot_num;
        }
      }
      pos = endptr - str;
    } else {
      for (int j = 0; j <= num; ++j) {
        // pos = line.find_first_of(' ', pos + 1);
        while (line[pos + 1] != ' ') {
          pos++;
        }
      }
    }
  }
  rec->slot_float_feasigns_.add_slot_feasigns(slot_float_feasigns,
                                              float_total_slot_num);
  rec->slot_uint64_feasigns_.add_slot_feasigns(slot_uint64_feasigns,
                                               uint64_total_slot_num);

  return (uint64_total_slot_num > 0);
}

void SlotRecordInMemoryDataFeed::AssignFeedVar(const Scope& scope) {
  CheckInit();
  for (int i = 0; i < use_slot_size_; ++i) {
    feed_vec_[i] =
        scope.FindVar(used_slots_info_[i].slot)->GetMutable<LoDTensor>();
  }
}

void SlotRecordInMemoryDataFeed::PutToFeedVec(const SlotRecord* ins_vec,
                                              int num) {
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
  paddle::platform::SetDeviceId(place_.GetDeviceId());
  pack_->pack_instance(ins_vec, num);
  BuildSlotBatchGPU(pack_->ins_num());
#else
  for (int j = 0; j < use_slot_size_; ++j) {
    auto& feed = feed_vec_[j];
    if (feed == nullptr) {
      continue;
    }

    auto& slot_offset = offset_[j];
    slot_offset.clear();
    slot_offset.reserve(num + 1);
    slot_offset.push_back(0);

    int total_instance = 0;
    auto& info = used_slots_info_[j];
    // fill slot value with default value 0
    if (info.type[0] == 'f') {  // float
      auto& batch_fea = batch_float_feasigns_[j];
      batch_fea.clear();

      for (int i = 0; i < num; ++i) {
        auto r = ins_vec[i];
        size_t fea_num = 0;
        float* slot_values =
            r->slot_float_feasigns_.get_values(info.slot_value_idx, &fea_num);
        batch_fea.resize(total_instance + fea_num);
        memcpy(
            &batch_fea[total_instance], slot_values, sizeof(float) * fea_num);
        total_instance += fea_num;
        slot_offset.push_back(total_instance);
      }

      float* feasign = batch_fea.data();
      float* tensor_ptr =
          feed->mutable_data<float>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(float));

    } else if (info.type[0] == 'u') {  // uint64
      auto& batch_fea = batch_uint64_feasigns_[j];
      batch_fea.clear();

      for (int i = 0; i < num; ++i) {
        auto r = ins_vec[i];
        size_t fea_num = 0;
        uint64_t* slot_values =
            r->slot_uint64_feasigns_.get_values(info.slot_value_idx, &fea_num);
        if (fea_num > 0) {
          batch_fea.resize(total_instance + fea_num);
          memcpy(&batch_fea[total_instance],
                 slot_values,
                 sizeof(uint64_t) * fea_num);
          total_instance += fea_num;
        }
        if (fea_num == 0) {
          batch_fea.resize(total_instance + fea_num);
          batch_fea[total_instance] = 0;
          total_instance += 1;
        }
        slot_offset.push_back(total_instance);
      }

      // no uint64_t type in paddlepaddle
      uint64_t* feasign = batch_fea.data();
      int64_t* tensor_ptr =
          feed->mutable_data<int64_t>({total_instance, 1}, this->place_);
      CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(int64_t));
    }

    if (info.dense) {
      if (info.inductive_shape_index != -1) {
        info.local_shape[info.inductive_shape_index] =
            total_instance / info.total_dims_without_inductive;
      }
      feed->Resize(phi::make_ddim(info.local_shape));
    } else {
      LoD data_lod{slot_offset};
      feed_vec_[j]->set_lod(data_lod);
    }
  }
#endif
}

void SlotRecordInMemoryDataFeed::ExpandSlotRecord(SlotRecord* rec) {
  SlotRecord& ins = (*rec);
  if (ins->slot_float_feasigns_.slot_offsets.empty()) {
    return;
  }
  size_t total_value_size = ins->slot_float_feasigns_.slot_values.size();
  if (float_total_dims_size_ == total_value_size) {
    return;
  }
  int float_slot_num =
      static_cast<int>(float_total_dims_without_inductives_.size());
  CHECK(float_slot_num == float_use_slot_size_);
  std::vector<float> old_values;
  std::vector<uint32_t> old_offsets;
  old_values.swap(ins->slot_float_feasigns_.slot_values);
  old_offsets.swap(ins->slot_float_feasigns_.slot_offsets);

  ins->slot_float_feasigns_.slot_values.resize(float_total_dims_size_);
  ins->slot_float_feasigns_.slot_offsets.assign(float_slot_num + 1, 0);

  auto& slot_offsets = ins->slot_float_feasigns_.slot_offsets;
  auto& slot_values = ins->slot_float_feasigns_.slot_values;

  uint32_t offset = 0;
  int num = 0;
  uint32_t old_off = 0;
  int dim = 0;

  for (int i = 0; i < float_slot_num; ++i) {
    dim = float_total_dims_without_inductives_[i];
    old_off = old_offsets[i];
    num = static_cast<int>(old_offsets[i + 1] - old_off);
    if (num == 0) {
      // fill slot value with default value 0
      for (int k = 0; k < dim; ++k) {
        slot_values[k + offset] = 0.0;
      }
    } else {
      if (num == dim) {
        memcpy(&slot_values[offset], &old_values[old_off], dim * sizeof(float));
      } else {
        // position fea
        // record position index need fix values
        int pos_idx = static_cast<int>(old_values[old_off]);
        for (int k = 0; k < dim; ++k) {
          if (k == pos_idx) {
            slot_values[k + offset] = 1.0;
          } else {
            slot_values[k + offset] = 0.0;
          }
        }
      }
    }
    slot_offsets[i] = offset;
    offset += dim;
  }
  slot_offsets[float_slot_num] = offset;
  CHECK(float_total_dims_size_ == static_cast<size_t>(offset));
}

bool SlotRecordInMemoryDataFeed::Start() {
  VLOG(4) << "entering SlotRecordInMemoryDataFeed::Start";
#ifdef _LINUX
  this->CheckSetFileList();
  if (input_channel_->Size() != 0) {
    std::vector<SlotRecord> data;
    input_channel_->Read(data);
  }
#endif
  if (batch_offsets_.size() > 0) {
    VLOG(3) << "batch_size offsets: " << batch_offsets_.size();
    enable_heterps_ = true;
    this->offset_index_ = 0;
  }
  this->finish_start_ = true;
#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
  CHECK(paddle::platform::is_gpu_place(this->place_));
  pack_ = BatchGpuPackMgr().get(this->GetPlace(), used_slots_info_);
#endif
#if defined(PADDLE_WITH_GPU_GRAPH) && defined(PADDLE_WITH_HETERPS)
  gpu_graph_data_generator_.AllocResource(this->place_, feed_vec_);
#endif
  return true;
}

int SlotRecordInMemoryDataFeed::Next() {
#ifdef _LINUX
  this->CheckStart();
  if (!gpu_graph_mode_) {
    VLOG(3) << "enable heter next: " << offset_index_
            << " batch_offsets: " << batch_offsets_.size();
    if (offset_index_ >= batch_offsets_.size()) {
      VLOG(3) << "offset_index: " << offset_index_
              << " batch_offsets: " << batch_offsets_.size();
      return 0;
    }
    auto& batch = batch_offsets_[offset_index_++];
    this->batch_size_ = batch.second;
    VLOG(3) << "batch_size_=" << this->batch_size_
            << ", thread_id=" << thread_id_;
    if (this->batch_size_ != 0) {
      PutToFeedVec(&records_[batch.first], this->batch_size_);
    } else {
      VLOG(3) << "finish reading for heterps, batch size zero, thread_id="
              << thread_id_;
    }
    VLOG(3) << "enable heter next: " << offset_index_
            << " batch_offsets: " << batch_offsets_.size()
            << " baych_size: " << this->batch_size_;
  } else {
    VLOG(3) << "datafeed in gpu graph mode";
#if defined(PADDLE_WITH_GPU_GRAPH) && defined(PADDLE_WITH_HETERPS)
    this->batch_size_ = gpu_graph_data_generator_.GenerateBatch();
#endif
  }

  return this->batch_size_;
#else
  return 0;
#endif
}

#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_HETERPS)
void SlotRecordInMemoryDataFeed::BuildSlotBatchGPU(const int ins_num) {
  int offset_cols_size = (ins_num + 1);
  size_t slot_total_num = (use_slot_size_ * offset_cols_size);
  pack_->resize_gpu_slot_offsets(slot_total_num * sizeof(size_t));

  auto& value = pack_->value();
  const UsedSlotGpuType* used_slot_gpu_types =
      static_cast<const UsedSlotGpuType*>(pack_->get_gpu_slots());
  FillSlotValueOffset(ins_num, use_slot_size_, pack_->gpu_slot_offsets(),
                      value.d_uint64_offset.data<int>(), uint64_use_slot_size_,
                      value.d_float_offset.data<int>(), float_use_slot_size_,
                      used_slot_gpu_types);

  size_t* d_slot_offsets = reinterpret_cast<size_t*>(pack_->gpu_slot_offsets());

  auto& offsets = pack_->offsets();
  offsets.resize(slot_total_num);
  auto& h_tensor_ptrs = pack_->h_tensor_ptrs();
  h_tensor_ptrs.resize(use_slot_size_);
  // alloc gpu memory
  pack_->resize_tensor();

  LoDTensor& float_tensor = pack_->float_tensor();
  LoDTensor& uint64_tensor = pack_->uint64_tensor();

  int64_t float_offset = 0;
  int64_t uint64_offset = 0;

  // copy index
  CUDA_CHECK(cudaMemcpy(offsets.data(),
                        d_slot_offsets,
                        slot_total_num * sizeof(size_t),
                        cudaMemcpyDeviceToHost));
  for (int j = 0; j < use_slot_size_; ++j) {
    auto& feed = feed_vec_[j];
    if (feed == nullptr) {
      h_tensor_ptrs[j] = nullptr;
      continue;
    }

    size_t* off_start_ptr = &offsets[j * offset_cols_size];

    int total_instance = static_cast<int>(off_start_ptr[offset_cols_size - 1]);
    CHECK(total_instance >= 0)
        << "slot idx:" << j << ", total instance:" << total_instance;
    auto& info = used_slots_info_[j];

    // fill slot value with default value 0
    if (info.type[0] == 'f') {  // float
      if (total_instance > 0) {
        feed->ShareDataWith(float_tensor.Slice(
            static_cast<int64_t>(float_offset),
            static_cast<int64_t>(float_offset + total_instance)));
        feed->Resize({total_instance, 1});
        float_offset += total_instance;
        h_tensor_ptrs[j] = feed->mutable_data<float>(this->place_);
      } else {
        h_tensor_ptrs[j] =
            feed->mutable_data<float>({total_instance, 1}, this->place_);
      }
    } else if (info.type[0] == 'u') {  // uint64
      if (total_instance > 0) {
        feed->ShareDataWith(uint64_tensor.Slice(
            static_cast<int64_t>(uint64_offset),
            static_cast<int64_t>(uint64_offset + total_instance)));
        feed->Resize({total_instance, 1});
        uint64_offset += total_instance;
        h_tensor_ptrs[j] = feed->mutable_data<int64_t>(this->place_);
      } else {
        h_tensor_ptrs[j] =
            feed->mutable_data<int64_t>({total_instance, 1}, this->place_);
      }
    }

    if (info.dense) {
      if (info.inductive_shape_index != -1) {
        info.local_shape[info.inductive_shape_index] =
            total_instance / info.total_dims_without_inductive;
      }
      feed->Resize(phi::make_ddim(info.local_shape));
    } else {
      LoD& lod = (*feed->mutable_lod());
      lod.resize(1);
      lod[0].resize(offset_cols_size);
      paddle::framework::MixVector<size_t> mixv_lod(&lod[0]);
      memcpy(mixv_lod.MutableData(platform::CPUPlace()),
             off_start_ptr,
             offset_cols_size * sizeof(size_t));
    }
  }
  void** dest_gpu_p = reinterpret_cast<void**>(pack_->slot_buf_ptr());
  CUDA_CHECK(cudaMemcpy(dest_gpu_p,
                        h_tensor_ptrs.data(),
                        use_slot_size_ * sizeof(void*),
                        cudaMemcpyHostToDevice));
  CopyForTensor(
       ins_num, use_slot_size_, dest_gpu_p, pack_->gpu_slot_offsets(),
       reinterpret_cast<const uint64_t*>(value.d_uint64_keys.data<int64_t>()),
       value.d_uint64_offset.data<int>(), value.d_uint64_lens.data<int>(),
       uint64_use_slot_size_, value.d_float_keys.data<float>(),
       value.d_float_offset.data<int>(), value.d_float_lens.data<int>(),
       float_use_slot_size_, used_slot_gpu_types);
}
#endif
//================================ new boxps
//=============================================
#ifdef PADDLE_WITH_BOX_PS
static const int MAX_FILE_BUFF = 4 * 1024 * 1024;
static const int MAX_EXT_LEN = 1024 * 1024;
static const int PAGE_BLOCK_SIZE = 4096;
static const int INT_BYTES = sizeof(int);
BinaryArchiveWriter::BinaryArchiveWriter() : fd_(-1) {
  capacity_ = MAX_FILE_BUFF + MAX_EXT_LEN;
  CHECK_EQ(0, posix_memalign(reinterpret_cast<void**>(&buff_), PAGE_BLOCK_SIZE,
                             capacity_));
}
BinaryArchiveWriter::~BinaryArchiveWriter() {
  close();
  if (buff_ != nullptr) {
    free(buff_);
    buff_ = nullptr;
  }
}
bool BinaryArchiveWriter::open(const std::string& path) {
  fd_ = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_APPEND | O_DIRECT,
               0777);
  if (fd_ < 0) {
    VLOG(0) << "open [" << path << "] failed";
    return false;
  }
  head_ = buff_;
  woffset_ = INT_BYTES;
  return true;
}
bool BinaryArchiveWriter::write(const SlotRecord& rec) {
  thread_local BinaryArchive ar;
  mutex_.lock();
  ar.SetWriteBuffer(&buff_[woffset_], capacity_ - woffset_, nullptr);
  ar << rec;
  woffset_ += ar.Length();
  if (woffset_ < MAX_FILE_BUFF) {
    mutex_.unlock();
    return true;
  }
  // set data length
  int data_len = woffset_ - (head_ - buff_) - INT_BYTES;
  CHECK(data_len > 0 && woffset_ <= capacity_)
      << "write offset: " << woffset_ << ", head offset:" << (head_ - buff_)
      << ", capacity: " << capacity_;
  *(reinterpret_cast<int*>(head_)) = data_len;
  // dio padding 4k
  int left = (woffset_ % PAGE_BLOCK_SIZE);
  int write_len = (woffset_ - left);
  int ret = ::write(fd_, buff_, write_len);
  memmove(buff_, &buff_[write_len], left);
  woffset_ = left + INT_BYTES;
  head_ = &buff_[left];
  mutex_.unlock();

  return (ret == write_len);
}
void BinaryArchiveWriter::close(void) {
  if (fd_ < 0) {
    return;
  }
  mutex_.lock();
  if (woffset_ > INT_BYTES) {
    // set data length
    int data_len = woffset_ - (head_ - buff_) - INT_BYTES;
    CHECK(data_len >= 0) << "write offset: " << woffset_
                         << ", head offset: " << (head_ - buff_);
    int pad_len = woffset_;
    if (data_len == 0) {
      pad_len = pad_len - INT_BYTES;
    } else {
      *(reinterpret_cast<int*>(head_)) = data_len;
    }
    if ((pad_len % PAGE_BLOCK_SIZE) != 0) {
      *(reinterpret_cast<int*>(&buff_[pad_len])) = 0;
      pad_len += (PAGE_BLOCK_SIZE - (pad_len % PAGE_BLOCK_SIZE));
    }
    // dio write 4k
    CHECK(::write(fd_, buff_, pad_len) == pad_len);
    woffset_ = 0;
  }
  mutex_.unlock();
  ::close(fd_);
  fd_ = -1;
}
class BinaryArchiveReader {
 public:
  BinaryArchiveReader() {
    capacity_ = MAX_FILE_BUFF + MAX_EXT_LEN;
    CHECK_EQ(0, posix_memalign(reinterpret_cast<void**>(&buff_),
                               PAGE_BLOCK_SIZE, capacity_));
  }
  ~BinaryArchiveReader() {
    if (buff_ != nullptr) {
      free(buff_);
      buff_ = nullptr;
    }
  }
  bool open(const std::string& path) {
    // dio read
    fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ < 0) {
      VLOG(0) << "open [" << path << "] failed";
      return false;
    }
    return true;
  }
  int read_all(std::function<int(BinaryArchive& ar)> proc_func) {  // NOLINT
    int lines = 0;

    int ret = 0;
    int body_len = 0;
    int left_len = 0;
    int need_len = 0;
    int buff_off = 0;
    char* ptr = buff_;

    BinaryArchive ar;
    while ((ret = ::read(fd_, ptr, (capacity_ - left_len - buff_off))) > 0) {
      left_len += ret;
      ptr = &buff_[buff_off];
      body_len = *(reinterpret_cast<int*>(ptr));
      if (body_len <= 0) {
        break;
      }
      need_len = body_len + INT_BYTES;
      if (left_len < need_len) {
        VLOG(0) << "left length: " << left_len
                << " less need length: " << need_len;
        break;
      }
      while (left_len >= need_len) {
        ar.SetReadBuffer(ptr + INT_BYTES, body_len, nullptr);
        lines += proc_func(ar);
        ptr += need_len;
        left_len -= need_len;
        if (left_len < INT_BYTES) {
          break;
        }
        body_len = *(reinterpret_cast<int*>(ptr));
        if (body_len <= 0) {
          break;
        }
        need_len = body_len + INT_BYTES;
      }
      if (left_len > 0) {
        int align_bytes = left_len % PAGE_BLOCK_SIZE;
        if (align_bytes == 0) {
          memmove(&buff_[0], ptr, left_len);
          ptr = &buff_[left_len];
          buff_off = 0;
        } else {
          buff_off = PAGE_BLOCK_SIZE - align_bytes;
          memmove(&buff_[buff_off], ptr, left_len);
          ptr = &buff_[buff_off + left_len];
        }
      } else {
        ptr = &buff_[0];
        buff_off = 0;
      }
    }

    return lines;
  }
  void close(void) {
    if (fd_ < 0) {
      return;
    }
    ::close(fd_);
  }

 private:
  int fd_ = -1;
  char* buff_ = nullptr;
  size_t capacity_ = 0;
};
void SlotPaddleBoxDataFeed::Init(const DataFeedDesc& data_feed_desc) {
  finish_init_ = false;
  finish_set_filelist_ = false;
  finish_start_ = false;

  slot_pool_ = &SlotRecordPool();

  PADDLE_ENFORCE(data_feed_desc.has_multi_slot_desc(),
                 "Multi_slot_desc has not been set.");

  paddle::framework::MultiSlotDesc multi_slot_desc =
      data_feed_desc.multi_slot_desc();
  SetBatchSize(data_feed_desc.batch_size());
  SetSampleRate(data_feed_desc.sample_rate());
  size_t all_slot_num = multi_slot_desc.slots_size();

  all_slots_.resize(all_slot_num);
  all_slots_info_.resize(all_slot_num);
  used_slots_info_.resize(all_slot_num);
  use_slot_size_ = 0;
  use_slots_.clear();

  float_total_dims_size_ = 0;
  float_total_dims_without_inductives_.clear();
  for (size_t i = 0; i < all_slot_num; ++i) {
    const auto& slot = multi_slot_desc.slots(i);
    all_slots_[i] = slot.name();

    AllSlotInfo& all_slot = all_slots_info_[i];
    all_slot.slot = slot.name();
    all_slot.type = slot.type();
    all_slot.used_idx = slot.is_used() ? use_slot_size_ : -1;
    all_slot.slot_value_idx = -1;

    if (slot.is_used()) {
      UsedSlotInfo& info = used_slots_info_[use_slot_size_];
      info.idx = i;
      info.slot = slot.name();
      info.type = slot.type();
      info.dense = slot.is_dense();
      info.total_dims_without_inductive = 1;
      info.inductive_shape_index = -1;

      // record float value and uint64_t value pos
      if (info.type[0] == 'u') {
        info.slot_value_idx = uint64_use_slot_size_;
        all_slot.slot_value_idx = uint64_use_slot_size_;
        ++uint64_use_slot_size_;
      } else if (info.type[0] == 'f') {
        info.slot_value_idx = float_use_slot_size_;
        all_slot.slot_value_idx = float_use_slot_size_;
        ++float_use_slot_size_;
      }

      use_slots_.push_back(slot.name());

      if (slot.is_dense()) {
        for (int j = 0; j < slot.shape_size(); ++j) {
          if (slot.shape(j) > 0) {
            info.total_dims_without_inductive *= slot.shape(j);
          }
          if (slot.shape(j) == -1) {
            info.inductive_shape_index = j;
          }
        }
      }
      if (info.type[0] == 'f') {
        float_total_dims_without_inductives_.push_back(
            info.total_dims_without_inductive);
        float_total_dims_size_ += info.total_dims_without_inductive;
      }
      info.local_shape.clear();
      for (int j = 0; j < slot.shape_size(); ++j) {
        info.local_shape.push_back(slot.shape(j));
      }
      ++use_slot_size_;
    }
  }
  used_slots_info_.resize(use_slot_size_);

  feed_vec_.resize(used_slots_info_.size());
  const int kEstimatedFeasignNumPerSlot = 5;  // Magic Number
  for (size_t i = 0; i < all_slot_num; i++) {
    batch_float_feasigns_.push_back(std::vector<float>());
    batch_uint64_feasigns_.push_back(std::vector<uint64_t>());
    batch_float_feasigns_[i].reserve(default_batch_size_ *
                                     kEstimatedFeasignNumPerSlot);
    batch_uint64_feasigns_[i].reserve(default_batch_size_ *
                                      kEstimatedFeasignNumPerSlot);
    offset_.push_back(std::vector<size_t>());
    offset_[i].reserve(default_batch_size_ +
                       1);  // Each lod info will prepend a zero
  }

  pipe_command_ = data_feed_desc.pipe_command();
  finish_init_ = true;
  input_type_ = data_feed_desc.input_type();

  rank_offset_name_ = data_feed_desc.rank_offset();
  pv_batch_size_ = data_feed_desc.pv_batch_size();

  // fprintf(stdout, "rank_offset_name: [%s]\n", rank_offset_name_.c_str());
  size_t pos = pipe_command_.find(".so");
  if (pos != std::string::npos) {
    pos = pipe_command_.rfind('|');
    if (pos == std::string::npos) {
      parser_so_path_ = pipe_command_;
      pipe_command_.clear();
    } else {
      parser_so_path_ = pipe_command_.substr(pos + 1);
      pipe_command_ = pipe_command_.substr(0, pos);
    }
    parser_so_path_ = paddle::string::erase_spaces(parser_so_path_);
  } else {
    parser_so_path_.clear();
  }
}
inline bool is_slot_values(const std::string &slot) {
  for (auto &c : slot) {
    if (c < '0' || c > '9') {
        return false;
    }
  }
  return true;
}
void SlotPaddleBoxDataFeed::GetUsedSlotIndex(
    std::vector<int>* used_slot_index, std::vector<std::string>* used_slot_name) {
  auto boxps_ptr = BoxWrapper::GetInstance();
  // get feasigns that FeedPass doesn't need
  const std::unordered_set<std::string>& slot_name_omited_in_feedpass_ =
      boxps_ptr->GetOmitedSlot();

  if (used_slot_index != nullptr) {
    used_slot_index->clear();
  }
  if (used_slot_name != nullptr) {
    used_slot_name->clear();
  }
  const std::vector<int>& slot_ids = boxps_ptr->GetSlotVector();
  for (int i = 0; i < use_slot_size_; ++i) {
    auto& info = used_slots_info_[i];
    if (info.type[0] != 'u') {
      continue;
    }
    if (!is_slot_values(info.slot)) {
      continue;
    }
    if (slot_name_omited_in_feedpass_.find(info.slot) !=
        slot_name_omited_in_feedpass_.end()) {
      continue;
    }
    int slot_id = atoi(info.slot.c_str());
    if (slot_ids.end() ==
        std::find(slot_ids.begin(), slot_ids.end(), slot_id)){
      continue;
    }
    if (used_slot_index != nullptr) {
        used_slot_index->push_back(info.slot_value_idx);
    }
    if (used_slot_name != nullptr) {
        used_slot_name->push_back(info.slot);
    }
  }
}
bool SlotPaddleBoxDataFeed::Start() {
  this->CheckSetFileList();
  this->offset_index_ = 0;
  this->finish_start_ = true;
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
  CHECK(paddle::platform::is_gpu_place(this->place_));
  pack_ = BatchGpuPackMgr().get(this->GetPlace(), used_slots_info_);
#elif defined(PADDLE_WITH_XPU_KP) && !defined(CPU_DATA_FEED)
  pack_ = BatchGpuPackMgr().get(this->GetPlace(), used_slots_info_);
#endif
  if (FLAGS_enable_async_datafeed_batch) {
    slot_pv_tensor_buf_ = std::make_shared<MiniBatchSlotPvTensorBuffer>(used_slots_info_, this->GetPlace());
    slot_pv_tensor_buf_next_ = std::make_shared<MiniBatchSlotPvTensorBuffer>(used_slots_info_, this->GetPlace());
  }
  return true;
}

int SlotPaddleBoxDataFeed::Next() {
  next_timer_.Resume();
  this->CheckStart();
  if (offset_index_ >= static_cast<int>(batch_offsets_.size())) {
    return 0;
  }
#ifdef PADDLE_WITH_XPU_KP
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  std::future<int> prepare_next_batch_rt = std::async(std::launch::async, [this]() {
    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
    return box_ptr->PrepareNextBatch(this->place_.GetDeviceId());
  });
#endif
  auto& batch = batch_offsets_[offset_index_++];
  if (enable_pv_merge_) {
    // join phase : output_pv_channel to consume_pv_channel
    this->batch_size_ = batch.second;
    if (this->batch_size_ != 0) {
      batch_timer_.Resume();
      PutToFeedPvVec(&pv_ins_[batch.first], this->batch_size_);
      batch_timer_.Pause();
    } else {
      VLOG(3) << "finish reading, batch size zero, thread_id=" << thread_id_;
    }

    if (FLAGS_enable_async_datafeed_batch && offset_index_ < static_cast<int>(batch_offsets_.size())) {
      auto & new_batch = batch_offsets_[offset_index_];
      if (new_batch.second != 0) {
        std::future<bool> prefetch_done = std::async(std::launch::async, 
                                                     std::bind(&SlotPaddleBoxDataFeed::PrefechNextBatchWithPv, 
                                                               this, &pv_ins_[new_batch.first], new_batch.second));
        slot_pv_tensor_buf_next_->set_buffer_done(std::move(prefetch_done));
      }
    }

#ifdef PADDLE_WITH_XPU_KP
    CHECK(prepare_next_batch_rt.get() == 0);
#endif
    next_timer_.Pause();

    return this->batch_size_;
  } else {
    this->batch_size_ = batch.second;
    batch_timer_.Resume();
    PutToFeedSlotVec(&records_[batch.first], this->batch_size_);
    // prefetch next batch
    if (FLAGS_enable_async_datafeed_batch && offset_index_ < static_cast<int>(batch_offsets_.size())) {
      auto & new_batch = batch_offsets_[offset_index_];
      //PrefechNextBatch(&records_[new_batch.first], new_batch.second);
      std::future<bool> prefetch_done = std::async(std::launch::async, 
          std::bind(&SlotPaddleBoxDataFeed::PrefechNextBatch, this, &records_[new_batch.first], new_batch.second));
      slot_pv_tensor_buf_next_->set_buffer_done(std::move(prefetch_done));
    }

#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
    // update set join q value
    if (FLAGS_padbox_slotrecord_extend_dim > 0) {
      // pcoc
      pack_->pack_qvalue();
    }
#endif
    batch_timer_.Pause();
#ifdef PADDLE_WITH_XPU_KP
    CHECK(prepare_next_batch_rt.get() == 0);
#endif
    next_timer_.Pause();

    return this->batch_size_;
  }
}
int SlotPaddleBoxDataFeed::GetPackInstance(SlotRecord** ins) {
  if (offset_index_ >= static_cast<int>(batch_offsets_.size())) {
    return 0;
  }
  auto& batch = batch_offsets_[offset_index_];
  *ins = &records_[batch.first];
  return batch.second;
}
int SlotPaddleBoxDataFeed::GetPackPvInstance(SlotPvInstance** pv_ins) {
  if (offset_index_ >= static_cast<int>(batch_offsets_.size())) {
    return 0;
  }
  auto& batch = batch_offsets_[offset_index_];
  *pv_ins = &pv_ins_[batch.first];
  return batch.second;
}
void SlotPaddleBoxDataFeed::AssignFeedVar(const Scope& scope) {
  CheckInit();
  for (int i = 0; i < use_slot_size_; ++i) {
    feed_vec_[i] =
        scope.FindVar(used_slots_info_[i].slot)->GetMutable<LoDTensor>();
  }
  // set rank offset memory
  if (enable_pv_merge_) {
    rank_offset_ = scope.FindVar(rank_offset_name_)->GetMutable<LoDTensor>();
  }
}
void SlotPaddleBoxDataFeed::PutToFeedPvVec(const SlotPvInstance* pvs, int num) {
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
  paddle::platform::SetDeviceId(place_.GetDeviceId());
  pack_->pack_pvinstance(pvs, num);
  int ins_num = pack_->ins_num();
  int pv_num = pack_->pv_num();
  GetRankOffsetGPU(pv_num, ins_num);
  BuildSlotBatchGPU(ins_num);
#elif defined(PADDLE_WITH_XPU_KP) && !defined(CPU_DATA_FEED)
  paddle::platform::SetXPUDeviceId(place_.GetDeviceId());
  if (FLAGS_enable_async_datafeed_batch) {
    if (!slot_pv_tensor_buf_next_->valid()) { // first_batch
      std::future<bool> prefetch_done = std::async(std::launch::async, 
          std::bind(&SlotPaddleBoxDataFeed::PrefechNextBatchWithPv, this, pvs, num));
      slot_pv_tensor_buf_next_->set_buffer_done(std::move(prefetch_done));
    } 

    slot_pv_tensor_buf_next_->wait_buffer_done();
    std::swap(slot_pv_tensor_buf_next_, slot_pv_tensor_buf_);
    auto & bufferd_feed_vec = slot_pv_tensor_buf_->feed_vec();
  
    for (int j = 0; j < use_slot_size_; ++j) {
      feed_vec_[j]->ShareDataWith(bufferd_feed_vec[j]);
      feed_vec_[j]->set_lod(*(bufferd_feed_vec[j].mutable_lod()));
    }

    LoDTensor & pv_buffered_tensor = slot_pv_tensor_buf_->pv_tensor();
    rank_offset_->ShareDataWith(pv_buffered_tensor);
    rank_offset_->set_lod(*(pv_buffered_tensor.mutable_lod()));

  } else { 
    pack_->pack_pvinstance(pvs, num);
    int ins_num = pack_->ins_num();
    int pv_num = pack_->pv_num();
    GetRankOffsetGPU(pv_num, ins_num);
    BuildSlotBatchGPU(ins_num);
  }
#else
  int ins_number = 0;
  pv_ins_vec_.clear();
  for (int i = 0; i < num; ++i) {
    auto& pv = pvs[i];
    ins_number += pv->ads.size();
    for (auto ins : pv->ads) {
      pv_ins_vec_.push_back(ins);
    }
  }
  GetRankOffset(pvs, num, ins_number);
  PutToFeedSlotVec(&pv_ins_vec_[0], ins_number);
#endif
}

// expand values
void SlotPaddleBoxDataFeed::ExpandSlotRecord(SlotRecord* rec) {
  SlotRecord& ins = (*rec);
  if (ins->slot_float_feasigns_.slot_offsets.empty()) {
    return;
  }
  size_t total_value_size = ins->slot_float_feasigns_.slot_values.size();
  if (float_total_dims_size_ == total_value_size) {
    return;
  }

  int float_slot_num =
      static_cast<int>(float_total_dims_without_inductives_.size());
  CHECK(float_slot_num == float_use_slot_size_);
  std::vector<float> old_values;
  std::vector<uint32_t> old_offsets;
  old_values.swap(ins->slot_float_feasigns_.slot_values);
  old_offsets.swap(ins->slot_float_feasigns_.slot_offsets);

  ins->slot_float_feasigns_.slot_values.resize(float_total_dims_size_);
  ins->slot_float_feasigns_.slot_offsets.assign(float_slot_num + 1, 0);

  auto& slot_offsets = ins->slot_float_feasigns_.slot_offsets;
  auto& slot_values = ins->slot_float_feasigns_.slot_values;

  uint32_t offset = 0;
  int num = 0;
  uint32_t old_off = 0;
  int dim = 0;

  for (int i = 0; i < float_slot_num; ++i) {
    dim = float_total_dims_without_inductives_[i];
    old_off = old_offsets[i];
    num = static_cast<int>(old_offsets[i + 1] - old_off);
    if (num == 0) {
      // fill slot value with default value 0
      for (int k = 0; k < dim; ++k) {
        slot_values[k + offset] = 0.0;
      }
    } else {
      if (num == dim) {
        memcpy(&slot_values[offset], &old_values[old_off], dim * sizeof(float));
      } else {
        // position fea
        // record position index need fix values
        int pos_idx = static_cast<int>(old_values[old_off]);
        for (int k = 0; k < dim; ++k) {
          if (k == pos_idx) {
            slot_values[k + offset] = 1.0;
          } else {
            slot_values[k + offset] = 0.0;
          }
        }
      }
    }
    slot_offsets[i] = offset;
    offset += dim;
  }
  slot_offsets[float_slot_num] = offset;
  CHECK(float_total_dims_size_ == static_cast<size_t>(offset));
}

bool SlotPaddleBoxDataFeed::PrefechNextBatchWithPv(const SlotPvInstance* pvs, int num) {
  int ins_number = 0;
  std::vector<SlotRecord> & pv_ins_vec = slot_pv_tensor_buf_next_->pv_ins_vec();
  pv_ins_vec.clear();
  for (int i = 0; i < num; ++i) {
    auto& pv = pvs[i];
    ins_number += pv->ads.size();
    for (auto ins : pv->ads) {
      pv_ins_vec.push_back(ins);
    }
  }
  int max_rank = 3;  // the value is setting
  int row = ins_number;
  int col = max_rank * 2 + 1;

  std::vector<int> rank_offset_mat(row * col, -1);
  rank_offset_mat.shrink_to_fit();

  CalRankOffsetCPU(pvs, num, ins_number, rank_offset_mat, max_rank, row, col);
  int * rank_offset = rank_offset_mat.data();
  slot_pv_tensor_buf_next_->resize_pv_tensor(row, col);
  int * tensor_ptr = slot_pv_tensor_buf_next_->pv_tensor().data<int>();
  CopyToFeedTensor(tensor_ptr, rank_offset, row * col * sizeof(int));

  return PrefechNextBatch(&pv_ins_vec[0], ins_number);
}

bool SlotPaddleBoxDataFeed::PrefechNextBatch(const SlotRecord* ins_vec, int num) {
  int uint64_total_len = 0, float_total_len = 0;
  
  auto & slot_float_feas = slot_pv_tensor_buf_next_->batch_float_feasigns();
  auto & slot_uint64_feas = slot_pv_tensor_buf_next_->batch_uint64_feasigns();
  auto & slot_offsets = slot_pv_tensor_buf_next_->offsets();

  for (int j = 0; j < use_slot_size_; ++j) {
    auto& slot_offset = slot_offsets[j];
    slot_offset.clear();
    slot_offset.reserve(num + 1);
    slot_offset.emplace_back(0);

    int slot_total_instance = 0;
    auto & info = used_slots_info_[j];
    // fill slot value with default value 0
    if (info.type[0] == 'f') {  // float
      auto& batch_fea = slot_float_feas[j];
      batch_fea.clear();

      for (int i = 0; i < num; ++i) {
        auto & r = ins_vec[i];
        size_t fea_num = 0;
        float* slot_values = r->slot_float_feasigns_.get_values(info.slot_value_idx, &fea_num);
        if (fea_num > 0) {
          float_total_len += fea_num;
          batch_fea.resize(slot_total_instance + fea_num);
          memcpy(&batch_fea[slot_total_instance], slot_values,
                sizeof(float) * fea_num);
          slot_total_instance += fea_num;
        }
        slot_offset.push_back(slot_total_instance);
      }

    } else if (info.type[0] == 'u') {  // uint64
      auto& batch_fea = slot_uint64_feas[j];
      batch_fea.clear();

      for (int i = 0; i < num; ++i) {
        auto & r = ins_vec[i];
        size_t fea_num = 0;
        uint64_t* slot_values = r->slot_uint64_feasigns_.get_values(info.slot_value_idx, &fea_num);
        if (fea_num > 0) {
          batch_fea.resize(slot_total_instance + fea_num);
          uint64_total_len += fea_num;
          memcpy(&batch_fea[slot_total_instance], slot_values,
                sizeof(uint64_t) * fea_num);
          slot_total_instance += fea_num;
        }
        slot_offset.push_back(slot_total_instance);
      }
    }
  }

  // alloc mem
  slot_pv_tensor_buf_next_->resize_tensor(float_total_len, uint64_total_len);

  // Copy to feed tensor & add lod
  LoDTensor & float_tensor = slot_pv_tensor_buf_next_->float_tensor();
  LoDTensor & uint64_tensor = slot_pv_tensor_buf_next_->uint64_tensor();
  std::vector<LoDTensor> & feed_vec = slot_pv_tensor_buf_next_->feed_vec();
  feed_vec.clear();
  feed_vec.resize(use_slot_size_);

  size_t slot_uint64_offset = 0;
  size_t slot_float_offset = 0;

  // shared buffer
  for (int j = 0; j < use_slot_size_; ++j) {
    size_t slot_total_len = slot_offsets[j][slot_offsets[j].size() - 1];
    
    int feedvec_len = (slot_total_len == 0 ? 1 : slot_total_len);
    
    auto& info = used_slots_info_[j];
    if (info.type[0] == 'f') {
      feed_vec[j].ShareDataWith(float_tensor.Slice(static_cast<int64_t>(slot_float_offset),
                                                    static_cast<int64_t>(slot_float_offset + feedvec_len)));
      slot_float_offset += feedvec_len;
      if (slot_total_len > 0) {
        CopyToFeedTensor(feed_vec[j].data<float>(), &slot_float_feas[j][0], slot_total_len * sizeof(float));
      }
    } else {
      feed_vec[j].ShareDataWith(uint64_tensor.Slice(static_cast<int64_t>(slot_uint64_offset),
                                                    static_cast<int64_t>(slot_uint64_offset + feedvec_len)));
      slot_uint64_offset += feedvec_len;
      if (slot_total_len > 0) {
        CopyToFeedTensor(feed_vec[j].data<int64_t>(), &slot_uint64_feas[j][0], slot_total_len * sizeof(int64_t));
      }
    }
    feed_vec[j].Resize({static_cast<long int>(slot_total_len), 1});
    if (info.dense) {
      if (info.inductive_shape_index != -1) {
        info.local_shape[info.inductive_shape_index] = slot_total_len / info.total_dims_without_inductive;
      }
      feed_vec[j].Resize(phi::make_ddim(info.local_shape));
    } else {
      LoD& lod = *(feed_vec[j].mutable_lod());
      lod.resize(1);
      lod[0].resize(num + 1);
      paddle::framework::MixVector<size_t> mixv_lod(&lod[0]);
      memcpy(mixv_lod.MutableData(platform::CPUPlace()),
            &slot_offsets[j][0],
            (num + 1) * sizeof(size_t));
    }
  }
  return true;
}

void SlotPaddleBoxDataFeed::PutToFeedSlotVec(const SlotRecord* ins_vec,
                                             int num) {
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX)
  paddle::platform::SetDeviceId(place_.GetDeviceId());
  pack_->pack_instance(ins_vec, num);
  BuildSlotBatchGPU(pack_->ins_num());

#elif defined(PADDLE_WITH_XPU_KP) && !defined(CPU_DATA_FEED)

  paddle::platform::SetXPUDeviceId(place_.GetDeviceId());
  
  if (FLAGS_enable_async_datafeed_batch) {
    if (!slot_pv_tensor_buf_next_->valid()) { // first_batch
      std::future<bool> prefetch_done = std::async(std::launch::async, 
          std::bind(&SlotPaddleBoxDataFeed::PrefechNextBatch, this, ins_vec, num));
      slot_pv_tensor_buf_next_->set_buffer_done(std::move(prefetch_done));
    } 
    slot_pv_tensor_buf_next_->wait_buffer_done();
    std::swap(slot_pv_tensor_buf_next_, slot_pv_tensor_buf_);
    auto & bufferd_feed_vec = slot_pv_tensor_buf_->feed_vec();
    for (int j = 0; j < use_slot_size_; ++j) {
      feed_vec_[j]->ShareDataWith(bufferd_feed_vec[j]);
      feed_vec_[j]->set_lod(*(bufferd_feed_vec[j].mutable_lod()));
    }
    batch_ins_num_ = num;
    ins_record_ptr_ = ins_vec;

  } else {
    pack_->pack_instance(ins_vec, num);
    BuildSlotBatchGPU(pack_->ins_num());
  }

#else // by cpu
  batch_ins_num_ = num;
  ins_record_ptr_ = ins_vec;
  for (int j = 0; j < use_slot_size_; ++j) {
    auto& feed = feed_vec_[j];
    if (feed == nullptr) {
      continue;
    }

    auto& slot_offset = offset_[j];
    slot_offset.clear();
    slot_offset.reserve(num + 1);
    slot_offset.push_back(0);

    int total_instance = 0;
    auto& info = used_slots_info_[j];
    // fill slot value with default value 0
    if (info.type[0] == 'f') {  // float
      auto& batch_fea = batch_float_feasigns_[j];
      batch_fea.clear();

      for (int i = 0; i < num; ++i) {
        auto r = ins_vec[i];
        size_t fea_num = 0;
        float* slot_values =
            r->slot_float_feasigns_.get_values(info.slot_value_idx, &fea_num);
        batch_fea.resize(total_instance + fea_num);
        memcpy(&batch_fea[total_instance], slot_values,
               sizeof(float) * fea_num);
        total_instance += fea_num;
        slot_offset.push_back(total_instance);
      }

      float* feasign = batch_fea.data();
      float* tensor_ptr =
          feed->mutable_data<float>({total_instance, 1}, this->place_);
      if (total_instance > 0) {
        CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(float));
      }
    } else if (info.type[0] == 'u') {  // uint64
      auto& batch_fea = batch_uint64_feasigns_[j];
      batch_fea.clear();

      for (int i = 0; i < num; ++i) {
        auto r = ins_vec[i];
        size_t fea_num = 0;
        uint64_t* slot_values =
            r->slot_uint64_feasigns_.get_values(info.slot_value_idx, &fea_num);
        if (fea_num > 0) {
          batch_fea.resize(total_instance + fea_num);
          memcpy(&batch_fea[total_instance], slot_values,
                 sizeof(uint64_t) * fea_num);
          total_instance += fea_num;
        }
        slot_offset.push_back(total_instance);
      }

      // no uint64_t type in paddlepaddle
      uint64_t* feasign = batch_fea.data();
      int64_t* tensor_ptr =
          feed->mutable_data<int64_t>({total_instance, 1}, this->place_);
      if (total_instance > 0) {
        CopyToFeedTensor(tensor_ptr, feasign, total_instance * sizeof(int64_t));
      }
    }

    if (info.dense) {
      if (info.inductive_shape_index != -1) {
        info.local_shape[info.inductive_shape_index] =
            total_instance / info.total_dims_without_inductive;
      }
      feed->Resize(phi::make_ddim(info.local_shape));
    } else {
      LoD data_lod{slot_offset};
      feed_vec_[j]->set_lod(data_lod);
    }
  }
#endif
}

// template<typename T>
// void print_vector_data(const std::string &name, const T *values, int size) {
//  std::ostringstream ostream;
//  for (int i = 0; i < size; ++i) {
//    ostream << " " << values[i];
//  }
//  LOG(WARNING) << "[" << name << "]" << ostream.str();
//}

void SlotPaddleBoxDataFeed::BuildSlotBatchGPU(const int ins_num) {
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX) || defined(PADDLE_WITH_XPU_KP) && !defined(CPU_DATA_FEED)
  fill_timer_.Resume();

  int offset_cols_size = (ins_num + 1);
  size_t slot_total_num = (use_slot_size_ * offset_cols_size);

  pack_->resize_gpu_slot_offsets(slot_total_num);

  auto& value = pack_->value();
  const UsedSlotGpuType* used_slot_gpu_types =
      static_cast<const UsedSlotGpuType*>(pack_->get_gpu_slots());
#if defined(PADDLE_WITH_CUDA)
  FillSlotValueOffset(ins_num, use_slot_size_, pack_->gpu_slot_offsets(),
                      value.d_uint64_offset.data<int>(), uint64_use_slot_size_,
                      value.d_float_offset.data<int>(), float_use_slot_size_,
                      used_slot_gpu_types);
#elif defined(PADDLE_WITH_XPU_KP)
  DataFeedPdboxXpuKernelHelper::FillSlotValueOffset(this->place_, 
      ins_num, use_slot_size_, reinterpret_cast<unsigned long long*>(pack_->gpu_slot_offsets()),
      value.d_uint64_offset.data<int>(), uint64_use_slot_size_,
      value.d_float_offset.data<int>(), float_use_slot_size_,
      used_slot_gpu_types);
#endif

  fill_timer_.Pause();
  size_t* d_slot_offsets = pack_->gpu_slot_offsets();

  offset_timer_.Resume();
  std::vector<size_t>& offsets = pack_->offsets();
  offsets.resize(slot_total_num);
  std::vector<void*>& h_tensor_ptrs = pack_->h_tensor_ptrs();
  h_tensor_ptrs.resize(use_slot_size_);
  // alloc gpu memory
  pack_->resize_tensor();

  LoDTensor& float_tensor = pack_->float_tensor();
  LoDTensor& uint64_tensor = pack_->uint64_tensor();

  int64_t float_offset = 0;
  int64_t uint64_offset = 0;
  offset_timer_.Pause();

#if defined(PADDLE_WITH_CUDA)
  auto stream = dynamic_cast<phi::GPUContext*>(
            platform::DeviceContextPool::Instance().Get(this->place_))
            ->stream();
#endif

  copy_timer_.Resume();
  // copy index
#if defined(PADDLE_WITH_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(offsets.data(), d_slot_offsets,
                        slot_total_num * sizeof(size_t),
                        cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

#elif defined(PADDLE_WITH_XPU_KP)
  platform::MemcpySyncD2H(offsets.data(), d_slot_offsets, slot_total_num * sizeof(size_t), this->place_);
#endif

  copy_timer_.Pause();
  data_timer_.Resume();

  for (int j = 0; j < use_slot_size_; ++j) {
    auto& feed = feed_vec_[j];
    if (feed == nullptr) {
      h_tensor_ptrs[j] = nullptr;
      continue;
    }

    size_t* off_start_ptr = &offsets[j * offset_cols_size];

    int total_instance = static_cast<int>(off_start_ptr[offset_cols_size - 1]);
    CHECK(total_instance >= 0) << "slot idx:" << j
                               << ", total instance:" << total_instance;

    auto& info = used_slots_info_[j];
    // fill slot value with default value 0
    if (info.type[0] == 'f') {  // float
      if (total_instance > 0) {
        feed->ShareDataWith(float_tensor.Slice(
            static_cast<int64_t>(float_offset),
            static_cast<int64_t>(float_offset + total_instance)));
        float_offset += total_instance;
      } else {
        feed->ShareDataWith(
            float_tensor.Slice(static_cast<int64_t>(float_offset),

                               static_cast<int64_t>(float_offset + 1)));
        float_offset += 1;
      }
      feed->Resize({total_instance, 1});
      h_tensor_ptrs[j] = feed->data<float>();
    } else if (info.type[0] == 'u') {  // uint64
      if (total_instance > 0) {
        feed->ShareDataWith(uint64_tensor.Slice(
            static_cast<int64_t>(uint64_offset),
            static_cast<int64_t>(uint64_offset + total_instance)));
        uint64_offset += total_instance;
      } else {
        feed->ShareDataWith(
            uint64_tensor.Slice(static_cast<int64_t>(uint64_offset),
                                static_cast<int64_t>(uint64_offset + 1)));
        uint64_offset += 1;
      }

      feed->Resize({total_instance, 1});
      h_tensor_ptrs[j] = feed->data<int64_t>();
    }
    if (info.dense) {
      if (info.inductive_shape_index != -1) {
        info.local_shape[info.inductive_shape_index] =
            total_instance / info.total_dims_without_inductive;
      }
      feed->Resize(phi::make_ddim(info.local_shape));
    } else {
      LoD& lod = (*feed->mutable_lod());
      lod.resize(1);
      lod[0].assign(off_start_ptr, off_start_ptr + offset_cols_size);
    }
  }
  data_timer_.Pause();

  trans_timer_.Resume();
  void** dest_gpu_p = reinterpret_cast<void**>(pack_->slot_buf_ptr());

#if defined(PADDLE_WITH_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(dest_gpu_p, h_tensor_ptrs.data(),
                        use_slot_size_ * sizeof(void*),
                        cudaMemcpyHostToDevice, stream));

  CopyForTensor(
      ins_num, use_slot_size_, dest_gpu_p, pack_->gpu_slot_offsets(),
      reinterpret_cast<const uint64_t*>(value.d_uint64_keys.data<int64_t>()),
      value.d_uint64_offset.data<int>(), value.d_uint64_lens.data<int>(),
      uint64_use_slot_size_, value.d_float_keys.data<float>(),
      value.d_float_offset.data<int>(), value.d_float_lens.data<int>(),
      float_use_slot_size_, used_slot_gpu_types);

#elif defined(PADDLE_WITH_XPU_KP)
  platform::MemcpySyncH2D(dest_gpu_p, h_tensor_ptrs.data(), use_slot_size_ * sizeof(unsigned long long), this->place_);
  DataFeedPdboxXpuKernelHelper::CopyForTensor(this->place_, ins_num, use_slot_size_, 
      reinterpret_cast<unsigned long long*>(dest_gpu_p), 
      reinterpret_cast<const unsigned long long*>(pack_->gpu_slot_offsets()),
      reinterpret_cast<const unsigned long long*>(value.d_uint64_keys.data<int64_t>()),
      value.d_uint64_offset.data<int>(), value.d_uint64_lens.data<int>(),
      uint64_use_slot_size_, value.d_float_keys.data<float>(),
      value.d_float_offset.data<int>(), value.d_float_lens.data<int>(),
      float_use_slot_size_, used_slot_gpu_types);
#endif


  trans_timer_.Pause();
#endif
}
int SlotPaddleBoxDataFeed::GetCurrentPhase() {
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  if (box_ptr->Mode() == 1) {  // For AucRunner
    return 1;
  } else {
    return box_ptr->Phase();
  }
}

void SlotPaddleBoxDataFeed::GetRankOffsetGPU(const int pv_num,
                                             const int ins_num) {
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX) || defined(PADDLE_WITH_XPU_KP) && !defined(CPU_DATA_FEED)
  int max_rank = 3;  // the value is setting
  int col = max_rank * 2 + 1;
  auto& value = pack_->value();
  int* tensor_ptr =
      rank_offset_->mutable_data<int>({ins_num, col}, this->place_);
#if defined(PADDLE_WITH_CUDA)
  CopyRankOffset(tensor_ptr, ins_num, pv_num, max_rank,
                 value.d_rank.data<int>(), value.d_cmatch.data<int>(),
                 value.d_ad_offset.data<int>(), col);

#elif defined(PADDLE_WITH_XPU_KP)
  auto dev_ctx = platform::DeviceContextPool::Instance().Get(this->place_);
  auto ctx = static_cast<platform::XPUDeviceContext*>(dev_ctx)->x_context();
  int r = xpu::constant<int>(ctx, tensor_ptr, rank_offset_->numel(), -1);
  PADDLE_ENFORCE_EQ(r,
                    XPU_SUCCESS,
                    platform::errors::External(
                        "XPU constant kernel return wrong value[%d %s]",
                        r,
                        XPUAPIErrorMsg[r]));
  DataFeedPdboxXpuKernelHelper::CopyRankOffset(this->place_, tensor_ptr, ins_num, pv_num, max_rank,
                                               value.d_rank.data<int>(), value.d_cmatch.data<int>(),
                                               value.d_ad_offset.data<int>(), col);
#endif
#endif
}

void SlotPaddleBoxDataFeed::CalRankOffsetCPU(const SlotPvInstance* pv_vec,
                                             int pv_num, int ins_number, std::vector<int> & rank_offset_mat, 
                                             int max_rank, int row, int col) {
  int index = 0;
  for (int i = 0; i < pv_num; i++) {
    auto pv_ins = pv_vec[i];
    int ad_num = pv_ins->ads.size();
    int index_start = index;
    for (int j = 0; j < ad_num; ++j) {
      auto ins = pv_ins->ads[j];
      int rank = -1;
      if ((ins->cmatch == 222 || ins->cmatch == 223) &&
          ins->rank <= static_cast<uint32_t>(max_rank) && ins->rank != 0) {
        rank = ins->rank;
      }

      rank_offset_mat[index * col] = rank;
      if (rank > 0) {
        for (int k = 0; k < ad_num; ++k) {
          auto cur_ins = pv_ins->ads[k];
          int fast_rank = -1;
          if ((cur_ins->cmatch == 222 || cur_ins->cmatch == 223) &&
              cur_ins->rank <= static_cast<uint32_t>(max_rank) &&
              cur_ins->rank != 0) {
            fast_rank = cur_ins->rank;
          }

          if (fast_rank > 0) {
            int m = fast_rank - 1;
            rank_offset_mat[index * col + 2 * m + 1] = cur_ins->rank;
            rank_offset_mat[index * col + 2 * m + 2] = index_start + k;
          }
        }
      }
      index += 1;
    }
  }
}

void SlotPaddleBoxDataFeed::GetRankOffset(const SlotPvInstance* pv_vec,
                                          int pv_num, int ins_number) {

  int max_rank = 3;  // the value is setting
  int row = ins_number;
  int col = max_rank * 2 + 1;

  std::vector<int> rank_offset_mat(row * col, -1);
  rank_offset_mat.shrink_to_fit();

  CalRankOffsetCPU(pv_vec, pv_num, ins_number, rank_offset_mat, max_rank, row, col);

  int* rank_offset = rank_offset_mat.data();
  int* tensor_ptr = rank_offset_->mutable_data<int>({row, col}, this->place_);
  CopyToFeedTensor(tensor_ptr, rank_offset, row * col * sizeof(int));
}

class SlotInsParserMgr {
  struct ParserInfo {
    void* hmodule = nullptr;
    paddle::framework::ISlotParser* parser = nullptr;
  };

 public:
  SlotInsParserMgr() {}
  ~SlotInsParserMgr() {
    if (obj_map_.empty()) {
      return;
    }

    mutex_.lock();
    for (auto it = obj_map_.begin(); it != obj_map_.end(); ++it) {
      auto& info = it->second;
      MyPadBoxFreeObject func_freeobj =
          (MyPadBoxFreeObject)dlsym(info.hmodule, "PadBoxFreeObject");
      if (func_freeobj) {
        func_freeobj(info.parser);
      }
      dlclose(info.hmodule);
    }
    obj_map_.clear();
    mutex_.unlock();
  }

  paddle::framework::ISlotParser* Get(const std::string& path,
                                      const std::vector<AllSlotInfo>& slots) {
    ParserInfo info;
    mutex_.lock();
    std::map<std::string, ParserInfo>::iterator itx = obj_map_.find(path);
    if (itx != obj_map_.end()) {
      mutex_.unlock();
      return itx->second.parser;
    }
    info.hmodule = dlopen(path.c_str(), RTLD_NOW);
    if (info.hmodule == nullptr) {
      mutex_.unlock();
      LOG(FATAL) << "open so path:[" << path << "] failed";
      return nullptr;
    }
    MyPadBoxGetObject func_getobj =
        (MyPadBoxGetObject)dlsym(info.hmodule, "PadBoxGetObject");
    if (func_getobj) {
      info.parser = func_getobj();
      if (!info.parser->Init(slots)) {
        mutex_.unlock();
        LOG(FATAL) << "init so path:[" << path << "] failed";
        return nullptr;
      }
    }
    obj_map_.insert(std::make_pair(path, info));
    mutex_.unlock();

    return info.parser;
  }

 private:
  std::mutex mutex_;
  std::map<std::string, ParserInfo> obj_map_;
};

SlotInsParserMgr& global_parser_pool() {
  static SlotInsParserMgr pool;
  return pool;
}

void SlotPaddleBoxDataFeed::LoadIntoMemory() {
  VLOG(3) << "LoadIntoMemory() begin, thread_id=" << thread_id_;
  if (!parser_so_path_.empty()) {
    LoadIntoMemoryByLib();
  } else {
    LoadIntoMemoryByCommand();
  }
}
// \n split by line
void SlotPaddleBoxDataFeed::LoadIntoMemoryByLine(void) {
  paddle::framework::ISlotParser* parser =
      global_parser_pool().Get(parser_so_path_, all_slots_info_);
  CHECK(parser != nullptr);

  boxps::PaddleDataReader* reader = nullptr;
  if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
    reader =
        boxps::PaddleDataReader::New(BoxWrapper::GetInstance()->GetFileMgr());
  }

  std::string filename;
  BufferedLineFileReader line_reader;
  line_reader.set_sample_rate(sample_rate_);

  BufferedLineFileReader::LineFunc line_func = nullptr;

  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    std::vector<SlotRecord> record_vec;
    platform::Timer timeline;
    timeline.Start();
    int offset = 0;
    int old_offset = 0;

    slot_pool_->get(&record_vec, OBJPOOL_BLOCK_SIZE);
    // get slotrecord object function
    auto record_func = [this, &offset, &record_vec, &old_offset](
        std::vector<SlotRecord>& vec, int num) {
      vec.resize(num);
      if (offset + num > OBJPOOL_BLOCK_SIZE) {
        input_channel_->WriteMove(offset, &record_vec[0]);
        slot_pool_->get(&record_vec[0], offset);
        record_vec.resize(OBJPOOL_BLOCK_SIZE);
        offset = 0;
        old_offset = 0;
      }
      for (int i = 0; i < num; ++i) {
        auto& ins = record_vec[offset + i];
        ins->reset();
        vec[i] = ins;
      }
      offset = offset + num;
    };
    line_func = [this, &parser, &record_vec, &offset, &filename, &record_func,
                 &old_offset](const std::string& line) {
      old_offset = offset;
      if (!parser->ParseOneInstance(line, record_func)) {
        offset = old_offset;
        LOG(WARNING) << "read file:[" << filename << "] item error, line:["
                     << line << "]";
        return false;
      }
      if (offset >= OBJPOOL_BLOCK_SIZE) {
        input_channel_->WriteMove(offset, &record_vec[0]);
        record_vec.clear();
        slot_pool_->get(&record_vec, OBJPOOL_BLOCK_SIZE);
        offset = 0;
      }
      return true;
    };

    int lines = 0;

    do {
      if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
        while (reader->open(filename) < 0) {
          sleep(1);
        }
        lines = line_reader.read_api(reader, line_func, lines);
        reader->close();
      } else {
        if (BoxWrapper::GetInstance()->UseAfsApi()) {
          this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(
              filename, this->pipe_command_);
        } else {
          int err_no = 0;
          this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
        }
        CHECK(this->fp_ != nullptr);
        __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
        lines = line_reader.read_file(this->fp_.get(), line_func, lines);
      }
    } while (line_reader.is_error());

    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (offset < OBJPOOL_BLOCK_SIZE) {
        slot_pool_->put(&record_vec[offset], (OBJPOOL_BLOCK_SIZE - offset));

      }
    } else {
      slot_pool_->put(&record_vec);
    }
    record_vec.clear();
    record_vec.shrink_to_fit();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemoryByLib() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_ << ", lines=" << lines
            << ", sample lines=" << line_reader.get_sample_line()
            << ", filesize=" << line_reader.file_size() / 1024.0 / 1024.0
            << "MB";
  }
  if (reader != nullptr) {
    delete reader;
  }

  VLOG(3) << "LoadIntoMemoryByLib() end, thread_id=" << thread_id_
          << ", total size: " << line_reader.file_size();

}
// split all file
void SlotPaddleBoxDataFeed::LoadIntoMemoryByFile(void) {
  paddle::framework::ISlotParser* parser =
      global_parser_pool().Get(parser_so_path_, all_slots_info_);
  CHECK(parser != nullptr);

  boxps::PaddleDataReader* reader = nullptr;
  if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
    reader =
        boxps::PaddleDataReader::New(BoxWrapper::GetInstance()->GetFileMgr());
  }
  // get slotrecord object
  auto pull_record_func = [this](std::vector<SlotRecord>& record_vec,
                                 int max_fetch_num, int offset) {
    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (max_fetch_num > 0) {
        slot_pool_->get(&record_vec[0], offset);
      } else {  // free all
        max_fetch_num = static_cast<int>(record_vec.size());
        if (max_fetch_num > offset) {
          slot_pool_->put(&record_vec[offset], (max_fetch_num - offset));
        }
      }
    } else if (max_fetch_num > 0) {
      slot_pool_->get(&record_vec, max_fetch_num);
    } else {
      slot_pool_->put(&record_vec);
    }
  };

  std::function<int(char* buf, int len)> read_func = nullptr;
  if (reader != nullptr) {
    read_func = [this, reader](char* buf, int len) {
      return reader->read(buf, len);
    };
  } else {
    read_func = [this](char* buf, int len) {
      return fread(buf, sizeof(char), len, this->fp_.get());
    };
  }

  std::string filename;
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    platform::Timer timeline;
    timeline.Start();

    int lines = 0;
    bool is_ok = true;
    do {
      if (reader != nullptr) {
        while (reader->open(filename) < 0) {
          sleep(1);
        }
      } else {
        if (BoxWrapper::GetInstance()->UseAfsApi()) {
          this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(
              filename, this->pipe_command_);
        } else {
          int err_no = 0;
          this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
        }
        CHECK(this->fp_ != nullptr);
        __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
      }
      if (FLAGS_enable_ins_parser_add_file_path) {
        is_ok = parser->ParseFileInstance(filename.c_str(), read_func,
                                          pull_record_func, lines);
      } else {
        is_ok = parser->ParseFileInstance(read_func, pull_record_func, lines);
      }
      if (reader != nullptr) {
        reader->close();
      }
      if (!is_ok) {
        LOG(WARNING) << "parser error, filename=" << filename
                     << ", lines=" << lines;
      }
    } while (!is_ok);
    timeline.Pause();
    VLOG(3) << "LoadIntoMemoryByLib() read all file, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_ << ", lines=" << lines;
  }
  if (reader != nullptr) {
    delete reader;
  }
}

// load local archive file
void SlotPaddleBoxDataFeed::LoadIntoMemoryByArchive(void) {
  BinaryArchiveReader reader;
  std::string filename;
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "LoadIntoMemoryByArchive PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    platform::Timer timeline;
    timeline.Start();

    int lines = 0;
    while (!reader.open(filename)) {
      sleep(1);
    }

    int offset = 0;
    std::vector<SlotRecord> data;
    slot_pool_->get(&data, OBJPOOL_BLOCK_SIZE);

    auto func = [this, &offset, &data](BinaryArchive& ar) {
      int lines = 0;
      while (ar.Cursor() < ar.Finish()) {
        auto& r = data[offset++];
        //           CHECK(r != nullptr);
        ar >> r;
        //           r->debug();
        if (offset >= OBJPOOL_BLOCK_SIZE) {
          CHECK(input_channel_->WriteMove(offset, &data[0]) ==
                static_cast<size_t>(offset));
          data.clear();
          offset = 0;
          slot_pool_->get(&data, OBJPOOL_BLOCK_SIZE);
        }
        ++lines;
      }
      return lines;
    };
    lines = reader.read_all(func);

    if (offset > 0) {
      CHECK(input_channel_->WriteMove(offset, &data[0]) ==
            static_cast<size_t>(offset));
      if (offset < OBJPOOL_BLOCK_SIZE) {
        slot_pool_->put(&data[offset], (OBJPOOL_BLOCK_SIZE - offset));
      }
    } else {
      slot_pool_->put(&data);
    }

    reader.close();

    timeline.Pause();

    VLOG(3) << "LoadIntoMemoryByArchive() read all file, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_ << ", lines=" << lines;
  }
}

void SlotPaddleBoxDataFeed::LoadIntoMemoryByLib(void) {
  if (is_archive_file_) {
    LoadIntoMemoryByArchive();
  } else if (FLAGS_enable_ins_parser_file) {
    // user defined file format analysis
    LoadIntoMemoryByFile();
  } else {
    LoadIntoMemoryByLine();
  }
}

void SlotPaddleBoxDataFeed::LoadIntoMemoryByCommand(void) {
  std::string filename;
  BufferedLineFileReader line_reader;
  line_reader.set_sample_rate(sample_rate_);

  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    int lines = 0;
    std::vector<SlotRecord> record_vec;
    platform::Timer timeline;
    timeline.Start();
    slot_pool_->get(&record_vec, OBJPOOL_BLOCK_SIZE);
    int offset = 0;

    do {
      if (BoxWrapper::GetInstance()->UseAfsApi()) {
        this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(
            filename, this->pipe_command_);
      } else {
        int err_no = 0;
        this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
      }
      CHECK(this->fp_ != nullptr);
      __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);

      lines = line_reader.read_file(
          this->fp_.get(),
          [this, &record_vec, &offset, &filename](const std::string& line) {
            if (ParseOneInstance(line, &record_vec[offset])) {
              ++offset;
            } else {
              LOG(WARNING) << "read file:[" << filename
                           << "] item error, line:[" << line << "]";
              return false;
            }
            if (offset >= OBJPOOL_BLOCK_SIZE) {
              input_channel_->WriteMove(offset, &record_vec[0]);
              record_vec.clear();
              slot_pool_->get(&record_vec, OBJPOOL_BLOCK_SIZE);
              offset = 0;
            }
            return true;
          },
          lines);
    } while (line_reader.is_error());
    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (offset < OBJPOOL_BLOCK_SIZE) {
        slot_pool_->put(&record_vec[offset], (OBJPOOL_BLOCK_SIZE - offset));

      }
    } else {
      slot_pool_->put(&record_vec);
    }
    record_vec.clear();
    record_vec.shrink_to_fit();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemory() read all lines, file=" << filename
            << ", lines=" << lines
            << ", sample lines=" << line_reader.get_sample_line()
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_;
  }
  VLOG(3) << "LoadIntoMemory() end, thread_id=" << thread_id_
          << ", total size: " << line_reader.file_size();

}

bool SlotPaddleBoxDataFeed::ParseOneInstance(const std::string& line,
                                             SlotRecord* ins) {
  SlotRecord& rec = (*ins);
  // parse line
  const char* str = line.c_str();
  char* endptr = const_cast<char*>(str);
  int pos = 0;

  thread_local std::vector<std::vector<float>> slot_float_feasigns;
  thread_local std::vector<std::vector<uint64_t>> slot_uint64_feasigns;
  slot_float_feasigns.resize(float_use_slot_size_);
  slot_uint64_feasigns.resize(uint64_use_slot_size_);

  if (parse_ins_id_) {
    int num = strtol(&str[pos], &endptr, 10);
    CHECK(num == 1);  // NOLINT
    pos = endptr - str + 1;
    size_t len = 0;
    while (str[pos + len] != ' ') {
      ++len;
    }
    rec->ins_id_ = std::string(str + pos, len);
    pos += len + 1;
  }
  //  if (parse_content_) {
  //    int num = strtol(&str[pos], &endptr, 10);
  //    CHECK(num == 1);  // NOLINT
  //    pos = endptr - str + 1;
  //    size_t len = 0;
  //    while (str[pos + len] != ' ') {
  //      ++len;
  //    }
  //    rec->content_ = std::string(str + pos, len);
  //    pos += len + 1;
  //  }
  if (parse_logkey_) {
    int num = strtol(&str[pos], &endptr, 10);
    CHECK(num == 1);  // NOLINT
    pos = endptr - str + 1;
    size_t len = 0;
    while (str[pos + len] != ' ') {
      ++len;
    }
    // parse_logkey
    std::string log_key = std::string(str + pos, len);
    uint64_t search_id;
    uint32_t cmatch;
    uint32_t rank;
    parser_log_key(log_key, &search_id, &cmatch, &rank);

    rec->ins_id_ = log_key;
    rec->search_id = search_id;
    rec->cmatch = cmatch;
    rec->rank = rank;
    pos += len + 1;
  }

  int float_total_slot_num = 0;
  int uint64_total_slot_num = 0;

  for (size_t i = 0; i < all_slots_info_.size(); ++i) {
    auto& info = all_slots_info_[i];
    int num = strtol(&str[pos], &endptr, 10);
    PADDLE_ENFORCE(num,
                   "The number of ids can not be zero, you need padding "
                   "it in data generator; or if there is something wrong with "
                   "the data, please check if the data contains unresolvable "
                   "characters.\nplease check this error line: %s",
                   str);
    if (info.used_idx != -1) {
      if (info.type[0] == 'f') {  // float
        auto& slot_fea = slot_float_feasigns[info.slot_value_idx];
        slot_fea.clear();
        for (int j = 0; j < num; ++j) {
          float feasign = strtof(endptr, &endptr);
          if (fabs(feasign) < 1e-6 && !used_slots_info_[info.used_idx].dense) {
            continue;
          }
          slot_fea.push_back(feasign);
          ++float_total_slot_num;
        }
      } else if (info.type[0] == 'u') {  // uint64
        auto& slot_fea = slot_uint64_feasigns[info.slot_value_idx];
        slot_fea.clear();
        for (int j = 0; j < num; ++j) {
          uint64_t feasign =
              static_cast<uint64_t>(strtoull(endptr, &endptr, 10));
          if (feasign == 0 && !used_slots_info_[info.used_idx].dense) {
            continue;
          }
          slot_fea.push_back(feasign);
          ++uint64_total_slot_num;
        }
      }
      pos = endptr - str;
    } else {
      for (int j = 0; j <= num; ++j) {
        // pos = line.find_first_of(' ', pos + 1);
        while (line[pos + 1] != ' ') {
          pos++;
        }
      }
    }
  }
  rec->slot_float_feasigns_.add_slot_feasigns(slot_float_feasigns,
                                              float_total_slot_num);
  rec->slot_uint64_feasigns_.add_slot_feasigns(slot_uint64_feasigns,
                                               uint64_total_slot_num);

  return (uint64_total_slot_num > 0);
}

void SlotPaddleBoxDataFeed::UnrollInstance(std::vector<SlotRecord>& items) {
  if (parser_so_path_.empty()) {
    return;

  }
  paddle::framework::ISlotParser* parser =
      global_parser_pool().Get(parser_so_path_, all_slots_info_);

  CHECK(parser != nullptr);
  if (parser->UnrollInstance(items, items.size(),
                             [this](std::vector<SlotRecord>& release) {
                               slot_pool_->put(&release);
                               release.clear();
                               release.shrink_to_fit();
                             })) {
    return;
  }
}

void SlotPaddleBoxDataFeedWithGpuReplicaCache::LoadIntoMemoryByLib(void) {
  paddle::framework::ISlotParser* parser =
      global_parser_pool().Get(parser_so_path_, all_slots_info_);
  CHECK(parser != nullptr);

  boxps::PaddleDataReader* reader = nullptr;
  if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
    reader =
        boxps::PaddleDataReader::New(BoxWrapper::GetInstance()->GetFileMgr());
  }
  std::string filename;
  BufferedLineFileReader line_reader;
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    std::vector<SlotRecord> record_vec;
    platform::Timer timeline;
    timeline.Start();
    const int max_fetch_num = OBJPOOL_BLOCK_SIZE;
    int offset = 0;

    slot_pool_->get(&record_vec, max_fetch_num);
    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
    auto& set = box_ptr->gpu_replica_cache.back();
    auto func = [this, &parser, &set, &record_vec, &offset, &max_fetch_num,
                 &filename](const std::string& line) {
      int old_offset = offset;
      if (!parser->ParseOneInstance(
              line,
              [this, &set](std::vector<float>& gpu_cache) -> int {
                return set.AddItems(gpu_cache);
              },
              [this, &offset, &record_vec, &max_fetch_num, &old_offset](
                  std::vector<SlotRecord>& vec, int num) {
                vec.resize(num);
                if (offset + num > max_fetch_num) {
                  // Considering the prob of show expanding is low, so we don't
                  // update STAT here
                  input_channel_->WriteMove(offset, &record_vec[0]);
                  slot_pool_->get(&record_vec[0], offset);
                  record_vec.resize(max_fetch_num);
                  offset = 0;
                  old_offset = 0;
                }
                for (int i = 0; i < num; ++i) {


                  auto& ins = record_vec[offset + i];
                  ins->reset();
                  vec[i] = ins;
                }
                offset = offset + num;
              })) {
        offset = old_offset;
        LOG(WARNING) << "read file:[" << filename << "] item error, line:["
                     << line << "]";
        return false;
      }

      if (offset >= max_fetch_num) {

        input_channel_->WriteMove(offset, &record_vec[0]);
        record_vec.clear();
        slot_pool_->get(&record_vec, max_fetch_num);
        offset = 0;
      }
      return true;
    };
    int lines = 0;
    do {
      if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
        while (reader->open(filename) < 0) {
          sleep(1);

        }
        lines = line_reader.read_api(reader, func, lines);
        reader->close();
      } else {
        if (BoxWrapper::GetInstance()->UseAfsApi()) {
          this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(
              filename, this->pipe_command_);
        } else {
          int err_no = 0;
          this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
        }
        CHECK(this->fp_ != nullptr);
        __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
        lines = line_reader.read_file(this->fp_.get(), func, lines);
      }
    } while (line_reader.is_error());
    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (offset < max_fetch_num) {
        slot_pool_->put(&record_vec[offset], (max_fetch_num - offset));
      }
    } else {
      slot_pool_->put(&record_vec);
    }
    record_vec.clear();
    record_vec.shrink_to_fit();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemoryByLib() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_ << ", count=" << lines
            << ", filesize=" << line_reader.file_size() / 1024.0 / 1024.0
            << "MB";
  }
  if (reader != nullptr) {
    delete reader;
  }

  VLOG(3) << "LoadIntoMemoryByLib() end, thread_id=" << thread_id_
          << ", total size: " << line_reader.file_size();
}

void SlotPaddleBoxDataFeedWithGpuReplicaCache::LoadIntoMemoryByCommand(void) {
  std::string filename;
  BufferedLineFileReader line_reader;
  std::vector<SlotRecord> record_vec;
  platform::Timer timeline;
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    int lines = 0;
    timeline.Start();
    int offset = 0;
    int gpu_cache_offset;
    int max_fetch_num = OBJPOOL_BLOCK_SIZE;
    slot_pool_->get(&record_vec, max_fetch_num);
    do {
      if (box_ptr->UseAfsApi()) {
        this->fp_ = box_ptr->OpenReadFile(filename, this->pipe_command_);
      } else {
        int err_no = 0;
        this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
      }
      CHECK(this->fp_ != nullptr);
      __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
      lines = line_reader.read_file(
          this->fp_.get(),
          [this, &record_vec, &offset, &max_fetch_num, &gpu_cache_offset,
           &box_ptr, &filename](const std::string& line) {
            if (line[0] == '#') {
              std::vector<float> gpu_cache;
              char* pos = const_cast<char*>(line.c_str() + 1);
              auto& set = box_ptr->gpu_replica_cache.back();
              for (int i = 0; i < set.emb_dim_; ++i) {
                float feasign = strtof(pos, &pos);
                gpu_cache.push_back(feasign);
              }
              gpu_cache_offset = set.AddItems(gpu_cache);
              return true;
            }
            if (ParseOneInstance(line, &record_vec[offset], gpu_cache_offset)) {
              ++offset;
            } else {
              LOG(WARNING) << "read file:[" << filename
                           << "] item error, line:[" << line << "]";
              return false;
            }
            if (offset >= max_fetch_num) {
              input_channel_->WriteMove(offset, &record_vec[0]);
              record_vec.clear();
              slot_pool_->get(&record_vec, max_fetch_num);
              offset = 0;
            }
            return true;
          },
          lines);
    } while (line_reader.is_error());
    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (offset < max_fetch_num) {
        slot_pool_->put(&record_vec[offset], (max_fetch_num - offset));
      }

    } else {
      slot_pool_->put(&record_vec);
    }
    record_vec.clear();
    record_vec.shrink_to_fit();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemory() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_;
  }
  VLOG(3) << "LoadIntoMemory() end, thread_id=" << thread_id_
          << ", total size: " << line_reader.file_size();
}

bool SlotPaddleBoxDataFeedWithGpuReplicaCache::ParseOneInstance(
    const std::string& line, SlotRecord* ins, int gpu_cache_offset) {
  SlotRecord& rec = (*ins);
  // parse line
  const char* str = line.c_str();
  char* endptr = const_cast<char*>(str);
  int pos = 0;

  thread_local std::vector<std::vector<float>> slot_float_feasigns;
  thread_local std::vector<std::vector<uint64_t>> slot_uint64_feasigns;
  slot_float_feasigns.resize(float_use_slot_size_);
  slot_uint64_feasigns.resize(uint64_use_slot_size_);

  if (parse_ins_id_) {
    int num = strtol(&str[pos], &endptr, 10);
    CHECK(num == 1);  // NOLINT
    pos = endptr - str + 1;
    size_t len = 0;
    while (str[pos + len] != ' ') {
      ++len;
    }
    rec->ins_id_ = std::string(str + pos, len);
    pos += len + 1;
  }
  if (parse_logkey_) {
    int num = strtol(&str[pos], &endptr, 10);
    CHECK(num == 1);  // NOLINT
    pos = endptr - str + 1;
    size_t len = 0;
    while (str[pos + len] != ' ') {
      ++len;
    }
    // parse_logkey
    std::string log_key = std::string(str + pos, len);
    uint64_t search_id;
    uint32_t cmatch;
    uint32_t rank;
    parser_log_key(log_key, &search_id, &cmatch, &rank);

    rec->ins_id_ = log_key;
    rec->search_id = search_id;
    rec->cmatch = cmatch;
    rec->rank = rank;
    pos += len + 1;
  }

  int float_total_slot_num = 0;
  int uint64_total_slot_num = 0;

  for (size_t i = 0; i < all_slots_info_.size(); ++i) {
    auto& info = all_slots_info_[i];
    if (i == 3) {
      auto& slot_fea = slot_uint64_feasigns[info.slot_value_idx];
      uint64_t feasign = static_cast<uint64_t>(gpu_cache_offset);
      slot_fea.clear();
      slot_fea.push_back(feasign);
      ++uint64_total_slot_num;
      continue;
    }
    int num = strtol(&str[pos], &endptr, 10);
    PADDLE_ENFORCE(num,
                   "The number of ids can not be zero, you need padding "
                   "it in data generator; or if there is something wrong with "
                   "the data, please check if the data contains unresolvable "
                   "characters.\nplease check this error line: %s",
                   str);

    if (info.used_idx != -1) {
      if (info.type[0] == 'f') {  // float
        auto& slot_fea = slot_float_feasigns[info.slot_value_idx];
        slot_fea.clear();
        for (int j = 0; j < num; ++j) {
          float feasign = strtof(endptr, &endptr);
          if (fabs(feasign) < 1e-6 && !used_slots_info_[info.used_idx].dense) {
            continue;
          }
          slot_fea.push_back(feasign);
          ++float_total_slot_num;

        }
      } else if (info.type[0] == 'u') {  // uint64
        auto& slot_fea = slot_uint64_feasigns[info.slot_value_idx];
        slot_fea.clear();
        for (int j = 0; j < num; ++j) {
          uint64_t feasign =
              static_cast<uint64_t>(strtoull(endptr, &endptr, 10));
          if (feasign == 0 && !used_slots_info_[info.used_idx].dense) {
            continue;
          }
          slot_fea.push_back(feasign);
          ++uint64_total_slot_num;
        }
      }
      pos = endptr - str;
    } else {
      for (int j = 0; j <= num; ++j) {

        // pos = line.find_first_of(' ', pos + 1);
        while (line[pos + 1] != ' ') {
          pos++;
        }
      }
    }
  }
  rec->slot_float_feasigns_.add_slot_feasigns(slot_float_feasigns,
                                              float_total_slot_num);
  rec->slot_uint64_feasigns_.add_slot_feasigns(slot_uint64_feasigns,
                                               uint64_total_slot_num);
  return (uint64_total_slot_num > 0);
}

void InputTableDataFeed::LoadIntoMemoryByLib() {
  paddle::framework::ISlotParser* parser =
      global_parser_pool().Get(parser_so_path_, all_slots_info_);
  CHECK(parser != nullptr);

  boxps::PaddleDataReader* reader = nullptr;
  if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
    reader =
        boxps::PaddleDataReader::New(BoxWrapper::GetInstance()->GetFileMgr());
  }

  std::string filename;
  BufferedLineFileReader line_reader;
  line_reader.set_sample_rate(sample_rate_);

  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  PADDLE_ENFORCE(!box_ptr->input_table_deque_.empty());
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;
    std::vector<SlotRecord> record_vec;
    platform::Timer timeline;
    timeline.Start();
    const int max_fetch_num = OBJPOOL_BLOCK_SIZE;
    int offset = 0;

    slot_pool_->get(&record_vec, max_fetch_num);
    auto func = [this, &box_ptr, &parser, &record_vec, &offset, &max_fetch_num,
                 &filename](const std::string& line) {
      int old_offset = offset;
      auto GetOffsetFunc = [&box_ptr](std::string& key) -> uint64_t {
        return box_ptr->input_table_deque_.back().GetIndexOffset(key);
      };

      if (!parser->ParseOneInstance(
              line, GetOffsetFunc,
              [this, &offset, &record_vec, &max_fetch_num, &old_offset](
                  std::vector<SlotRecord>& vec, int num) {
                vec.resize(num);
                if (offset + num > max_fetch_num) {
                  // Considering the prob of show expanding is low, so we don't
                  // update STAT here
                  input_channel_->WriteMove(offset, &record_vec[0]);
                  slot_pool_->get(&record_vec[0], offset);
                  record_vec.resize(max_fetch_num);
                  offset = 0;
                  old_offset = 0;
                }
                for (int i = 0; i < num; ++i) {
                  auto& ins = record_vec[offset + i];
                  ins->reset();
                  vec[i] = ins;
                }
                offset = offset + num;
              })) {
        offset = old_offset;
        LOG(WARNING) << "read file:[" << filename << "] item error, line:["
                     << line << "]";
        return false;
      }
      if (offset >= max_fetch_num) {
        input_channel_->WriteMove(offset, &record_vec[0]);
        record_vec.clear();
        slot_pool_->get(&record_vec, max_fetch_num);
        offset = 0;
      }
      return true;
    };
    int lines = 0;
    do {
      if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
        while (reader->open(filename) < 0) {
          sleep(1);
        }
        lines = line_reader.read_api(reader, func, lines);
        reader->close();
      } else {
        if (BoxWrapper::GetInstance()->UseAfsApi()) {
          this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(
              filename, this->pipe_command_);
        } else {
          int err_no = 0;
          this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
        }
        CHECK(this->fp_ != nullptr);
        __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
        lines = line_reader.read_file(this->fp_.get(), func, lines);
      }
    } while (line_reader.is_error());
    if (offset > 0) {
      input_channel_->WriteMove(offset, &record_vec[0]);
      if (offset < max_fetch_num) {
        slot_pool_->put(&record_vec[offset], (max_fetch_num - offset));
      }
    } else {
      slot_pool_->put(&record_vec);

    }
    record_vec.clear();
    record_vec.shrink_to_fit();
    timeline.Pause();
    VLOG(3) << "LoadIntoMemoryByLib() read all lines, file=" << filename
            << ", cost time=" << timeline.ElapsedSec()
            << " seconds, thread_id=" << thread_id_ << ", lines=" << lines
            << ", sample lines=" << line_reader.get_sample_line()
            << ", filesize=" << line_reader.file_size() / 1024.0 / 1024.0
            << "MB";
  }

  if (reader != nullptr) {
    delete reader;
  }

  VLOG(3) << "LoadIntoMemoryByLib() end, thread_id=" << thread_id_
          << ", total size: " << line_reader.file_size();
}

void InputIndexDataFeed::LoadIntoMemory() {
  std::vector<AllSlotInfo> slots_info;
  paddle::framework::ISlotParser* parser =
      global_parser_pool().Get(parser_so_path_, slots_info);
  CHECK(parser != nullptr);

  boxps::PaddleDataReader* reader = nullptr;
  if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
    reader =
        boxps::PaddleDataReader::New(BoxWrapper::GetInstance()->GetFileMgr());
  }

  std::string filename;
  BufferedLineFileReader line_reader;
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  PADDLE_ENFORCE(!box_ptr->input_table_deque_.empty());
  while (this->PickOneFile(&filename)) {
    VLOG(3) << "PickOneFile, filename=" << filename
            << ", thread_id=" << thread_id_;

    auto func = [this, &box_ptr, &filename, &parser](const std::string& line) {
      auto ret = parser->ParseIndexData(
          line, [&box_ptr](std::string& key, std::vector<float>& vec) {
            box_ptr->input_table_deque_.back().AddIndexData(key, vec);
          });
      if (!ret) {
        LOG(WARNING) << "read file:[" << filename << "] item error, line:["
                     << line << "]";
        return false;
      }
      return true;
    };

    int lines = 0;
    do {
      if (BoxWrapper::GetInstance()->UseAfsApi() && pipe_command_.empty()) {
        while (reader->open(filename) < 0) {
          sleep(1);
        }
        lines = line_reader.read_api(reader, func, lines);
        reader->close();
      } else {
        if (BoxWrapper::GetInstance()->UseAfsApi()) {
          this->fp_ = BoxWrapper::GetInstance()->OpenReadFile(
              filename, this->pipe_command_);
        } else {
          int err_no = 0;
          this->fp_ = fs_open_read(filename, &err_no, this->pipe_command_);
        }
        CHECK(this->fp_ != nullptr);
        __fsetlocking(&*(this->fp_), FSETLOCKING_BYCALLER);
        lines = line_reader.read_file(this->fp_.get(), func, lines);
      }
    } while (line_reader.is_error());
    VLOG(3) << "read file:[" << filename << "], lines:[" << lines << "]";
  }

  if (reader) {
    delete reader;
  }
}
#endif
////////////////////////////// pack ////////////////////////////////////
#if defined(PADDLE_WITH_CUDA) && defined(_LINUX) || defined(PADDLE_WITH_XPU_KP)
MiniBatchGpuPack::MiniBatchGpuPack(const paddle::platform::Place& place,
                                   const std::vector<UsedSlotInfo>& infos) {
  place_ = place;
#if defined(PADDLE_WITH_CUDA)
  stream_ = dynamic_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();
#endif
  ins_num_ = 0;
  pv_num_ = 0;
  used_float_num_ = 0;
  used_uint64_num_ = 0;
  enable_pv_ = false;

  used_slot_size_ = static_cast<int>(infos.size());
  for (int i = 0; i < used_slot_size_; ++i) {
    auto& info = infos[i];
    if (info.type[0] == 'u') {
      gpu_used_slots_.push_back({1, info.slot_value_idx});
      ++used_uint64_num_;
    } else {
      gpu_used_slots_.push_back({0, info.slot_value_idx});
      ++used_float_num_;
    }
  }
  gpu_slots_ = memory::AllocShared(
      place_, gpu_used_slots_.size() * sizeof(UsedSlotGpuType));

#if defined(PADDLE_WITH_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(gpu_slots_->ptr(), gpu_used_slots_.data(),
                             gpu_used_slots_.size() * sizeof(UsedSlotGpuType),
                             cudaMemcpyHostToDevice, stream_));
#elif defined(PADDLE_WITH_XPU_KP)
  platform::MemcpySyncH2D(gpu_slots_->ptr(), gpu_used_slots_.data(), 
                          gpu_used_slots_.size() * sizeof(UsedSlotGpuType), this->place_);
#endif
  slot_buf_ptr_ = memory::AllocShared(place_, used_slot_size_ * sizeof(void*));
#ifdef PADDLE_WITH_BOX_PS
  int device_id = place.GetDeviceId();
  VLOG(3) << "begin get batch pack device id: " << device_id;
  qvalue_tensor_ = &BoxWrapper::GetInstance()->GetQTensor(device_id);
#endif
  // sync
#if defined(PADDLE_WITH_CUDA)
  CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif
}

MiniBatchGpuPack::~MiniBatchGpuPack() {}

void MiniBatchGpuPack::reset(const paddle::platform::Place& place) {
  place_ = place;
#if defined(PADDLE_WITH_CUDA)
  stream_ = dynamic_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(place))
          ->stream();
#endif
  ins_num_ = 0;
  pv_num_ = 0;
  enable_pv_ = false;

  pack_timer_.Reset();
  trans_timer_.Reset();

#ifdef PADDLE_WITH_BOX_PS
  int device_id = place.GetDeviceId();
  qvalue_tensor_ = &BoxWrapper::GetInstance()->GetQTensor(device_id);
#endif
}

void MiniBatchGpuPack::pack_pvinstance(const SlotPvInstance* pv_ins, int num) {
  pv_num_ = num;
  buf_.h_ad_offset.resize(num + 1);
  buf_.h_ad_offset[0] = 0;
  size_t ins_number = 0;

  ins_vec_.clear();
  for (int i = 0; i < num; ++i) {
    auto& pv = pv_ins[i];
    ins_number += pv->ads.size();
    for (auto ins : pv->ads) {
      ins_vec_.push_back(ins);
    }
    buf_.h_ad_offset[i + 1] = ins_number;
  }
  buf_.h_rank.resize(ins_number);
  buf_.h_cmatch.resize(ins_number);
  enable_pv_ = true;

  pack_instance(&ins_vec_[0], ins_number);
}

void MiniBatchGpuPack::pack_all_data(const SlotRecord* ins_vec, int num) {
  int uint64_total_num = 0;
  int float_total_num = 0;

  buf_.h_uint64_lens.resize(num + 1);
  buf_.h_uint64_lens[0] = 0;
  buf_.h_float_lens.resize(num + 1);
  buf_.h_float_lens[0] = 0;

  if (enable_pv_) {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      uint64_total_num += r->slot_uint64_feasigns_.slot_values.size();
      buf_.h_uint64_lens[i + 1] = uint64_total_num;
      float_total_num += r->slot_float_feasigns_.slot_values.size();
      buf_.h_float_lens[i + 1] = float_total_num;

      buf_.h_rank[i] = r->rank;
      buf_.h_cmatch[i] = r->cmatch;
    }
  } else {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      uint64_total_num += r->slot_uint64_feasigns_.slot_values.size();
      buf_.h_uint64_lens[i + 1] = uint64_total_num;
      float_total_num += r->slot_float_feasigns_.slot_values.size();
      buf_.h_float_lens[i + 1] = float_total_num;
    }
  }

  int uint64_cols = (used_uint64_num_ + 1);
  buf_.h_uint64_offset.resize(uint64_cols * num);
  buf_.h_uint64_keys.resize(uint64_total_num);

  int float_cols = (used_float_num_ + 1);
  buf_.h_float_offset.resize(float_cols * num);
  buf_.h_float_keys.resize(float_total_num);

  size_t fea_num = 0;
  uint64_total_num = 0;
  float_total_num = 0;
  for (int i = 0; i < num; ++i) {
    auto r = ins_vec[i];
    auto& uint64_feasigns = r->slot_uint64_feasigns_;
    fea_num = uint64_feasigns.slot_values.size();
    if (fea_num > 0) {
      memcpy(&buf_.h_uint64_keys[uint64_total_num],
             uint64_feasigns.slot_values.data(), fea_num * sizeof(uint64_t));

    }
    uint64_total_num += fea_num;
    // copy uint64 offset
    memcpy(&buf_.h_uint64_offset[i * uint64_cols],
           uint64_feasigns.slot_offsets.data(), sizeof(int) * uint64_cols);


    auto& float_feasigns = r->slot_float_feasigns_;
    fea_num = float_feasigns.slot_values.size();
    memcpy(&buf_.h_float_keys[float_total_num],
           float_feasigns.slot_values.data(), fea_num * sizeof(float));

    float_total_num += fea_num;

    // copy float offset
    memcpy(&buf_.h_float_offset[i * float_cols],
           float_feasigns.slot_offsets.data(), sizeof(int) * float_cols);
  }

  CHECK(uint64_total_num == static_cast<int>(buf_.h_uint64_lens.back()))
      << "uint64 value length error";
  CHECK(float_total_num == static_cast<int>(buf_.h_float_lens.back()))
      << "float value length error";
}
void MiniBatchGpuPack::pack_uint64_data(const SlotRecord* ins_vec, int num) {
  int uint64_total_num = 0;

  buf_.h_float_lens.clear();
  buf_.h_float_keys.clear();
  buf_.h_float_offset.clear();

  buf_.h_uint64_lens.resize(num + 1);
  buf_.h_uint64_lens[0] = 0;

  if (enable_pv_) {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      uint64_total_num += r->slot_uint64_feasigns_.slot_values.size();
      buf_.h_uint64_lens[i + 1] = uint64_total_num;

      buf_.h_rank[i] = r->rank;
      buf_.h_cmatch[i] = r->cmatch;
    }
  } else {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      uint64_total_num += r->slot_uint64_feasigns_.slot_values.size();
      buf_.h_uint64_lens[i + 1] = uint64_total_num;
    }
  }

  int uint64_cols = (used_uint64_num_ + 1);
  buf_.h_uint64_offset.resize(uint64_cols * num);
  buf_.h_uint64_keys.resize(uint64_total_num);

  size_t fea_num = 0;
  uint64_total_num = 0;
  for (int i = 0; i < num; ++i) {
    auto r = ins_vec[i];
    auto& uint64_feasigns = r->slot_uint64_feasigns_;
    fea_num = uint64_feasigns.slot_values.size();
    if (fea_num > 0) {
      memcpy(&buf_.h_uint64_keys[uint64_total_num],
             uint64_feasigns.slot_values.data(), fea_num * sizeof(uint64_t));

    }
    uint64_total_num += fea_num;
    // copy uint64 offset
    memcpy(&buf_.h_uint64_offset[i * uint64_cols],
           uint64_feasigns.slot_offsets.data(), sizeof(int) * uint64_cols);

  }
  CHECK(uint64_total_num == static_cast<int>(buf_.h_uint64_lens.back()))
      << "uint64 value length error";
}
void MiniBatchGpuPack::pack_float_data(const SlotRecord* ins_vec, int num) {
  int float_total_num = 0;

  buf_.h_uint64_lens.clear();
  buf_.h_uint64_offset.clear();
  buf_.h_uint64_keys.clear();

  buf_.h_float_lens.resize(num + 1);
  buf_.h_float_lens[0] = 0;

  if (enable_pv_) {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      float_total_num += r->slot_float_feasigns_.slot_values.size();
      buf_.h_float_lens[i + 1] = float_total_num;

      buf_.h_rank[i] = r->rank;
      buf_.h_cmatch[i] = r->cmatch;
    }
  } else {
    for (int i = 0; i < num; ++i) {
      auto r = ins_vec[i];
      float_total_num += r->slot_float_feasigns_.slot_values.size();
      buf_.h_float_lens[i + 1] = float_total_num;
    }
  }

  int float_cols = (used_float_num_ + 1);
  buf_.h_float_offset.resize(float_cols * num);
  buf_.h_float_keys.resize(float_total_num);

  size_t fea_num = 0;
  float_total_num = 0;
  for (int i = 0; i < num; ++i) {
    auto r = ins_vec[i];
    auto& float_feasigns = r->slot_float_feasigns_;
    fea_num = float_feasigns.slot_values.size();
    memcpy(&buf_.h_float_keys[float_total_num],
           float_feasigns.slot_values.data(), fea_num * sizeof(float));

    float_total_num += fea_num;

    // copy float offset
    memcpy(&buf_.h_float_offset[i * float_cols],
           float_feasigns.slot_offsets.data(), sizeof(int) * float_cols);

  }
  CHECK(float_total_num == static_cast<int>(buf_.h_float_lens.back()))
      << "float value length error";
}

void MiniBatchGpuPack::pack_instance(const SlotRecord* ins_vec, int num) {
  pack_timer_.Resume();
  ins_num_ = num;
  batch_ins_ = ins_vec;
  CHECK(used_uint64_num_ > 0 || used_float_num_ > 0);
  // uint64 and float
  if (used_uint64_num_ > 0 && used_float_num_ > 0) {
    pack_all_data(ins_vec, num);
  } else if (used_uint64_num_ > 0) {  // uint64
    pack_uint64_data(ins_vec, num);
  } else {  // only float
    pack_float_data(ins_vec, num);
  }
  pack_timer_.Pause();
  // to gpu or xpu
  transfer_to_gpu();
}

void MiniBatchGpuPack::transfer_to_gpu(void) {
  trans_timer_.Resume();
  if (enable_pv_) {
    copy_host2device(&value_.d_ad_offset, buf_.h_ad_offset.data(),
                     buf_.h_ad_offset.size());
    copy_host2device(&value_.d_rank, buf_.h_rank.data(), buf_.h_rank.size());
    copy_host2device(&value_.d_cmatch, buf_.h_cmatch.data(),
                     buf_.h_cmatch.size());
  }
  copy_host2device(&value_.d_uint64_lens, buf_.h_uint64_lens.data(),
                   buf_.h_uint64_lens.size());
  copy_host2device<int64_t>(
      &value_.d_uint64_keys,
      reinterpret_cast<int64_t*>(buf_.h_uint64_keys.data()),
      buf_.h_uint64_keys.size());
  copy_host2device(&value_.d_uint64_offset, buf_.h_uint64_offset.data(),
                   buf_.h_uint64_offset.size());

  copy_host2device(&value_.d_float_lens, buf_.h_float_lens.data(),
                   buf_.h_float_lens.size());
  copy_host2device(&value_.d_float_keys, buf_.h_float_keys.data(),
                   buf_.h_float_keys.size());
  copy_host2device(&value_.d_float_offset, buf_.h_float_offset.data(),
                   buf_.h_float_offset.size());
#if defined(PADDLE_WITH_CUDA)
  CUDA_CHECK(cudaStreamSynchronize(stream_));
#endif
  trans_timer_.Pause();
}

//================================ pcoc
//=========================================
// pack pcoc q to gpu
void MiniBatchGpuPack::pack_qvalue(void) {
  int len = ins_num_ * extend_dim_;
  std::vector<float> qvalue;
  qvalue.resize(len);

  int off = 0;
  char* ptr = NULL;
  for (int i = 0; i < ins_num_; ++i) {
    ptr = reinterpret_cast<char*>(batch_ins_[i]);
    float* q = reinterpret_cast<float*>(&ptr[sizeof(SlotRecordObject)]);
    for (int k = 0; k < extend_dim_; ++k) {
      qvalue[off++] = q[k];
    }
  }
  CHECK(off == len);

  float* tensor_ptr =
      qvalue_tensor_->mutable_data<float>({len, 1}, this->place_);
#if defined(PADDLE_WITH_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(tensor_ptr, &qvalue[0], len * sizeof(float),
                             cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
#elif defined(PADDLE_WITH_XPU_KP)
  platform::MemcpySyncH2D(tensor_ptr, &qvalue[0], len * sizeof(float), this->place_);
#endif
}

// store pcoc q value
void MiniBatchGpuPack::store_qvalue(const std::vector<Tensor>& qvalue) {
  CHECK(static_cast<int>(qvalue.size()) == extend_dim_);
  SlotRecord* batch_records = const_cast<SlotRecord*>(batch_ins_);
  char* ptr = NULL;
  for (int i = 0; i < extend_dim_; ++i) {
    CHECK(static_cast<int>(qvalue[i].numel()) == ins_num_);
    const float* q = qvalue[i].data<float>();
    for (int k = 0; k < ins_num_; ++k) {
      ptr = reinterpret_cast<char*>(batch_records[k]);
      reinterpret_cast<float*>(&ptr[sizeof(SlotRecordObject)])[i] = q[k];
    }
  }
}
#endif

}  // namespace framework
}  // namespace paddle
