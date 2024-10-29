/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/platform/device/xpu/xpu_info.h"

#include <algorithm>
#include <cstdlib>
#include <string>

#include "gflags/gflags.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/enforce_xpu.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/profiler/mem_tracing.h"
#include "paddle/fluid/memory/stats.h"

DECLARE_uint64(xpu_memory_limit_mb);
DECLARE_double(fraction_of_xpu_memory_to_use);
DECLARE_uint64(initial_xpu_memory_in_mb);
DECLARE_uint64(reallocate_xpu_memory_in_mb);

constexpr static float fraction_reserve_xpu_memory = 0.05f;

PADDLE_DEFINE_EXPORTED_bool(enable_xpu_memory_usage_log,
                            false,
                            "Whether to print the message of xpu memory usage "
                            "at exit, mainly used for UT and CI.");
PADDLE_DEFINE_EXPORTED_bool(enable_xpu_memory_usage_log_mb,
                            true,
                            "Whether to print the message of xpu memory usage "
                            "MB as a unit of measurement.");

namespace paddle {
namespace platform {

/**************************** Version Management **************************/

//! Get the version of XPU Driver
int GetDriverVersion() { return phi::backends::xpu::GetDriverVersion(); }

//! Get the version of XPU Runtime
int GetRuntimeVersion() { return phi::backends::xpu::GetRuntimeVersion(); }

/**************************** Device Management **************************/

int GetXPUDeviceCount() { return phi::backends::xpu::GetXPUDeviceCount(); }

int GetXPUCurrentDeviceId() {
  return phi::backends::xpu::GetXPUCurrentDeviceId();
}

void SetXPUDeviceId(int id) { phi::backends::xpu::SetXPUDeviceId(id); }

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices() {
  // use user specified XPUs in single-node multi-process mode.
  return phi::backends::xpu::GetXPUSelectedDevices();
}

/**************************** Memory Management **************************/

void MemcpySyncH2D(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUPlace& dst_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(dst_place);
  dev_ctx->Wait();
  phi::backends::xpu::MemcpySyncH2D(dst, src, count, dst_place, *dev_ctx);
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUPlace& src_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  dev_ctx->Wait();
  phi::backends::xpu::MemcpySyncD2H(dst, src, count, src_place, *dev_ctx);
}

// if src.device == dst.device and you need sync , after call this function,
// need to call xpu_wait()
void MemcpySyncD2D(void* dst,
                   const platform::XPUPlace& dst_place,
                   const void* src,
                   const platform::XPUPlace& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}

void MemcpySyncH2D(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUL3Place& dst_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(dst_place);
  dev_ctx->Wait();
  phi::backends::xpu::MemcpySyncH2D(dst, src, count, dst_place, *dev_ctx);
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const platform::XPUL3Place& src_place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  dev_ctx->Wait();
  phi::backends::xpu::MemcpySyncD2H(dst, src, count, src_place, *dev_ctx);
}

// if src.device == dst.device and you need sync , after call this function,
// need to call xpu_wait()
void MemcpySyncD2D(void* dst,
                   const platform::XPUL3Place& dst_place,
                   const void* src,
                   const platform::XPUL3Place& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}
void MemcpySyncD2D(void* dst,
                   const platform::XPUPlace& dst_place,
                   const void* src,
                   const platform::XPUL3Place& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}
void MemcpySyncD2D(void* dst,
                   const platform::XPUL3Place& dst_place,
                   const void* src,
                   const platform::XPUPlace& src_place,
                   size_t count) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  auto* dev_ctx = pool.GetByPlace(src_place);
  phi::backends::xpu::MemcpySyncD2D(
      dst, dst_place, src, src_place, count, *dev_ctx);
}

void MemcpySyncH2D(void* dst,
                   const void* src,
                   size_t count,
                   const platform::Place& dst_place) {
  if (dst_place.GetType() == phi::AllocationType::XPUL3) {
    platform::XPUL3Place place_dst(dst_place.GetDeviceId());
    MemcpySyncH2D(dst, src, count, place_dst);
  } else if(dst_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_dst(dst_place.GetDeviceId());
    MemcpySyncH2D(dst, src, count, place_dst);
  }
}

void MemcpySyncD2H(void* dst,
                   const void* src,
                   size_t count,
                   const platform::Place& src_place) {
  if (src_place.GetType() == phi::AllocationType::XPUL3) {
    platform::XPUL3Place place_src(src_place.GetDeviceId());
    MemcpySyncD2H(dst, src, count, place_src);
  } else if(src_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_src(src_place.GetDeviceId());
    MemcpySyncD2H(dst, src, count, place_src);
  }
}

// if src.device == dst.device and you need sync , after call this function,
// need to call xpu_wait()
void MemcpySyncD2D(void* dst,
                   const platform::Place& dst_place,
                   const void* src,
                   const platform::Place& src_place,
                   size_t count) {
  if (src_place.GetType() == phi::AllocationType::XPUL3 && src_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUL3Place place_src(src_place.GetDeviceId());
    platform::XPUPlace place_dst(dst_place.GetDeviceId());
    MemcpySyncD2D(dst, place_dst, src, place_src, count);
  } else if (src_place.GetType() == phi::AllocationType::XPU && src_place.GetType() == phi::AllocationType::XPUL3) {
    platform::XPUPlace place_src(src_place.GetDeviceId());
    platform::XPUL3Place place_dst(dst_place.GetDeviceId());
    MemcpySyncD2D(dst, place_dst, src, place_src, count);
  } else if (src_place.GetType() == phi::AllocationType::XPU && src_place.GetType() == phi::AllocationType::XPU) {
    platform::XPUPlace place_src(src_place.GetDeviceId());
    platform::XPUPlace place_dst(dst_place.GetDeviceId());
    MemcpySyncD2D(dst, place_dst, src, place_src, count);
  } else if (src_place.GetType() == phi::AllocationType::XPUL3 && src_place.GetType() == phi::AllocationType::XPUL3) {
    platform::XPUL3Place place_src(src_place.GetDeviceId());
    platform::XPUL3Place place_dst(dst_place.GetDeviceId());
    MemcpySyncD2D(dst, place_dst, src, place_src, count);
  }
}

void XPUStreamSync(xpuStream stream) {
  PADDLE_ENFORCE_XDNN_SUCCESS(xpu_wait(stream), "xpu_wait");
}

/**************************** Others **************************/

phi::backends::xpu::XPUVersion get_xpu_version(int dev_id) {
  return phi::backends::xpu::get_xpu_version(dev_id);
}

std::once_flag XPUMLHandler::init_flag_;

XPUMLHandler::XPUMLHandler() {
  std::call_once(XPUMLHandler::init_flag_, &XPUMLHandler::init_ml);
  xpumlDeviceGetCount(&device_nums_);
  device_handlers_.resize(device_nums_);
  mem_infos_.resize(device_nums_);
  for (unsigned int i = 0; i < device_nums_; ++i) {
      xpumlDeviceGetHandleByIndex(i, &device_handlers_[i]);
  }
}


/**************************** Memory Management **************************/
// == Memory monitor ==
void XPUMLHandler::init_ml() {
    xpumlInit();
}

bool XPUMLHandler::getMemoryUsageInfo(int dev_id, unsigned long long *total,
                                      unsigned long long* used, unsigned long long *free) {
  if(xpumlDeviceGetMemoryInfo(device_handlers_[dev_id], &mem_infos_[dev_id]) != xpumlReturn_enum::XPUML_SUCCESS) {
    return false;
  }
  *total = mem_infos_[dev_id].totalGlobalMemory;
  *free = mem_infos_[dev_id].freeGlobalMemory;
  *used = mem_infos_[dev_id].usedGlobalMemory;
  return true;
}

bool XPUMLHandler::getL3UsageInfo(int dev_id, unsigned long long *total,
                                  unsigned long long *used, unsigned long long *free) {
  if(xpumlDeviceGetMemoryInfo(device_handlers_[dev_id], &mem_infos_[dev_id]) != xpumlReturn_enum::XPUML_SUCCESS) {
    return false;
  }
  *total = mem_infos_[dev_id].totalL3Memory;
  *free = mem_infos_[dev_id].freeL3Memory;
  *used = mem_infos_[dev_id].usedL3Memory;
  return true;
}

std::tuple<unsigned long long, unsigned long long, unsigned long long> XPUMLHandler::getMemoryUsageTuple(int dev_id) {
  if(xpumlDeviceGetMemoryInfo(device_handlers_[dev_id], &mem_infos_[dev_id]) != xpumlReturn_enum::XPUML_SUCCESS) {
    return {0, 0, 0};
  }
  return {mem_infos_[dev_id].totalGlobalMemory, 
          mem_infos_[dev_id].usedGlobalMemory, 
          mem_infos_[dev_id].freeGlobalMemory};

}

std::tuple<unsigned long long, unsigned long long, unsigned long long> XPUMLHandler::getL3UsageTuple(int dev_id) {
  if(xpumlDeviceGetMemoryInfo(device_handlers_[dev_id], &mem_infos_[dev_id]) != xpumlReturn_enum::XPUML_SUCCESS) {
    return {0, 0, 0};
  }
  return {mem_infos_[dev_id].totalL3Memory, 
          mem_infos_[dev_id].usedL3Memory, 
          mem_infos_[dev_id].freeL3Memory};  
}


// == Memory malloc & free ==



class RecordedXpuMallocHelper {
 private:
  explicit RecordedXpuMallocHelper(int dev_id, uint64_t limit_size = 0)
      : dev_id_(dev_id), limit_size_(limit_size) {
    if (NeedRecord()) {
      mtx_.reset(new std::mutex());
    }

    if (FLAGS_enable_xpu_memory_usage_log) {
      // A fake UPDATE to trigger the construction of memory stat instances,
      // make sure that they are destructed after RecordedXpuMallocHelper.
      DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id, 0);
      DEVICE_MEMORY_STAT_UPDATE(Allocated, dev_id, 0);
    }
  }

  DISABLE_COPY_AND_ASSIGN(RecordedXpuMallocHelper);

 public:
  ~RecordedXpuMallocHelper() {
    if (FLAGS_enable_xpu_memory_usage_log) {
      if (FLAGS_enable_xpu_memory_usage_log_mb) {
        std::cout << "[Memory Usage (MB)] gpu " << dev_id_ << " : Reserved = "
                  << DEVICE_MEMORY_STAT_PEAK_VALUE(Reserved, dev_id_) /
                         1048576.0
                  << ", Allocated = "
                  << DEVICE_MEMORY_STAT_PEAK_VALUE(Allocated, dev_id_) /
                         1048576.0
                  << std::endl;
      } else {
        std::cout << "[Memory Usage (Byte)] gpu " << dev_id_ << " : Reserved = "
                  << DEVICE_MEMORY_STAT_PEAK_VALUE(Reserved, dev_id_)
                  << ", Allocated = "
                  << DEVICE_MEMORY_STAT_PEAK_VALUE(Allocated, dev_id_)
                  << std::endl;
      }
    }
  }

  static RecordedXpuMallocHelper *Instance(int dev_id) {
    static std::vector<std::unique_ptr<RecordedXpuMallocHelper>> instances_;

    std::call_once(once_flag_, [] {
      int dev_cnt = GetXPUDeviceCount();
      instances_.reserve(dev_cnt);
      for (int i = 0; i < dev_cnt; ++i) {
        instances_.emplace_back(
            new RecordedXpuMallocHelper(i, FLAGS_xpu_memory_limit_mb << 20));
      }
    });

    PADDLE_ENFORCE_GE(
        dev_id,
        0,
        platform::errors::OutOfRange(
            "Device id must be not less than 0, but got %d.", dev_id));
    PADDLE_ENFORCE_LT(
        dev_id,
        instances_.size(),
        platform::errors::OutOfRange("Device id %d exceeds gpu card number %d.",
                                     dev_id,
                                     instances_.size()));
    return instances_[dev_id].get();
  }

  XPUError_t Malloc(void **ptr, size_t size, bool malloc_managed_memory = false) {
    // CHECK(malloc_managed_memory == false) << "xpu not supported yet";
    if (UNLIKELY(NeedRecord() && cur_size_.load() + size > limit_size_)) {
      return XPUERR_NOMEM;
    }
 
    XPUDeviceGuard guard(dev_id_);
    
    int result = xpu_malloc(ptr, size);
    VLOG(10) << "[xpu_malloc] size=" << static_cast<double>(size) / (1 << 20)
               << " MB, result=" << result;
    
    if (result == 0) {
      if (UNLIKELY(NeedRecord())) {
        cur_size_.fetch_add(size);
        DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, size);
        platform::RecordMemEvent(ptr,
                                XPUPlace(dev_id_),
                                size,
                                platform::TracerMemEventType::ReservedAllocate);      
      }

      return XPU_SUCCESS;
    } else {
      return XPUERR_NOMEM;
    }
  }

  void Free(void *ptr, size_t size) {
    XPUDeviceGuard guard(dev_id_);
    xpu_free(ptr);
    if (UNLIKELY(NeedRecord())) {
      cur_size_.fetch_sub(size);
      DEVICE_MEMORY_STAT_UPDATE(Reserved, dev_id_, -size);
      platform::RecordMemEvent(ptr,
                                XPUPlace(dev_id_),
                                size,
                                platform::TracerMemEventType::ReservedFree);
    }
  }

  bool GetMemInfo(size_t *avail,
                  size_t *total,
                  size_t *actual_avail,
                  size_t *actual_total) {
    unsigned long long uint64_total = 0, used = 0, free = 0;
    
    CHECK(ml_handler.getMemoryUsageInfo(dev_id_, &uint64_total, &used, &free) == true) << "get mem usage info failed";
    *actual_avail = uint64_total - free;
    *actual_total = uint64_total;

    if (UNLIKELY(NeedRecord())) {
      std::lock_guard<std::mutex> guard(*mtx_);
      *avail = std::min(*actual_avail, limit_size_ - cur_size_.load());
      *total = std::min(*actual_total, limit_size_);
      return *total < *actual_total;
    } else {
      *avail = *actual_avail;
      *total = *actual_total;
      return false;
    }
  }

  inline bool NeedRecord() const { return limit_size_ != 0; }

  uint64_t RecordedSize() const { return cur_size_.load(); }

  uint64_t LimitSize() const { return limit_size_; }


 private:
  const int dev_id_;
  const uint64_t limit_size_;
  std::atomic<uint64_t> cur_size_{0};

  mutable std::unique_ptr<std::mutex> mtx_;
  static std::once_flag once_flag_;

  XPUMLHandler ml_handler;
};

std::once_flag RecordedXpuMallocHelper::once_flag_;

XPUError_t RecordedXpuMalloc(void **ptr, size_t size, int dev_id, bool malloc_managed_memory) {
  return RecordedXpuMallocHelper::Instance(dev_id)->Malloc(ptr, size, malloc_managed_memory);
}

void RecordedXpuFree(void *p, size_t size, int dev_id) {
  return RecordedXpuMallocHelper::Instance(dev_id)->Free(p, size);
}

bool RecordedXpuMemGetInfo(size_t *avail,
                           size_t *total,
                           size_t *actual_avail,
                           size_t *actual_total,
                           int dev_id) {
  return RecordedXpuMallocHelper::Instance(dev_id)->GetMemInfo(avail, total, actual_avail, actual_total);
}

size_t XpuAvailableMemToAlloc() {
  XPUMLHandler handler;
  unsigned long long total = 0;
  unsigned long long used = 0;
  unsigned long long free = 0;
  bool re = handler.getMemoryUsageInfo(GetXPUCurrentDeviceId(), &total, &used, &free);
  CHECK(re == true) << "query mem info failed";

  size_t reserving = static_cast<size_t>(fraction_reserve_xpu_memory * free);
  // If available size is less than minimum chunk size, no usable memory exists
  size_t available_to_alloc = free - reserving;
  size_t min_chunk_size = XpuMinChunkSize();
  if (available_to_alloc < min_chunk_size) {
    available_to_alloc = 0;
  }
  return available_to_alloc;
}

static size_t XpuAllocSize(bool realloc) {
  size_t available_to_alloc = XpuAvailableMemToAlloc();
  PADDLE_ENFORCE_GT(
      available_to_alloc,
      0,
      platform::errors::ResourceExhausted("Not enough available XPU memory."));
  // If FLAGS_initial_gpu_memory_in_mb is 0, then initial memory will be
  // allocated by fraction
  size_t flag_mb = realloc ? FLAGS_reallocate_xpu_memory_in_mb
                           : FLAGS_initial_xpu_memory_in_mb;
  size_t alloc_bytes =
      (flag_mb > 0ul
           ? flag_mb << 20
           : available_to_alloc * FLAGS_fraction_of_xpu_memory_to_use);
  PADDLE_ENFORCE_GE(
      available_to_alloc,
      alloc_bytes,
      platform::errors::ResourceExhausted("Not enough available GPU memory."));
  VLOG(10) << "Alloc size is " << (alloc_bytes >> 20)
           << " MiB, is it Re-alloc: " << realloc;
  return alloc_bytes;
}


size_t XpuInitAllocSize() {
  return XpuAllocSize(false);
}

size_t XpuReallocSize() {
  return XpuAllocSize(true);
}

size_t XpuMaxAllocSize() {
  return std::max(XpuInitAllocSize(), XpuReallocSize());
}

size_t XpuMinChunkSize() {
  return 1 << 8;
}


size_t XpuMaxChunkSize() {

  size_t max_chunk_size = XpuMaxAllocSize();
  VLOG(10) << "Max chunk size " << (max_chunk_size >> 20) << "M";
  return max_chunk_size;

}

// for test 
class MallocCnter {
public:
  static MallocCnter & getInstance() {
    static MallocCnter instance;
    return instance;
  }

  void inc_malloc_cnt(int dev_id) {
    CHECK(dev_id >= 0 && dev_id < 8);
    malloc_cnts[dev_id]++;
  }

  int get_malloc_cnt(int dev_id) {
    CHECK(dev_id >= 0 && dev_id < 8);
    return malloc_cnts[dev_id].load();
  }

private:
  MallocCnter() {}
  std::atomic<int> malloc_cnts[8];
};

int get_malloc_cnt(int dev_id) {
  return MallocCnter::getInstance().get_malloc_cnt(dev_id);
}

int inc_malloc_cnt(int dev_id) {
  MallocCnter::getInstance().inc_malloc_cnt(dev_id);
  return 0;
}

}  // namespace platform
}  // namespace paddle
