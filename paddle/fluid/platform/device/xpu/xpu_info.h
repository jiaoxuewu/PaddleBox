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
#pragma once

#ifdef PADDLE_WITH_XPU
#include <vector>
#include <mutex>

#include "paddle/fluid/platform/place.h"
#include "paddle/phi/backends/xpu/xpu_info.h"
#include "xpu/runtime.h"
#include "xpu/xpuml.h"

namespace paddle {

using xpuStream = XPUStream;
using xpuEventHandle = XPUEvent;

namespace platform {

/***** Version Management *****/

//! Get the version of XPU Driver
int GetDriverVersion();

//! Get the version of XPU Runtime
int GetRuntimeVersion();

/***** Device Management *****/

//! Get the total number of XPU devices in system.
int GetXPUDeviceCount();

//! Set the XPU device id for next execution.
void SetXPUDeviceId(int device_id);

//! Get the current XPU device id in system.
int GetXPUCurrentDeviceId();

//! Get a list of device ids from environment variable or use all.
std::vector<int> GetXPUSelectedDevices();

/***** Memory Management *****/

//! Copy memory from address src to dst synchronously.
void MemcpySyncH2D(void *dst,
                   const void *src,
                   size_t count,
                   const platform::XPUPlace &dst_place);
void MemcpySyncD2H(void *dst,
                   const void *src,
                   size_t count,
                   const platform::XPUPlace &src_place);
void MemcpySyncD2D(void *dst,
                   const platform::XPUPlace &dst_place,
                   const void *src,
                   const platform::XPUPlace &src_place,
                   size_t count);
void MemcpySyncH2D(void *dst,
                   const void *src,
                   size_t count,
                   const platform::XPUL3Place &dst_place);
void MemcpySyncD2H(void *dst,
                   const void *src,
                   size_t count,
                   const platform::XPUL3Place &src_place);
void MemcpySyncD2D(void *dst,
                   const platform::XPUL3Place &dst_place,
                   const void *src,
                   const platform::XPUL3Place &src_place,
                   size_t count);
void MemcpySyncD2D(void *dst,
                   const platform::XPUPlace &dst_place,
                   const void *src,
                   const platform::XPUL3Place &src_place,
                   size_t count);
void MemcpySyncD2D(void *dst,
                   const platform::XPUL3Place &dst_place,
                   const void *src,
                   const platform::XPUPlace &src_place,
                   size_t count);
void MemcpySyncH2D(void *dst,
                   const void *src,
                   size_t count,
                   const platform::Place &dst_place);
void MemcpySyncD2H(void *dst,
                   const void *src,
                   size_t count,
                   const platform::Place &src_place);
void MemcpySyncD2D(void *dst,
                   const platform::Place &dst_place,
                   const void *src,
                   const platform::Place &src_place,
                   size_t count);

//! Blocks until stream has completed all operations.
void XPUStreamSync(xpuStream stream);

using XPUDeviceGuard = phi::backends::xpu::XPUDeviceGuard;

phi::backends::xpu::XPUVersion get_xpu_version(int dev_id);

class XPUMLHandler {
public:
    XPUMLHandler();
    // total, used, free
    bool getMemoryUsageInfo(int dev_id, unsigned long long *total, unsigned long long* used, unsigned long long *free);
    bool getL3UsageInfo(int dev_id, unsigned long long *total, unsigned long long *used, unsigned long long *free);

    // (total, used, free)
    std::tuple<unsigned long long, unsigned long long, unsigned long long> getMemoryUsageTuple(int dev_id);
    std::tuple<unsigned long long, unsigned long long, unsigned long long> getL3UsageTuple(int dev_id);

private:
    static void init_ml();

    static std::once_flag init_flag_;

    std::vector<xpumlDevice_t> device_handlers_;
    std::vector<xpumlMemory_t> mem_infos_;
    unsigned int device_nums_;
};

XPUError_t RecordedXpuMalloc(void **ptr, size_t size, int dev_id, bool malloc_managed_memory = false);

void RecordedXpuFree(void *p, size_t size, int dev_id);

bool RecordedXpuMemGetInfo(size_t *avail,
                           size_t *total,
                           size_t *actual_avail,
                           size_t *actual_total,
                           int dev_id);

size_t XpuMinChunkSize();
size_t XpuMaxChunkSize();

size_t XpuInitAllocSize();
size_t XpuReallocSize();
size_t XpuMaxAllocSize();

// for calculate malloc times
int get_malloc_cnt(int dev_id);
int inc_malloc_cnt(int dev_id);

}  // namespace platform
}  // namespace paddle
#endif
