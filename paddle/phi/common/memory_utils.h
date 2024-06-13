// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <future>  // NOLINT
#include <unordered_map>
#include "paddle/fluid/memory/malloc.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/stream.h"

namespace phi {

/*
  NOTE(YuanRisheng) Why should we add the following code?
  We need this because MemoryUtils::instance() is a singleton object and we
  don't recommend using singleton object in kernels. So, we wrap it using a
  function and if we delete this singleton object in future, it will be easy to
  change code.
*/

namespace memory_utils {
class Buffer {
 public:
  explicit Buffer(const phi::Place& place) : place_(place) {}

  template <typename T>
  T* Alloc(size_t size) {
    using AllocT = typename std::
        conditional<std::is_same<T, void>::value, uint8_t, T>::type;
    if (UNLIKELY(size == 0)) return nullptr;
    size *= sizeof(AllocT);
    if (allocation_ == nullptr || allocation_->size() < size) {
      allocation_ = paddle::memory::Alloc(place_, size);
    }
    return reinterpret_cast<T*>(allocation_->ptr());
  }

  template <typename T>
  const T* Get() const {
    return reinterpret_cast<const T*>(
        allocation_ && allocation_->size() > 0 ? allocation_->ptr() : nullptr);
  }

  template <typename T>
  T* GetMutable() {
    return reinterpret_cast<T*>(
        allocation_ && allocation_->size() > 0 ? allocation_->ptr() : nullptr);
  }

  size_t Size() const { return allocation_ ? allocation_->size() : 0; }

  phi::Place GetPlace() const { return place_; }

 private:
  Allocator::AllocationPtr allocation_;
  phi::Place place_;
};

template <typename StreamType>
struct ThrustAllocator {
  typedef char value_type;
  ThrustAllocator(phi::Place place, StreamType stream) {
    place_ = place;
    stream_ = stream;
  }
  ~ThrustAllocator() {}
  char* allocate(std::ptrdiff_t num_bytes) {
    auto storage =
        paddle::memory::AllocShared(place_,
                    num_bytes,
                    phi::Stream(reinterpret_cast<phi::StreamId>(stream_)));
    char* ptr = reinterpret_cast<char*>(storage->ptr());
    busy_allocation_.emplace(std::make_pair(ptr, storage));
    return ptr;
  }
  void deallocate(char* ptr, size_t) {
    allocation_map_type::iterator iter = busy_allocation_.find(ptr);
    // CHECK(iter != busy_allocation_.end());
    busy_allocation_.erase(iter);
  }

 private:
  typedef std::unordered_map<char*, std::shared_ptr<Allocation>>
      allocation_map_type;
  allocation_map_type busy_allocation_;
  phi::Place place_;
  StreamType stream_;
};

}  // namespace memory_utils

}  // namespace phi
