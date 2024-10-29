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

#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"

namespace paddle {
namespace framework {

template <typename KeyType, typename ValType>
class XPUCacheArray {
 public:
  explicit XPUCacheArray(long long capacity, DevPlace& place) : capacity_(capacity), size_(0) {

 }

  virtual ~XPUCacheArray() {
  }

 private:
};
#endif

template <typename KeyType, typename ValType, typename OptimizerType>
class HashTable {
 public:
  explicit HashTable(size_t capacity, DevPlace& place);
  virtual ~HashTable();
  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;

  template <typename StreamType>
  void insert(const paddle::platform::Place& place, const KeyType* d_keys, const ValType* d_vals, size_t len,
              StreamType stream);

  template <typename StreamType>
  void get(const paddle::platform::Place& place, const KeyType* d_keys, ValType* d_vals, size_t len,
           StreamType stream);

  template <typename StreamType, typename OptimizerType>
  void update(const paddle::platform::Place& place, const KeyType* d_keys, size_t len,
              StreamType stream);

  template <typename StreamType>
  void dump_to_cpu(int devid, StreamType stream);

 private:
  XPUCacheArray<KeyType, ValType>* container_;
};

}  // end namespace framework
}  // end namespace paddle
