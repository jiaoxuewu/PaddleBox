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
#include "paddle/fluid/framework/fleet/heter_ps/hashtable.h"

template <typename KeyType, typename ValType, typename GradType>
class HeterHelper {
public:
  void build_ps(int num, KeyType* h_keys, ValType* h_vals, size_t len,
                size_t chunk_size, int stream_num);
  void pull_sparse(int num, KeyType* d_keys, ValType* d_vals, size_t len);
  void push_sparse(int num, KeyType* d_keys, GradType* d_grads, size_t len);
  void end_pass();
protected:
  using Table = HashTable<KeyType, ValType>;
  std::vector<Table*> tables_;
}