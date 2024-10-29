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

#include "paddle/fluid/framework/fleet/box_sparse/heter_helper.h"

namespace paddle {
namespace framework {

template <typename KeyType, typename ValType, typename GradType>
void HeterHelper<KeyType, ValType, GradType>::build_ps() {

}

template <typename KeyType, typename ValType, typename GradType>
void HeterHelper<KeyType, ValType, GradType>::pull_sparse(int num,
        KeyType* d_keys, ValType* d_vals, size_t len) {

}

template <typename KeyType, typename ValType, typename GradType>
void HeterHelper<KeyType, ValType, GradType>::push_sparse(int num,
        KeyType* d_keys, ValType* d_vals, size_t len) {

}

template <typename KeyType, typename ValType, typename GradType>
void HeterHelper<KeyType, ValType, GradType>::end_pass() {

}

}  // end namespace framework
}  // end namespace paddle
#endif