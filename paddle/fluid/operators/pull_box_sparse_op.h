//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/math_function.h"

DECLARE_bool(enable_pull_box_padding_zero);

namespace paddle {
namespace operators {

template <typename T>
static void PaddingZeros(const framework::ExecutionContext &ctx,
                         framework::LoDTensor *data, int batch_size,
                         int hidden_size) {
  // set data
  data->Resize({1, hidden_size});
  auto place = ctx.GetPlace();
  T *value_ptr = data->mutable_data<T>(place);
  if (platform::is_cpu_place(place)) {
    memset(value_ptr, 0, sizeof(T) * hidden_size);
  } else {
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    math::set_constant(dev_ctx, data, 0);
  }
  //  auto data_eigen = framework::EigenVector<T>::Flatten(*data);
  //  auto &place = *ctx.template device_context<platform::CUDADeviceContext>()
  //                     .eigen_device();
  //  data_eigen.device(place) = data_eigen.constant(static_cast<T>(0));

  // set lod
  std::vector<size_t> v_lod(batch_size + 1, 1);
  v_lod[0] = 0;
  paddle::framework::LoD data_lod;
  data_lod.push_back(v_lod);
  data->set_lod(data_lod);
}

template <typename T>
static void PullCacheValuesFunctor(const framework::ExecutionContext &ctx) {
#ifdef PADDLE_WITH_BOX_PS
  const auto *input = ctx.Input<framework::LoDTensor>("Id");
  auto *output = ctx.Output<framework::LoDTensor>("Out");

  auto batch_size = input->dims()[0];

  uint64_t *input_data = reinterpret_cast<uint64_t *>(
      const_cast<int64_t *>(input->data<int64_t>()));
  float *output_data =
      const_cast<float *>(output->mutable_data<float>(ctx.GetPlace()));

  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  int i = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace()).GetDeviceId();

  box_ptr->gpu_replica_cache.front().PullCacheValue(input_data, output_data,
                                                    batch_size, i);
#endif
}

template <typename T>
static void LookupInputFunctor(const framework::ExecutionContext &ctx) {
#ifdef PADDLE_WITH_BOX_PS
  const auto *input = ctx.Input<framework::LoDTensor>("Id");
  auto *output = ctx.Output<framework::LoDTensor>("Out");
  auto batch_size = input->dims()[0];
  uint64_t *input_data = reinterpret_cast<uint64_t *>(
      const_cast<int64_t *>(input->data<int64_t>()));
  float *output_data =
      const_cast<float *>(output->mutable_data<float>(ctx.GetPlace()));

  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  size_t device_id =
      BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace()).GetDeviceId();
  box_ptr->input_table_deque_.front().LookupInput(input_data, output_data,
                                                  batch_size, device_id);
#endif
}

template <typename T>
static void PullBoxSparseFunctor(const framework::ExecutionContext &ctx) {
  auto inputs = ctx.MultiInput<framework::LoDTensor>("Ids");
  auto outputs = ctx.MultiOutput<framework::LoDTensor>("Out");
  const auto slot_size = inputs.size();
  std::vector<const uint64_t *> all_keys(slot_size);
  // BoxPS only supports float now
  std::vector<float *> all_values(slot_size);
  std::vector<int64_t> slot_lengths(slot_size);
  auto hidden_size = ctx.Attr<int>("size");
  const int slot_idx = ctx.Attr<int>("slot_idx");
  // get batch size
  int batch_size = -1;
  if (slot_idx != -1) {
    if (slot_idx > static_cast<int>(slot_size)) {
      const auto *slot = inputs[0];
      batch_size =
          slot->lod().size() ? slot->lod()[0].size() - 1 : slot->dims()[0];
    } else {
      batch_size = inputs[slot_idx]->dims()[0];
    }
  } else {
    for (size_t i = 0; i < slot_size; ++i) {
      const auto *slot = inputs[i];
      if (slot->numel() == 0) {
        continue;
      }
      int cur_batch_size =
          slot->lod().size() ? slot->lod()[0].size() - 1 : slot->dims()[0];
      if (batch_size == -1) {
        batch_size = cur_batch_size;
      } else {
        PADDLE_ENFORCE_EQ(
            batch_size, cur_batch_size,
            platform::errors::PreconditionNotMet(
                "The batch size of all input slots should be same, "
                "please cheack"));
      }
    }
  }
  for (size_t i = 0; i < slot_size; ++i) {
    const auto *slot = inputs[i];
    auto *output = outputs[i];
    int64_t numel = slot->numel();
    if (numel == 0) {
      if (FLAGS_enable_pull_box_padding_zero) {
        // only support GPU
        PaddingZeros<T>(ctx, output, batch_size, hidden_size);
      }
      continue;
    }
    const uint64_t *single_slot_keys =
        reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
    all_keys[i] = single_slot_keys;
    slot_lengths[i] = numel;
    all_values[i] = output->mutable_data<T>(ctx.GetPlace());
  }

#ifdef PADDLE_WITH_BOX_PS
  int skip_offset = ctx.Attr<int>("offset");
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  auto expand_dim = box_ptr->GetExpandEmbedDim();
  box_ptr->PullSparse(ctx.GetPlace(), all_keys, all_values, slot_lengths,
                      hidden_size, expand_dim, skip_offset, true);
#endif
}

template <typename T>
static void PushBoxSparseFunctor(const framework::ExecutionContext &ctx) {
  auto inputs = ctx.MultiInput<framework::LoDTensor>("Ids");
  auto d_output =
      ctx.MultiInput<framework::Tensor>(framework::GradVarName("Out"));
  const auto slot_size = inputs.size();
  std::vector<const uint64_t *> all_keys(slot_size);
  std::vector<const float *> all_grad_values(slot_size);
  std::vector<int64_t> slot_lengths(slot_size);
  const int slot_idx = ctx.Attr<int>("slot_idx");

  int batch_size = -1;
  if (slot_idx != -1) {
    if (slot_idx > static_cast<int>(slot_size)) {
      batch_size = 1;
    } else {
      batch_size = inputs[slot_idx]->dims()[0];
    }
  }
  for (size_t i = 0; i < slot_size; i++) {
    const auto *slot = inputs[i];
    int64_t numel = slot->numel();
    if (numel == 0) continue;
    const uint64_t *single_slot_keys =
        reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
    all_keys[i] = single_slot_keys;
    slot_lengths[i] = numel;
    if (slot_idx == -1) {
      int cur_batch_size =
          slot->lod().size() ? slot->lod()[0].size() - 1 : slot->dims()[0];
      if (batch_size == -1) {
        batch_size = cur_batch_size;
      } else {
        PADDLE_ENFORCE_EQ(
            batch_size, cur_batch_size,
            platform::errors::PreconditionNotMet(
                "The batch size of all input slots should be same, "
                "please cheack"));
      }
    }
    const float *grad_value = d_output[i]->data<float>();
    all_grad_values[i] = grad_value;
  }

#ifdef PADDLE_WITH_BOX_PS
  auto hidden_size = ctx.Attr<int>("size");
  int skip_offset = ctx.Attr<int>("offset");
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  auto expand_dim = box_ptr->GetExpandEmbedDim();
  box_ptr->PushSparseGrad(ctx.GetPlace(), all_keys, all_grad_values,
                          slot_lengths, hidden_size, expand_dim, batch_size,
                          skip_offset, true);
#endif
}

using LoDTensor = framework::LoDTensor;
template <typename T>
class PullBoxSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PullBoxSparseFunctor<T>(ctx);
  }
};

using LoDTensor = framework::LoDTensor;
template <typename T>
class PullCacheValuesCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PullCacheValuesFunctor<T>(ctx);
  }
};

template <typename T>
class PushBoxSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushBoxSparseFunctor<T>(ctx);
  }
};

template <typename T>
class LookupInputCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    LookupInputFunctor<T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle
