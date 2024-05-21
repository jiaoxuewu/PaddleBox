//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace operators {

template<typename T>
static void PullBoxExtendedSparseFunctor(
    const framework::ExecutionContext &ctx) {
  auto inputs = ctx.MultiInput<framework::Tensor>("Ids");
  auto outputs = ctx.MultiOutput<framework::Tensor>("Out");
  auto outputs_extend = ctx.MultiOutput<framework::Tensor>("OutExtend");
  auto flags = ctx.Attr<std::vector<int>>("mask");

  const auto slot_size = inputs.size();
  std::vector<const uint64_t*> all_keys(slot_size);

  int total_dims0 = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    total_dims0 += outputs[i]->dims()[0];
  }
  int total_expand_dims0 = 0;
  for (size_t i = 0; i < outputs_extend.size(); ++i) {
    total_expand_dims0 += outputs_extend[i]->dims()[0];
  }

  int max_total_dims0 = total_dims0;
  bool is_expand_slot_small = true;
  if(total_dims0>total_expand_dims0) {
    is_expand_slot_small = true;
    max_total_dims0 = total_dims0;
  } else {
    is_expand_slot_small = false;
    max_total_dims0 = total_expand_dims0;
  }

  std::vector<int> slot_dims0_offset(slot_size);
  int offset = 0;
  int dims1 = 0;
  int expand_dims1 = 0;

  size_t embedx_offset = 0;
  size_t expand_offset = 0;
  for (int i = 0; i < (int)slot_size; i++) {
    slot_dims0_offset[i] = offset;
    if(flags.empty()) {
      offset += outputs[i]->dims()[0];
    } else {
      if(is_expand_slot_small){
        if (flags[i] & 0x01) {
          offset += outputs[embedx_offset]->dims()[0];
          dims1 = outputs[embedx_offset]->dims()[1];
          embedx_offset++;
        } else {
          offset += 0;
        }
        if(flags[i] & 0x02) {
          expand_dims1 = outputs_extend[expand_offset]->dims()[1];
          expand_offset++;
        }
      } else {
        if (flags[i] & 0x02) {
          offset += outputs_extend[expand_offset]->dims()[0];
          expand_dims1 = outputs_extend[expand_offset]->dims()[1];
          expand_offset++;
        } else {
          offset += 0;
        }
        if(flags[i] & 0x01) {
          dims1 = outputs[embedx_offset]->dims()[1];
          embedx_offset++;
        }
      }
    }
  }

  framework::LoDTensor total_values;
  total_values.Resize(phi::make_ddim({max_total_dims0 * (dims1 + expand_dims1)}));
  total_values.mutable_data<T>(ctx.GetPlace());

  // BoxPS only supports float now
  std::vector<float*> all_values(slot_size * 2);
  std::vector<int64_t> slot_lengths(slot_size);
  if (flags.empty()) {
    for (size_t i = 0; i < slot_size; i++) {
      const auto *slot = inputs[i];
      const uint64_t *single_slot_keys =
          reinterpret_cast<const uint64_t*>(slot->data<int64_t>());
      all_keys[i] = single_slot_keys;
      slot_lengths[i] = slot->numel();
      if (outputs[i]->numel() == 0) {
        outputs[i]->set_layout(paddle::framework::DataLayout::UNDEFINED);
      } else {
        size_t offset = slot_dims0_offset[i] * dims1 * sizeof(T);
        total_values.set_offset(offset);
        outputs[i]->ShareBufferWith(total_values);
      }
      auto *output = outputs[i]->mutable_data<T>(ctx.GetPlace());
      all_values[i] = reinterpret_cast<float*>(output);
      if(outputs_extend[i]->numel() == 0) {
        outputs_extend[i]->set_layout(paddle::framework::DataLayout::UNDEFINED);
      } else {
        size_t offset = slot_dims0_offset[i] * expand_dims1 * sizeof(T);
        total_values.set_offset(max_total_dims0 * dims1 * sizeof(T) + offset);
        outputs_extend[i]->ShareBufferWith(total_values);
      }
      auto *output_extend = outputs_extend[i]->mutable_data<T>(ctx.GetPlace());
      all_values[i + slot_size] = reinterpret_cast<float*>(output_extend);
    }
  } else {
    size_t embedx_offset = 0;
    size_t expand_offset = 0;
    for (size_t i = 0; i < slot_size; i++) {
      const auto *slot = inputs[i];
      const uint64_t *single_slot_keys =
          reinterpret_cast<const uint64_t*>(slot->data<int64_t>());
      all_keys[i] = single_slot_keys;
      slot_lengths[i] = slot->numel();
      if (flags[i] & 0x01) {
        if (outputs[embedx_offset]->numel() == 0) {
          outputs[embedx_offset]->set_layout(paddle::framework::DataLayout::UNDEFINED);
        } else {
          size_t offset = slot_dims0_offset[i] * dims1 * sizeof(T);
          total_values.set_offset(offset);
          outputs[embedx_offset]->ShareBufferWith(total_values);
        }
        auto *output = outputs[embedx_offset]->mutable_data<T>(ctx.GetPlace());
        all_values[i] = reinterpret_cast<float*>(output);
        if (outputs[embedx_offset]->numel() == 0) {
          all_values[i] = nullptr;
        }
        ++embedx_offset;
      } else {
        all_values[i] = 0;
      }
      if (flags[i] & 0x02) {
        if(outputs_extend[expand_offset]->numel()==0) {
          outputs_extend[expand_offset]->set_layout(paddle::framework::DataLayout::UNDEFINED);
        } else {
          size_t offset = slot_dims0_offset[i] * expand_dims1 * sizeof(T);
          total_values.set_offset(max_total_dims0 * dims1 * sizeof(T) + offset);
          outputs_extend[expand_offset]->ShareBufferWith(total_values);
        }
        auto *output_extend = outputs_extend[expand_offset]->mutable_data<T>(ctx.GetPlace());
        all_values[i + slot_size] = reinterpret_cast<float*>(output_extend);
        if(outputs_extend[expand_offset]->numel() == 0) {
          all_values[i] = nullptr;
        }
        ++expand_offset;
      } else {
        all_values[i + slot_size] = 0;
      }
    }
  }
  total_values.set_offset(0);
#ifdef PADDLE_WITH_BOX_PS
  int skip_offset = ctx.Attr<int>("offset");
  auto emb_size = ctx.Attr<int>("emb_size");
  auto emb_extended_size = ctx.Attr<int>("emb_extended_size");
  auto expand_only = ctx.Attr<bool>("expand_only");
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  box_ptr->PullSparse(ctx.GetPlace(), all_keys, all_values, slot_lengths,
                      emb_size, emb_extended_size, skip_offset, expand_only);
  if (std::getenv("DUMP_XPU_PUSH_SPARSE_INPUT") != nullptr) {
    auto names = ctx.OutputNames("Out");
    for (int i = 0; i <int(outputs.size()); i++) {
      TensorFormatter formatter;
      const std::string &name = names[i];
      formatter.SetPrintTensorType(true);
      formatter.SetPrintTensorShape(true);
      formatter.SetPrintTensorLod(true);
      formatter.SetPrintTensorLayout(true);
      // formatter.SetSummarize(static_cast<int64_t>(Attr<int>("summarize")));
      // formatter.SetPrintFilePath("dev"+std::to_string(ctx.GetPlace().device)+".push_sparse_input.txt");
      std::string message = std::string("---embs_all_")+std::to_string(i)+std::string("---");
      formatter.Print(*(outputs[i]), name, message);
    }
    names = ctx.OutputNames("OutExtend");
    for (int i = 0; i <int(outputs_extend.size()); i++) {
      TensorFormatter formatter;
      const std::string &name = names[i];
      formatter.SetPrintTensorType(true);
      formatter.SetPrintTensorShape(true);
      formatter.SetPrintTensorLod(true);
      formatter.SetPrintTensorLayout(true);
      // formatter.SetSummarize(static_cast<int64_t>(Attr<int>("summarize")));
      // formatter.SetPrintFilePath("dev"+std::to_string(ctx.GetPlace().device)+".push_sparse_input.txt");
      std::string message = std::string("---expand_all_")+std::to_string(i)+std::string("---");
      formatter.Print(*(outputs_extend[i]), name, message);
    }
  }
#endif
}

template<typename T>
static void PushBoxExtendedSparseFunctor(
    const framework::ExecutionContext &ctx) {
  auto inputs = ctx.MultiInput<framework::LoDTensor>("Ids");
  auto d_output = ctx.MultiInput<framework::Tensor>(
      framework::GradVarName("Out"));
  auto d_output_extend = ctx.MultiInput<framework::Tensor>(
      framework::GradVarName("OutExtend"));
  auto flags = ctx.Attr<std::vector<int>>("mask");

  const auto slot_size = inputs.size();
  std::vector<const uint64_t*> all_keys(slot_size);
  std::vector<const float*> all_grad_values(slot_size * 2);
  std::vector<int64_t> slot_lengths(slot_size);
  int batch_size = -1;

  if (flags.empty()) {
    for (size_t i = 0; i < slot_size; i++) {
      const auto *slot = inputs[i];
      const uint64_t *single_slot_keys =
          reinterpret_cast<const uint64_t*>(slot->data<int64_t>());
      all_keys[i] = single_slot_keys;
      slot_lengths[i] = slot->numel();
      int cur_batch_size =
          slot->lod().size() ? slot->lod()[0].size() - 1 : slot->dims()[0];
      if (batch_size == -1) {
        batch_size = cur_batch_size;
      } else {
        PADDLE_ENFORCE_EQ(batch_size, cur_batch_size,
            platform::errors::PreconditionNotMet(
                "The batch size of all input slots should be same,"
                    "please cheack"));
      }
      const float *grad_value = d_output[i]->data<float>();
      all_grad_values[i] = reinterpret_cast<const float*>(grad_value);
      const float *grad_value_extend = d_output_extend[i]->data<float>();
      all_grad_values[i + slot_size] =
          reinterpret_cast<const float*>(grad_value_extend);
    }
  } else {
    int embedx_offset = 0;
    int expand_offset = 0;
    for (size_t i = 0; i < slot_size; i++) {
      const auto *slot = inputs[i];
      const uint64_t *single_slot_keys =
          reinterpret_cast<const uint64_t*>(slot->data<int64_t>());
      all_keys[i] = single_slot_keys;
      slot_lengths[i] = slot->numel();
      int cur_batch_size =
          slot->lod().size() ? slot->lod()[0].size() - 1 : slot->dims()[0];
      if (batch_size == -1) {
        batch_size = cur_batch_size;
      } else {
        PADDLE_ENFORCE_EQ(batch_size, cur_batch_size,
            platform::errors::PreconditionNotMet(
                "The batch size of all input slots should be same,"
                    "please cheack"));
      }
      if (flags[i] & 0x01) {
        const float *grad_value = d_output[embedx_offset]->data<float>();
        all_grad_values[i] = reinterpret_cast<const float*>(grad_value);
        ++embedx_offset;
      } else {
        all_grad_values[i] = 0;
      }
      if (flags[i] & 0x02) {
        const float *grad_value_extend = d_output_extend[expand_offset]->data<
            float>();
        all_grad_values[i + slot_size] =
            reinterpret_cast<const float*>(grad_value_extend);
        ++expand_offset;
      } else {
        all_grad_values[i + slot_size] = 0;
      }
    }
  }
#ifdef PADDLE_WITH_BOX_PS
  int skip_offset = ctx.Attr<int>("offset");
  auto emb_size = ctx.Attr<int>("emb_size");
  auto emb_extended_size = ctx.Attr<int>("emb_extended_size");
  auto expand_only = ctx.Attr<bool>("expand_only");
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  if (std::getenv("DUMP_XPU_PUSH_SPARSE_INPUT") != nullptr) {
    auto names = ctx.InputNames(framework::GradVarName("OutExtend"));
    for (int i = (d_output_extend.size()-1); i >=0; i--) {
      TensorFormatter formatter;
      const std::string &name = names[i];
      formatter.SetPrintTensorType(true);
      formatter.SetPrintTensorShape(true);
      formatter.SetPrintTensorLod(true);
      formatter.SetPrintTensorLayout(true);
      // formatter.SetSummarize(static_cast<int64_t>(Attr<int>("summarize")));
      // formatter.SetPrintFilePath("dev"+std::to_string(ctx.GetPlace().device)+".push_sparse_input.txt");
      std::string message = std::string("---expand_all_")+std::to_string(i)+std::string("---");
      formatter.Print(*(d_output_extend[i]), "print_" + name, message);
    }

    names = ctx.InputNames(framework::GradVarName("Out"));
    for (int i = (d_output.size()-1); i >=0; i--) {
      TensorFormatter formatter;
      const std::string &name = names[i];
      formatter.SetPrintTensorType(true);
      formatter.SetPrintTensorShape(true);
      formatter.SetPrintTensorLod(true);
      formatter.SetPrintTensorLayout(true);
      // formatter.SetSummarize(static_cast<int64_t>(Attr<int>("summarize")));
      // formatter.SetPrintFilePath("dev"+std::to_string(ctx.GetPlace().device)+".push_sparse_input.txt");
      std::string message = std::string("---embs_all_")+std::to_string(i)+std::string("---");
      formatter.Print(*(d_output[i]), "print_" + name, message);
    }
  }
  box_ptr->PushSparseGrad(ctx.GetPlace(), all_keys, all_grad_values,
                          slot_lengths, emb_size, emb_extended_size, batch_size,
                          skip_offset, expand_only);
#endif
}

using LoDTensor = framework::LoDTensor;
template<typename T>
class PullBoxExtendedSparseCPUKernel: public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PullBoxExtendedSparseFunctor<T>(ctx);
  }
};

template<typename T>
class PushBoxExtendedSparseCPUKernel: public framework::OpKernel<T> {
public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushBoxExtendedSparseFunctor<T>(ctx);
  }
};

}  // namespace operators
}  // namespace paddle
