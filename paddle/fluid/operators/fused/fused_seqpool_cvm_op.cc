/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fused_seqpool_cvm_op.h"
#ifdef PADDLE_WITH_BOX_PS
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#else
#include "paddle/fluid/framework/threadpool.h"
#endif
#include <string>
namespace paddle {
namespace operators {

class FusedSeqpoolCVMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("X").size(), 1UL,
                      "Inputs(X) of FusedSeqpoolCVMOp should not be empty.");
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(), 1UL,
                      "Outputs(Out) of FusedSeqpoolCVMOp should not be empty.");

    auto cvm_dims = ctx->GetInputDim("CVM");
    PADDLE_ENFORCE_EQ(
        cvm_dims.size(), 2UL,
        platform::errors::InvalidArgument("Input(CVM)'s rank should be 2."));
    PADDLE_ENFORCE_EQ(cvm_dims[1], 2UL, platform::errors::InvalidArgument(
                                            "The 2nd dimension of "
                                          "Input(CVM) should be 2."));

    auto ins_dims = ctx->GetInputsDim("X");
    const int cvm_offset = ctx->Attrs().Get<int>("cvm_offset");
    const size_t num_inputs = ins_dims.size();
    std::vector<framework::DDim> outs_dims;
    outs_dims.resize(num_inputs);
    bool use_cvm = ctx->Attrs().Get<bool>("use_cvm");
    bool clk_filter = ctx->Attrs().Get<bool>("clk_filter");
    const int embed_thres_size = ctx->Attrs().Get<int>("embed_thres_size");
    const int embedx_concate_size = ctx->Attrs().Get<int>("embedx_concate_size");
    //const bool fill_zero = ctx->Attrs().Get<bool>("fill_zero");

    // need filter quant_ratio more than zero
    if (ctx->Attrs().Get<bool>("need_filter")) {
      const int quant_ratio = ctx->Attrs().Get<int>("quant_ratio");
      PADDLE_ENFORCE_GT(
          quant_ratio, 0,
                      platform::errors::InvalidArgument(
              "Input need filter quant_ratio should be greater than 0"));
    }

    PADDLE_ENFORCE_GT(num_inputs, 0UL,
                      platform::errors::InvalidArgument(
                          "Input tensors count should be greater than 0, "
                          "but received value is %d.",
                          num_inputs));

    // The output height should be confirmed in Compute,
    // since input lod is not accessible here.
    PADDLE_ENFORCE_EQ(ins_dims[0].size(), 2,
                      platform::errors::InvalidArgument(
                          "The dims size of first input should be equal to 2, "
                          "but received value is %d.",
                          ins_dims[0].size()));

    for (size_t i = 0; i < num_inputs; ++i) {
      const auto dims = ins_dims[i];
      int rank = dims.size();
      if (use_cvm) {
        PADDLE_ENFORCE_GT(
            dims[rank - 1], 2,
                "Shape error in %lu id, the last dimension(embedding) of the "
                "'X' tensor must be larger than 2.",
            i);
      }
      // input lod is not accessible here
      std::vector<int64_t> out_dim;
      if (use_cvm) {
        if (clk_filter) {
          out_dim = {-1, (dims[rank - 1] - 1) * embedx_concate_size};
        } else {
          out_dim = {-1, dims[rank - 1] * embedx_concate_size};
        }
      } else {
        out_dim = {-1, (dims[rank - 1] - cvm_offset - embed_thres_size) * embedx_concate_size};
      }
      outs_dims[i] = phi::make_ddim(out_dim);
    }
    ctx->SetOutputsDim("Out", outs_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.device_context());
      }
};

class FusedSeqpoolCVMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(vector<LoDTensor>) The input tensors of"
             " operator.")
        .AsDuplicable();
    AddInput("CVM",
             "(Tensor),  a 2-D Tensor with shape [N x 2], where N is the batch "
             "size, 2 is show and click.");
    AddOutput("Out",
              "(vector<Tensor>) The output of Op does not contain LoD "
              "information.")
        .AsDuplicable();
    AddAttr<std::string>("pooltype",
                         "(string, default 'SUM') the pooling pooltype of "
                         "SequencePoolOp, only support SUM now.")
        .SetDefault("SUM")
        .InEnum({"SUM"});
    AddAttr<float>("pad_value",
                   "(float, default 0.0) The value to pad for empty sequence.")
        .SetDefault(0.0);
    AddAttr<bool>("use_cvm", "bool, use cvm or not").SetDefault(true);
    AddAttr<bool>("need_filter", "(bool, default false)").SetDefault(false);
    AddAttr<bool>("embed_threshold_filter", "(bool, default false)")
        .SetDefault(false);
    AddAttr<float>("show_coeff", "(float, default 0.2)").SetDefault(0.2);
    AddAttr<float>("clk_coeff", "(float, default 1)").SetDefault(1);
    AddAttr<float>("threshold", "(float, default 0.96)").SetDefault(0.96);
    AddAttr<float>("embed_threshold", "(float, default 0)").SetDefault(0);
    AddAttr<int>("cvm_offset", "(int, default 2)").SetDefault(2);
    AddAttr<int>("quant_ratio", "(int, default 128)").SetDefault(0);
    AddAttr<bool>("clk_filter", "(bool, default false)").SetDefault(false);
    AddAttr<int>("embed_thres_size", "(int, default 0)").SetDefault(0);
    AddAttr<int>("embedx_concate_size", "(int, default 1)").SetDefault(1);
    AddAttr<bool>("embedx_concate_filter", "(bool, default false)").SetDefault(false);
    AddAttr<bool>("fill_zero", "(bool, default true)").SetDefault(true);
    AddAttr<bool>("fix_ctr_to_click", "(bool, default false)").SetDefault(false);

    AddComment(R"DOC(
Fuse multiple pairs of Sequence Pool and CVM Operator.

)DOC");
  }
};

class FusedSeqpoolCVMGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto og_dims = ctx->GetInputsDim(framework::GradVarName("Out"));
    auto x_dims = ctx->GetInputsDim("X");
    auto cvm_dims = ctx->GetInputDim("CVM");
    const int cvm_offset = ctx->Attrs().Get<int>("cvm_offset");
    bool use_cvm = ctx->Attrs().Get<bool>("use_cvm");
    bool clk_filter = ctx->Attrs().Get<bool>("clk_filter");
    const int embed_thres_size = ctx->Attrs().Get<int>("embed_thres_size");
    const int embedx_concate_size = ctx->Attrs().Get<int>("embedx_concate_size");

    PADDLE_ENFORCE_EQ(
        cvm_dims.size(), 2,
        platform::errors::InvalidArgument("Input(CVM)'s rank should be 2."));

    for (size_t i = 0; i < og_dims.size(); i++) {
      PADDLE_ENFORCE_EQ(
          og_dims[i].size(), x_dims[i].size(),
          platform::errors::InvalidArgument(
              "The rank of output grad must equal to Input(X). But "
              "received: input rank %u, input shape [%s].",
              og_dims[i].size(), og_dims[i]));
      if (use_cvm) {
        auto o_dim = og_dims[i][og_dims[i].size() - 1];
        if (clk_filter) {  // filter clk need + 1
          o_dim = o_dim / embedx_concate_size + 1;
        } else {
          o_dim = o_dim / embedx_concate_size;
        }
        PADDLE_ENFORCE_EQ(
            o_dim, x_dims[i][og_dims[i].size() - 1],
            platform::errors::InvalidArgument(
                "The dimension mismatch between Input(OUT@GRAD) and "
                "Input(X). Received Input(OUT@GRAD): input rank %u, "
                "input shape [%s]; received Input(X): input rank %u, "
                "input shape [%s].",
                og_dims[i].size(), og_dims[i], x_dims[i].size(), x_dims[i]));
      } else {
        PADDLE_ENFORCE_EQ(
            og_dims[i][og_dims[i].size() - 1],
            (x_dims[i][og_dims[i].size() - 1] - cvm_offset - embed_thres_size) * embedx_concate_size,
            platform::errors::InvalidArgument(
                "The dimension mismatch between Input(OUT@GRAD) and "
                "Input(X). Received Input(OUT@GRAD): input rank %u, "
                "input shape [%s]; received Input(X): input rank %u, "
                "input shape [%s].",
                og_dims[i].size(), og_dims[i], x_dims[i].size(), x_dims[i]));
      }
    }
    for (size_t i = 0; i < x_dims.size(); ++i) {
      ctx->ShareLoD("X", framework::GradVarName("X"), i, i);
      ctx->ShareDim("X", framework::GradVarName("X"), i, i);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class FusedSeqpoolCVMGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("fused_seqpool_cvm_grad");
    op_desc_ptr->SetInput("X", this->Input("X"));
    op_desc_ptr->SetInput("CVM", this->Input("CVM"));

    op_desc_ptr->SetInput(framework::GradVarName("Out"),
                          this->OutputGrad("Out"));
    op_desc_ptr->SetOutput(framework::GradVarName("X"),
                           this->InputGrad("X", false));
    op_desc_ptr->SetOutput(framework::GradVarName("CVM"),
                           this->InputGrad("CVM"));
    op_desc_ptr->SetAttrMap(this->Attrs());
  }
};

using LoDTensor = framework::LoDTensor;

#define CHECK_USE_QUANT  \
    if (need_filter && quant_ratio > 0) { \
     auto &show = input_data[k * embedding_size]; \
     auto &click = input_data[k * embedding_size + 1]; \
     if ((show - click) * show_coeff + click * clk_coeff < threshold) { \
        continue; \
     } \
    }

#define QUANT_VALUE(val)  \
    (static_cast<int>(val * quant_ratio + 0.5) / static_cast<float>(quant_ratio));

#define CHECK_QUANT_AND_GETVAL(in_val)  \
    if (quant_ratio > 0) {  \
      if (need_filter) { \
        auto &show = input_data[k * embedding_size]; \
        auto &click = input_data[k * embedding_size + 1]; \
        if ((show - click) * show_coeff + click * clk_coeff < threshold) {  \
           continue;  \
        }  \
      } \
      val += QUANT_VALUE(in_val); \
    } else { \
      val += in_val; \
    }

template <typename T>
class FusedSeqpoolCVMOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto inputs = ctx.MultiInput<LoDTensor>("X");
    auto outputs = ctx.MultiOutput<framework::Tensor>("Out");

    const auto slot_size = inputs.size();
    auto padding_value = ctx.Attr<float>("pad_value");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    bool need_filter = ctx.Attr<bool>("need_filter");
    float show_coeff = ctx.Attr<float>("show_coeff");
    float clk_coeff = ctx.Attr<float>("clk_coeff");
    float threshold = ctx.Attr<float>("threshold");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    const int quant_ratio = ctx.Attr<int>("quant_ratio");
    bool clk_filter = ctx.Attr<bool>("clk_filter");

    auto place = ctx.GetPlace();

    int batch_size = -1;
    int embedding_size = inputs[0]->numel() / inputs[0]->dims()[0];
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
    box_ptr->ExecuteFunc(place, slot_size, [&](const size_t &i) {
#else
    paddle::framework::parallel_run_dynamic(slot_size, [&](const size_t &i) {
#endif
      const auto *input = inputs[i];
      CHECK(input->lod().size() == 1);

      auto lod_data = input->lod()[0];
      int cur_batch = lod_data.size() - 1;
      if (batch_size == -1) {
        batch_size = cur_batch;
      } else {
        CHECK(batch_size == cur_batch) << "batch: " << batch_size
            << ", current: " << cur_batch;
      }

      const T *input_data = reinterpret_cast<const T*>(input->data<T>());
      auto *output = outputs[i];
      if (use_cvm) {
        if (clk_filter) {
          int dim_size = embedding_size - 1;
          output->Resize({cur_batch, dim_size});
          T *out_data = reinterpret_cast<T*>(output->mutable_data<T>(place));
          // ins
          for (int j = 0; j < cur_batch; ++j) {
            auto &start = lod_data[j];
            auto &end = lod_data[j + 1];
            // show, embed, embedx
            for (int dimx = 0; dimx < dim_size; ++dimx) {
              double val = padding_value;
              if (dimx == 0) { // show
                for (auto k = start; k < end; ++k) {
                  CHECK_USE_QUANT;
                  val += input_data[k * embedding_size + dimx];
                }
                out_data[j * dim_size + dimx] = log(val + 1);
              } else {
                for (auto k = start; k < end; ++k) {
                  CHECK_QUANT_AND_GETVAL(input_data[k * embedding_size + 1 + dimx]);
                }
                out_data[j * dim_size + dimx] = val;
              }
            }
          }
        } else {
          output->Resize({cur_batch, embedding_size});
          T *out_data = reinterpret_cast<T*>(output->mutable_data<T>(place));
          // ins
          for (int j = 0; j < cur_batch; ++j) {
            auto &start = lod_data[j];
            auto &end = lod_data[j + 1];
            // show, click, embed, embedx
            for (int dimx = 0; dimx < embedding_size; ++dimx) {
              double val = padding_value;
              if (dimx == 0) { // show
                for (auto k = start; k < end; ++k) {
                  CHECK_USE_QUANT;
                  val += input_data[k * embedding_size + dimx];
                }
                out_data[j * embedding_size + dimx] = log(val + 1);
              } else if (dimx == 1) { // ctr log(click) - log(show)
                for (auto k = start; k < end; ++k) {
                  CHECK_USE_QUANT;
                  val += input_data[k * embedding_size + dimx];
                }
                out_data[j * embedding_size + dimx] = log(val + 1)
                    - out_data[j * embedding_size];
              } else {
                for (auto k = start; k < end; ++k) {
                  CHECK_QUANT_AND_GETVAL(input_data[k * embedding_size + dimx]);
                }
                out_data[j * embedding_size + dimx] = val;
              }
            }
          }
        }
      } else {
        int dim_size = embedding_size - cvm_offset;
        output->Resize({cur_batch, dim_size});
        // no cvm
        T *out_data = reinterpret_cast<T*>(output->mutable_data<T>(place));
        // ins
        for (int j = 0; j < cur_batch; ++j) {
          auto &start = lod_data[j];
          auto &end = lod_data[j + 1];
          // show, click, embed, embedx
          for (int dimx = 0; dimx < dim_size; ++dimx) {
            double val = padding_value;
            for (auto k = start; k < end; ++k) {
              CHECK_QUANT_AND_GETVAL(input_data[k * embedding_size + cvm_offset + dimx]);
            }
            out_data[j * dim_size + dimx] = val;
          }
        }
      }
    });
  }
};

template <typename T>
class FusedSeqpoolCVMGradOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto out_grads = ctx.MultiInput<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
    auto *cvm = ctx.Input<LoDTensor>("CVM");

    std::string pooltype = ctx.Attr<std::string>("pooltype");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    bool clk_filter = ctx.Attr<bool>("clk_filter");

    const auto slot_size = in_grads.size();
    auto place = ctx.GetPlace();

    int embedding_size = in_grads[0]->numel() / in_grads[0]->dims()[0];
    int dim_size = embedding_size;
    int dim_off = 0;
    if (use_cvm) {
      if (clk_filter) {
        dim_size = embedding_size - 1;
        dim_off = 1;
      }
    } else {
      dim_size = embedding_size - cvm_offset;
      dim_off = cvm_offset;
    }
    int batch_size = -1;
#ifdef PADDLE_WITH_BOX_PS
    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
    box_ptr->ExecuteFunc(place, slot_size, [&](const size_t i) {
#else
    paddle::framework::parallel_run_dynamic(slot_size, [&](const size_t &i) {
#endif
//    for (size_t i = 0; i < slot_size; ++i) {
      auto *in_grad = in_grads[i];

      auto lod_data = in_grad->lod()[0];
      int cur_batch = lod_data.size() - 1;
      if (batch_size == -1) {
        batch_size = cur_batch;
      } else {
        CHECK(batch_size == cur_batch) << "batch: " << batch_size
            << ", current: " << cur_batch;
      }

      const T *cvm_data = reinterpret_cast<const T*>(cvm->data<T>());
      const T *out_grads_value =
          reinterpret_cast<const T*>(out_grads[i]->data<T>());
      T *in_grads_value = reinterpret_cast<T*>(in_grad->mutable_data<T>(place));

      for (int j = 0; j < cur_batch; ++j) {
        auto &start = lod_data[j];
        auto &end = lod_data[j + 1];
        for (int dim_id = 0; dim_id < embedding_size; ++dim_id) {
          const T &val = (dim_id < cvm_offset) ?
              cvm_data[j * cvm_offset + dim_id] :
                  out_grads_value[j * dim_size + dim_id - dim_off];
          for (auto k = start; k < end; ++k) {
            in_grads_value[k * embedding_size + dim_id] = val;
          }
        }
      }
    });
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(fused_seqpool_cvm, ops::FusedSeqpoolCVMOp,
                  ops::FusedSeqpoolCVMOpMaker,
                  ops::FusedSeqpoolCVMGradOpMaker<paddle::framework::OpDesc>,
                  ops::FusedSeqpoolCVMGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_seqpool_cvm_grad, ops::FusedSeqpoolCVMGradOp)

REGISTER_OP_CPU_KERNEL(fused_seqpool_cvm,
                       ops::FusedSeqpoolCVMOpCPUKernel<float>)
REGISTER_OP_CPU_KERNEL(fused_seqpool_cvm_grad,
                       ops::FusedSeqpoolCVMGradOpCPUKernel<float>)
