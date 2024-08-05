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

#include <string>

#include "paddle/fluid/operators/fused/fused_seqpool_cvm_with_diff_thres_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

#define GET_BLOCK(N) \
  ((N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS)

#define CUDA_KERNEL_LOOP(i, n)                                  \
  for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// normal
template <typename T>
__global__ void FusedSeqpoolKernelNormal(const size_t N,
                                         T **input_values,
                                         T *seqpool_output_values,
                                         const size_t *lods_values,
                                         const int batch_size,
                                         const int embedding_size,
                                         const float pad_value) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id

    auto &start = lods_values[x * (batch_size + 1) + y];
    auto &end = lods_values[x * (batch_size + 1) + y + 1];
    double val = pad_value;
    for (auto k = start; k < end; ++k) {
      val += *(input_values[x] + k * embedding_size + offset);
    }
    seqpool_output_values[i] = val;
  }
}
// not need filter quant
template <typename T>
__global__ void FusedSeqpoolKernelQuant(const size_t N,
                                        T **input_values,
                                        T *seqpool_output_values,
                                        const size_t *lods_values,
                                        const int batch_size,
                                        const int embedding_size,
                                        const float pad_value,
                                        const int cvm_offset,
                                        const int quant_ratio) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id

    auto &start = lods_values[x * (batch_size + 1) + y];
    auto &end = lods_values[x * (batch_size + 1) + y + 1];

    double val = pad_value;
    // quant
    for (auto k = start; k < end; ++k) {
      if (offset < cvm_offset) {  // show click
        val += *(input_values[x] + k * embedding_size + offset);
      } else {
        val += ((static_cast<int>(
                    *(input_values[x] + k * embedding_size + offset) *
                        quant_ratio +
                    0.5)) /
                static_cast<float>(quant_ratio));
      }
    }
    seqpool_output_values[i] = val;
  }
}
// quant filter
template <typename T>
__global__ void FusedSeqpoolKernelQuantFilter(const size_t N,
                                              T **input_values,
                                              T *seqpool_output_values,
                                              const size_t *lods_values,
                                              const int batch_size,
                                              const int embedding_size,
                                              const float pad_value,
                                              const int cvm_offset,
                                              const float show_coeff,
                                              const float clk_coeff,
                                              const float threshold,
                                              const int quant_ratio,
                                              const bool xbox_diff_thres_filter,
                                              const float *threshold_vec_gpu) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx id
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id
    auto &start = lods_values[x * (batch_size + 1) + y];
    auto &end = lods_values[x * (batch_size + 1) + y + 1];

    double val = pad_value;
    for (auto k = start; k < end; ++k) {
      T *in = (input_values[x] + k * embedding_size);
      T &show = in[0];
      T &click = in[1];
      if (!xbox_diff_thres_filter) {
        // normal threshold filter
        if ((show - click) * show_coeff + click * clk_coeff < threshold) {
          continue;
        }
      } else {
        if ((show - click) * show_coeff + click * clk_coeff <
            threshold_vec_gpu[x]) {
          continue;
        }
      }
      // if ((show - click) * show_coeff + click * clk_coeff < threshold) {
      //   continue;
      // }
      if (offset < cvm_offset) {  // show & click
        val += in[offset];
      } else {
        val += ((static_cast<int>(in[offset] * quant_ratio + 0.5)) /
                static_cast<float>(quant_ratio));
      }
    }
    seqpool_output_values[i] = val;
  }
}
// join need show click input
template <typename T>
__global__ void FusedCVMKernelWithCVM(const size_t N,
                                      T **output_values,
                                      const T *seqpool_output_values,
                                      const int batch_size,
                                      const int embedding_size,
                                      const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    // set ptr
    const T *in = &seqpool_output_values[key * embedding_size];
    T *out = (output_values[x] + y * embedding_size + offset);
    if (offset == 0) {  // log(show + 1)
      *out = log(in[0] + 1);
    } else if (offset == 1) {  // ctr = log(click + 1) - log(show + 1)
      *out = log(in[1] + 1) - log(in[0] + 1);
    } else {
      *out = in[offset];
    }
  }
}
// join only need show input
template <typename T>
__global__ void FusedCVMKernelWithShow(const size_t N,
                                       T **output_values,
                                       const T *seqpool_output_values,
                                       const int batch_size,
                                       const int embedding_size,
                                       const int noclk_embedding_size) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / noclk_embedding_size;
    int offset = i % noclk_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    const T *in = &seqpool_output_values[key * embedding_size];
    T *out = (output_values[x] + y * noclk_embedding_size + offset);
    if (offset == 0) {  // show
      *out = log(in[0] + 1);
    } else {  // skip click offset + 1
      *out = in[offset + 1];
    }
  }
}
// update not need show click input
template <typename T>
__global__ void FusedCVMKernelNoCVM(const size_t N,
                                    T **output_values,
                                    const T *seqpool_output_values,
                                    const int batch_size,
                                    const int no_cvm_embedding_size,
                                    const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / no_cvm_embedding_size;
    int offset = i % no_cvm_embedding_size;
    int x = key / batch_size;  // slot id
    int y = key % batch_size;  // ins id
    const T *in =
        &seqpool_output_values[key * (no_cvm_embedding_size + cvm_offset)];

    // no cvm
    *(output_values[x] + y * no_cvm_embedding_size + offset) =
        in[offset + cvm_offset];
  }
}

template <typename T>
void FusedSeqpoolCVM(const paddle::platform::Place &place,
                     const std::vector<const T *> &input_data,
                     const std::vector<T *> &output_data,
                     T *seqpool_outputs_ptr,
                     const size_t *lods_ptr,
                     const int batch_size,
                     const int slot_num,
                     const int embedding_size,
                     const float padding_value,
                     const bool use_cvm,
                     const int cvm_offset,
                     float need_filter,
                     float show_coeff,
                     float clk_coeff,
                     float threshold,
                     const int quant_ratio,
                     const bool clk_filter,
                     const bool xbox_diff_thres_filter,
                     std::vector<float> &threshold_vec) {
  auto stream = dynamic_cast<phi::GPUContext *>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();

  size_t total_ptr_len = input_data.size() + output_data.size();
  auto temp_ptr = memory::AllocShared(place, total_ptr_len * sizeof(void *));
  void *ptr = temp_ptr->ptr();

  T **gpu_input_values = reinterpret_cast<T **>(temp_ptr->ptr());
  cudaMemcpyAsync(gpu_input_values,
                  input_data.data(),
                  input_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice,
                  stream);
  T **gpu_output_values =
      reinterpret_cast<T **>(&gpu_input_values[input_data.size()]);
  cudaMemcpyAsync(gpu_output_values,
                  output_data.data(),
                  output_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice,
                  stream);

  size_t vec_len = threshold_vec.size();
  auto vec_temp_ptr = memory::AllocShared(place, total_ptr_len * sizeof(float));
  float *threshold_vec_gpu = reinterpret_cast<float *>(vec_temp_ptr->ptr());
  cudaMemcpyAsync(threshold_vec_gpu,
                  threshold_vec.data(),
                  threshold_vec.size() * sizeof(float),
                  cudaMemcpyHostToDevice,
                  stream);

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  // first sum pool
  if (need_filter) {  // quant need filter
    FusedSeqpoolKernelQuantFilter<<<GET_BLOCK(N),
                                    PADDLE_CUDA_NUM_THREADS,
                                    0,
                                    stream>>>(N,
                                              gpu_input_values,
                                              seqpool_outputs_ptr,
                                              lods_ptr,
                                              batch_size,
                                              embedding_size,
                                              padding_value,
                                              cvm_offset,
                                              show_coeff,
                                              clk_coeff,
                                              threshold,
                                              quant_ratio,
                                              xbox_diff_thres_filter,
                                              threshold_vec_gpu);
  } else if (quant_ratio > 0) {  // quant not filter
    FusedSeqpoolKernelQuant<<<GET_BLOCK(N),
                              PADDLE_CUDA_NUM_THREADS,
                              0,
                              stream>>>(N,
                                        gpu_input_values,
                                        seqpool_outputs_ptr,
                                        lods_ptr,
                                        batch_size,
                                        embedding_size,
                                        padding_value,
                                        cvm_offset,
                                        quant_ratio);
  } else {  // normal
    FusedSeqpoolKernelNormal<<<GET_BLOCK(N),
                               PADDLE_CUDA_NUM_THREADS,
                               0,
                               stream>>>(N,
                                         gpu_input_values,
                                         seqpool_outputs_ptr,
                                         lods_ptr,
                                         batch_size,
                                         embedding_size,
                                         padding_value);
  }
  // second log
  if (use_cvm) {
    if (clk_filter) {  // skip click
      N = static_cast<size_t>(batch_size * slot_num * (embedding_size - 1));
      FusedCVMKernelWithShow<<<GET_BLOCK(N),
                               PADDLE_CUDA_NUM_THREADS,
                               0,
                               stream>>>(N,
                                         gpu_output_values,
                                         seqpool_outputs_ptr,
                                         batch_size,
                                         embedding_size,
                                         embedding_size - 1);
    } else {
      FusedCVMKernelWithCVM<<<GET_BLOCK(N),
                              PADDLE_CUDA_NUM_THREADS,
                              0,
                              stream>>>(N,
                                        gpu_output_values,
                                        seqpool_outputs_ptr,
                                        batch_size,
                                        embedding_size,
                                        cvm_offset);
    }
  } else {
    // not need show click input
    N = static_cast<size_t>(batch_size * slot_num *
                            (embedding_size - cvm_offset));
    FusedCVMKernelNoCVM<<<GET_BLOCK(N), PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        N,
        gpu_output_values,
        seqpool_outputs_ptr,
        batch_size,
        (embedding_size - cvm_offset),
        cvm_offset);
  }
}
// join grad
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithCVM(const size_t N,
                                                 T **out_grads_values,
                                                 T **in_grads_values,
                                                 T **cvm_values,
                                                 const size_t *lods_values,
                                                 const int batch_size,
                                                 const int embedding_size,
                                                 const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    auto &start = lods_values[x * (batch_size + 1) + y];
    auto &end = lods_values[x * (batch_size + 1) + y + 1];
    T &val = (offset < cvm_offset)
                 ? *(cvm_values[x] + y * cvm_offset + offset)
                 : *(out_grads_values[x] + y * embedding_size + offset);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}
// join only show not has click
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelWithShow(const size_t N,
                                                  T **out_grads_values,
                                                  T **in_grads_values,
                                                  T **cvm_values,
                                                  const size_t *lods_values,
                                                  const int batch_size,
                                                  const int embedding_size,
                                                  const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    auto &start = lods_values[x * (batch_size + 1) + y];
    auto &end = lods_values[x * (batch_size + 1) + y + 1];
    T &val =
        (offset < cvm_offset)
            ? *(cvm_values[x] + y * cvm_offset + offset)
            : *(out_grads_values[x] + y * (embedding_size - 1) + offset - 1);
    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}
// update grad
template <typename T>
__global__ void FusedSeqpoolCVMGradKernelNoCVM(const size_t N,
                                               T **out_grads_values,
                                               T **in_grads_values,
                                               T **cvm_values,
                                               const size_t *lods_values,
                                               const int batch_size,
                                               const int embedding_size,
                                               const int cvm_offset) {
  CUDA_KERNEL_LOOP(i, N) {
    int key = i / embedding_size;
    int offset = i % embedding_size;  // embedx offset
    int x = key / batch_size;         // slot id
    int y = key % batch_size;         // ins id

    auto &start = lods_values[x * (batch_size + 1) + y];
    auto &end = lods_values[x * (batch_size + 1) + y + 1];
    T &val = (offset < cvm_offset)
                 ? *(cvm_values[x] + y * cvm_offset + offset)
                 : *(out_grads_values[x] + y * (embedding_size - cvm_offset) +
                     offset - cvm_offset);

    for (auto k = start; k < end; ++k) {
      *(in_grads_values[x] + k * embedding_size + offset) = val;
    }
  }
}
template <typename T>
void FusedSeqpoolCVMGrad(const paddle::platform::Place &place,
                         const std::vector<const T *> &out_grads_data,
                         const std::vector<T *> &in_grads_data,
                         const std::vector<const T *> &cvm_data,
                         const size_t *lods_values,
                         const int batch_size,
                         const int slot_num,
                         const int embedding_size,
                         const bool use_cvm,
                         const int cvm_offset,
                         const bool clk_filter) {
  auto stream = dynamic_cast<phi::GPUContext *>(
                    platform::DeviceContextPool::Instance().Get(place))
                    ->stream();
  size_t total_ptr_len =
      out_grads_data.size() + in_grads_data.size() + cvm_data.size();
  auto temp_ptr = memory::AllocShared(place, total_ptr_len * sizeof(void *));
  T **gpu_out_grads_values = reinterpret_cast<T **>(temp_ptr->ptr());
  cudaMemcpyAsync(gpu_out_grads_values,
                  out_grads_data.data(),
                  out_grads_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice,
                  stream);

  T **gpu_in_grads_values =
      reinterpret_cast<T **>(&gpu_out_grads_values[out_grads_data.size()]);
  cudaMemcpyAsync(gpu_in_grads_values,
                  in_grads_data.data(),
                  in_grads_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice,
                  stream);

  T **gpu_cvm_values =
      reinterpret_cast<T **>(&gpu_in_grads_values[in_grads_data.size()]);
  cudaMemcpyAsync(gpu_cvm_values,
                  cvm_data.data(),
                  cvm_data.size() * sizeof(T *),
                  cudaMemcpyHostToDevice,
                  stream);

  size_t N = static_cast<size_t>(batch_size * slot_num * embedding_size);
  if (use_cvm) {
    if (clk_filter) {
      FusedSeqpoolCVMGradKernelWithShow<<<GET_BLOCK(N),
                                          PADDLE_CUDA_NUM_THREADS,
                                          0,
                                          stream>>>(N,
                                                    gpu_out_grads_values,
                                                    gpu_in_grads_values,
                                                    gpu_cvm_values,
                                                    lods_values,
                                                    batch_size,
                                                    embedding_size,
                                                    cvm_offset);
    } else {
      // join grad
      FusedSeqpoolCVMGradKernelWithCVM<<<GET_BLOCK(N),
                                         PADDLE_CUDA_NUM_THREADS,
                                         0,
                                         stream>>>(N,
                                                   gpu_out_grads_values,
                                                   gpu_in_grads_values,
                                                   gpu_cvm_values,
                                                   lods_values,
                                                   batch_size,
                                                   embedding_size,
                                                   cvm_offset);
    }
  } else {
    // update grad
    FusedSeqpoolCVMGradKernelNoCVM<<<GET_BLOCK(N),
                                     PADDLE_CUDA_NUM_THREADS,
                                     0,
                                     stream>>>(N,
                                               gpu_out_grads_values,
                                               gpu_in_grads_values,
                                               gpu_cvm_values,
                                               lods_values,
                                               batch_size,
                                               embedding_size,
                                               cvm_offset);
  }
}

template <typename T>
class FusedSeqpoolCVMWithDiffThresCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto inputs = ctx.MultiInput<LoDTensor>("X");
    auto outputs = ctx.MultiOutput<framework::Tensor>("Out");

    const int slot_size = static_cast<int>(inputs.size());
    std::vector<const float *> input_data(slot_size);
    std::vector<T *> output_data(slot_size);

    phi::DenseTensor seqpool_tensor;

    auto padding_value = ctx.Attr<float>("pad_value");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    bool need_filter = ctx.Attr<bool>("need_filter");
    float show_coeff = ctx.Attr<float>("show_coeff");
    float clk_coeff = ctx.Attr<float>("clk_coeff");
    float threshold = ctx.Attr<float>("threshold");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    const int quant_ratio = ctx.Attr<int>("quant_ratio");
    bool clk_filter = ctx.Attr<bool>("clk_filter");

    bool xbox_diff_thres_filter = ctx.Attr<bool>("xbox_diff_thres_filter");
    auto threshold_vec_param = ctx.Attr<std::vector<float>>("threshold_vec");
    std::vector<float> threshold_vec(threshold_vec_param.begin(),
                                     threshold_vec_param.end());

    framework::GPULodVector gpu_lods[slot_size];
    auto place = ctx.GetPlace();

    CHECK(inputs[0]->dims()[0] > 0);
    int embedding_size = inputs[0]->numel() / inputs[0]->dims()[0];
    int batch_size = inputs[0]->lod()[0].size() - 1;

    T *seqpool_ptr = seqpool_tensor.mutable_data<T>(
        {slot_size * batch_size, embedding_size}, place);
    // lod ptr
    auto lods_values = memory::AllocShared(
        place, sizeof(size_t) * slot_size * (batch_size + 1));
    size_t *lods_ptr = reinterpret_cast<size_t *>(lods_values->ptr());

    auto stream = dynamic_cast<phi::GPUContext *>(
                      platform::DeviceContextPool::Instance().Get(place))
                      ->stream();

    for (int i = 0; i < slot_size; ++i) {
      const auto *input = inputs[i];
      CHECK(input->lod().size() == 1);

      auto lod_data = input->lod()[0];
      int cur_batch = lod_data.size() - 1;
      CHECK(batch_size == cur_batch)
          << "batch: " << batch_size << ", current: " << cur_batch;

      input_data[i] = reinterpret_cast<const T *>(input->data<T>());
      auto *output = outputs[i];
      if (use_cvm) {
        if (clk_filter) {
          output->Resize({batch_size, embedding_size - 1});
        } else {
          output->Resize({batch_size, embedding_size});
        }
      } else {
        output->Resize({batch_size, embedding_size - cvm_offset});
      }
      output_data[i] = reinterpret_cast<T *>(output->mutable_data<T>(place));
      // copy load to gpu
      cudaMemcpyAsync(&lods_ptr[(batch_size + 1) * i],
                      lod_data.data(),
                      lod_data.size() * sizeof(size_t),
                      cudaMemcpyHostToDevice,
                      stream);
    }
    FusedSeqpoolCVM(place,
                    input_data,
                    output_data,
                    seqpool_ptr,
                    lods_ptr,
                    batch_size,
                    slot_size,
                    embedding_size,
                    padding_value,
                    use_cvm,
                    cvm_offset,
                    need_filter,
                    show_coeff,
                    clk_coeff,
                    threshold,
                    quant_ratio,
                    clk_filter,
                    xbox_diff_thres_filter,
                    threshold_vec);
  }
};

template <typename T>
class FusedSeqpoolCVMWithDiffThresGradCUDAKernel
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_grads = ctx.MultiInput<LoDTensor>(framework::GradVarName("Out"));
    auto in_grads = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));
    auto *cvm = ctx.Input<LoDTensor>("CVM");

    std::string pooltype = ctx.Attr<std::string>("pooltype");
    auto use_cvm = ctx.Attr<bool>("use_cvm");
    const int cvm_offset = ctx.Attr<int>("cvm_offset");
    bool clk_filter = ctx.Attr<bool>("clk_filter");

    const int slot_size = static_cast<int>(in_grads.size());
    std::vector<const T *> out_grads_data(slot_size);
    std::vector<T *> in_grads_data(slot_size);
    std::vector<const T *> cvm_data(slot_size);

    auto place = ctx.GetPlace();

    CHECK(in_grads[0]->dims()[0] > 0);
    int embedding_size = in_grads[0]->numel() / in_grads[0]->dims()[0];
    int batch_size = in_grads[0]->lod()[0].size() - 1;

    // lod ptr
    auto lods_values = memory::AllocShared(
        place, sizeof(size_t) * slot_size * (batch_size + 1));
    size_t *lods_ptr = reinterpret_cast<size_t *>(lods_values->ptr());
    auto stream = dynamic_cast<phi::GPUContext *>(
                      platform::DeviceContextPool::Instance().Get(place))
                      ->stream();

    for (int i = 0; i < slot_size; ++i) {
      auto *in_grad = in_grads[i];
      auto lod_data = in_grad->lod()[0];
      int cur_batch = lod_data.size() - 1;
      CHECK(batch_size == cur_batch)
          << "batch: " << batch_size << ", current: " << cur_batch;
      auto *out_grad = out_grads[i];
      out_grads_data[i] = reinterpret_cast<const T *>(out_grad->data<T>());

      in_grads_data[i] = reinterpret_cast<T *>(in_grad->mutable_data<T>(place));
      // copy load to gpu
      cudaMemcpyAsync(&lods_ptr[(batch_size + 1) * i],
                      lod_data.data(),
                      lod_data.size() * sizeof(size_t),
                      cudaMemcpyHostToDevice,
                      stream);
      cvm_data[i] = reinterpret_cast<const T *>(cvm->data<T>());
    }
    FusedSeqpoolCVMGrad(place,
                        out_grads_data,
                        in_grads_data,
                        cvm_data,
                        lods_ptr,
                        batch_size,
                        slot_size,
                        embedding_size,
                        use_cvm,
                        cvm_offset,
                        clk_filter);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_with_diff_thres,
                        ops::FusedSeqpoolCVMWithDiffThresCUDAKernel<float>);

REGISTER_OP_CUDA_KERNEL(fused_seqpool_cvm_with_diff_thres_grad,
                        ops::FusedSeqpoolCVMWithDiffThresGradCUDAKernel<float>);
