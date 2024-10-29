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
#include <string>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_memory_aligment.h"
#if defined(PADDLE_WITH_BOX_PS)
#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#include "paddle/fluid/platform/device/gpu/nccl_helper.h"
#elif defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_XPU)
#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/bkcl_helper.h"
#endif
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/platform/collective_helper.h"
#endif
#include "paddle/fluid/operators/tensor_formatter.h"

#if defined(TRACE_PROFILE) && (defined(PADDLE_WITH_XPU_KP) || defined(PADDLE_WITH_XPU))
// The producer side.
#include <scalopus_tracing/tracing.h>
#include <scalopus_transport/transport_loopback.h>
// The catapult recorder side.
#include <scalopus_catapult/catapult_recorder.h>
#include <scalopus_general/endpoint_manager_poll.h>
#include <scalopus_general/general_provider.h>
#include <scalopus_tracing/native_trace_provider.h>
#endif

namespace paddle {
namespace operators {

class CMixAllGatherOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

template <typename T>
class CMixAllGatherOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_THROW("CMixAllGather op do not support CPUKernel for now.");
  }
};

// template<typename T>
// int check_illegal_count(const T* a, int size, char *buf) {
//    int zero = 0;
//    int nan = 0;
//    int inf = 0;
//    for (int i = 0; i < size; ++i) {
//        if (a[i] == 0) {
//            zero = zero + 1;
//        } else if (isnan(a[i])) {
//            nan = nan + 1;
//        } else if (isinf(a[i])) {
//            inf = inf + 1;
//        }
//    }
//    return snprintf(buf, 2048, "(SIZE:%d,NA:%d,INF:%d,ZERO:%d),", size, nan,
//    inf, zero);
//}
// void print_cpu_data(const char *name, int device, const void *address, const
// paddle::platform::float16 *a, int size) {
//
//}
// void print_cpu_data(const char *name, int device, const void *address, const
// double *a, int size) {
//
//}
// void print_cpu_data(const char *name, int device, const void *address, const
// int *a, int size) {
//
//}
// void print_cpu_data(const char *name, int device, const void *address, const
// int64_t *a, int size) {
//
//}
// template<typename T>
// void print_cpu_data(const char *name, int device, const void *address, const
// T *a, int size) {
//    char szbuf[8193] = {0};
//    int offset = check_illegal_count(a, size, szbuf);
//    if (size > 100) {
//        int step = size / 100;
//        for (int i = 0; i < size; i = i + step) {
//            offset += snprintf(szbuf + offset, 8192 - offset, "%f,", a[i]);
//        }
//    } else {
//        for (int i = 0; i < size; ++ i) {
//            offset += snprintf(szbuf + offset, 8192 - offset, "%f,", a[i]);
//        }
//    }
//    fprintf(stdout, "[%d]%s(%p):%s\n", device, name, address, szbuf);
//}
//
// template<typename T>
// void print_gpu_data(const char *name, const T *a, int size, int device,
// cudaStream_t stream) {
//    T *buf = 0;
//    cudaHostAlloc((void **)&buf, sizeof(T) * size, cudaHostAllocDefault);
//    cudaMemcpyAsync(buf, a, size * sizeof(float), cudaMemcpyDeviceToHost,
//    stream);
//    cudaStreamSynchronize(stream);
//    print_cpu_data(name, device, a, buf, size);
//    cudaFreeHost(buf);
//}

template <typename T>
class CMixAllGatherOpCUDAKernel : public framework::OpKernel<T> {
  static const int NCCL_ALLREDUCE = 0;
  static const int NCCL_MIXALLGATHER = 1;
  static const int NCCL_ALLGATHER = 2;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if defined(PADDLE_WITH_NCCL) && defined(PADDLE_WITH_BOX_PS)
    auto in_tensors = ctx.MultiInput<framework::LoDTensor>("Input");
    auto fused_tensor = ctx.Output<framework::LoDTensor>("Output");
    //    auto in_var_names = ctx.InputNames("Input");

    int nranks = ctx.Attr<int>("nranks");
    int rank_id = ctx.Attr<int>("rankid");
    int nccl_mode = ctx.Attr<int>("nccl_mode");
    int ring_id = ctx.Attr<int>("ring_id");
    bool multi_nccl = ctx.Attr<bool>("multi_nccl");

    auto place = ctx.GetPlace();

    int device_id = place.GetDeviceId();
    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();

    box_ptr->DenseNcclTimer(device_id, false, 0x03);

    int64_t numel = 0;
    ncclDataType_t nccl_dtype =
            platform::ToNCCLDataType(framework::TransToProtoVarType(in_tensors[0]->dtype()));
    GetTensorMemSize(in_tensors, &numel);

    int64_t offset = 0;
    int64_t recv_len = 0;
    int64_t pad_len = 0;
    T *recvbuff = nullptr;
    T *sendbuff = nullptr;

    auto comm = platform::NCCLCommContext::Instance().Get(0, device_id);
    int comm_rank_num = comm->nranks();
    int device_num = platform::GetGPUDeviceCount();

    if (nccl_mode == NCCL_ALLGATHER) {  // allgather
      if (comm_rank_num == device_num) {
        if (multi_nccl) {
          offset = numel * (rank_id + device_id * nranks);
        } else {
          offset = numel * (device_num * rank_id + device_id);
        }
      } else {
        offset = numel * comm->rank();
      }
      recv_len = numel * nranks * device_num;
      recvbuff = fused_tensor->mutable_data<T>(
          {static_cast<int64_t>(recv_len), 1}, place);
      sendbuff = &recvbuff[offset];
    } else if (nccl_mode == NCCL_MIXALLGATHER) {  // mixallgather
      CHECK(comm_rank_num == device_num);
      if (nranks > 1 && multi_nccl) {
        if ((numel % device_num) != 0) {
          pad_len = device_num - (numel % device_num);
          numel = numel + pad_len;
          //          printf("total %ld, pad len: %ld\n", numel, pad_len);
        }
        recv_len = numel * nranks + nranks * (numel / device_num);
        recvbuff = fused_tensor->mutable_data<T>(
            {static_cast<int64_t>(recv_len), 1}, place);
        sendbuff = recvbuff;
      } else {
        offset = numel * rank_id;
        recv_len = numel * nranks;
        recvbuff = fused_tensor->mutable_data<T>(
            {static_cast<int64_t>(recv_len), 1}, place);
        sendbuff = &recvbuff[offset];
      }
    } else {  // allreduce
      if (nranks > 1 && comm_rank_num == device_num &&
          ((numel % device_num) != 0)) {
        pad_len = device_num - (numel % device_num);
        numel = numel + pad_len;
      }
      recvbuff = fused_tensor->mutable_data<T>({numel, 1}, place);
      sendbuff = recvbuff;
      recv_len = numel;
    }

    auto dev_ctx = paddle::platform::DeviceContextPool::Instance().Get(place);
    CHECK(static_cast<int64_t>(recv_len) == fused_tensor->numel());
    // copy input datas
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      int64_t len = in_tensors[i]->numel();
      auto sub_tensor = fused_tensor->Slice(offset, offset + len);
      framework::TensorCopy(*in_tensors[i], place, *dev_ctx, &sub_tensor);
      offset += len;
    }

    cudaStream_t stream =
            dynamic_cast<phi::GPUContext *>(dev_ctx)->stream();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    box_ptr->DenseNcclTimer(device_id, true, 0x02);

    if (nranks > 1 && comm_rank_num == device_num) {
      if (multi_nccl) {  // multi node nccl more than two network card
        if (nccl_mode == NCCL_ALLREDUCE) {  // allreduce
          // [inner reducescatter->node allreduce->allgather]
          int64_t part_param_len = numel / device_num;
          T *recv_ptr = &recvbuff[device_id * part_param_len];
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclReduceScatter(
              sendbuff, recv_ptr, part_param_len, nccl_dtype, ncclSum,
              comm->comm(), stream));
          CHECK(box_ptr->SyncDense(stream, part_param_len, recv_ptr, recv_ptr,
                                   device_id, false));
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
              recv_ptr, recvbuff, part_param_len, nccl_dtype, comm->comm(),
              stream));
        } else if (nccl_mode == NCCL_ALLGATHER) {  // allgather
          // [node allgather-> inner allgather]
          CHECK(box_ptr->SyncDense(stream, numel, sendbuff,
                                   &recvbuff[numel * device_id * nranks],
                                   device_id, true));
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
              &recvbuff[numel * device_id * nranks], recvbuff, numel * nranks,
              nccl_dtype, comm->comm(), stream));
        } else {  // mixallgather
          // [inner reducescatter->node allgather->inner allgather]
          int64_t part_param_len = numel / device_num;
          T *recv_ptr = &recvbuff[device_id * part_param_len];
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclReduceScatter(
              sendbuff, recv_ptr, part_param_len, nccl_dtype, ncclSum,
              comm->comm(), stream));
          T *recv_buff_ext = &recvbuff[numel * nranks];
          CHECK(box_ptr->SyncDense(stream, part_param_len, recv_ptr,
                                   recv_buff_ext, device_id, true));
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
          // slice_multi_tensor op need symmetric data
          for (int rank = 0; rank < nranks; ++rank) {
            // gather data by rank id order
            PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
                &recv_buff_ext[part_param_len * rank],
                &recvbuff[rank * (numel + part_param_len)], part_param_len,
                nccl_dtype, comm->comm(), stream));
          }
        }
      } else {  // only single network card
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupStart());
        if (nccl_mode == NCCL_ALLGATHER) {  // allgather
          // [inner allgather->device 0 node allgather->inner bcast]
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
              sendbuff, &recvbuff[numel * device_num * rank_id], numel,
              nccl_dtype, comm->comm(), stream));
          if (device_id == 0) {
            if (ctx.Attr<bool>("use_boxps_nccl")) {
              CHECK(box_ptr->SyncDense(stream, numel * device_num,
                                       &recvbuff[numel * device_num * rank_id],
                                       recvbuff, 0, true));
            } else {
              // node allgather
              auto node_comm =
                  platform::NCCLCommContext::Instance().Get(ring_id, 0);
              PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
                  &recvbuff[numel * device_num * rank_id], recvbuff,
                  numel * device_num, nccl_dtype, node_comm->comm(), stream));
            }
          }
        } else {  // mixallgather allreduce
          // [inner reduce to device 0 -> device 0 node allgather or allreduce
          // -> inner bcast]
          PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclReduce(
              sendbuff, sendbuff, numel, nccl_dtype, ncclSum, 0, comm->comm(),
              stream));
          if (device_id == 0) {
            if (ctx.Attr<bool>("use_boxps_nccl")) {
              CHECK(box_ptr->SyncDense(stream, numel, sendbuff, recvbuff, 0,
                                       (nccl_mode == NCCL_MIXALLGATHER)));
            } else {
              auto node_comm =
                  platform::NCCLCommContext::Instance().Get(ring_id, 0);
              if (nccl_mode == NCCL_MIXALLGATHER) {
                // allgather
                PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
                    sendbuff, recvbuff, numel, nccl_dtype, node_comm->comm(),
                    stream));
              } else {
                // allreduce
                PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
                    sendbuff, recvbuff, numel, nccl_dtype, ncclSum,
                    node_comm->comm(), stream));
              }
            }
          }
        }
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclGroupEnd());
        // broadcast to all device
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclBcast(
            recvbuff, recv_len, nccl_dtype, 0, comm->comm(), stream));
      }
    } else {  // single node or one ring
      if (nccl_mode == NCCL_ALLGATHER) {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllGather(
            sendbuff, recvbuff, numel, nccl_dtype, comm->comm(), stream));
      } else {
        PADDLE_ENFORCE_GPU_SUCCESS(platform::dynload::ncclAllReduce(
            sendbuff, recvbuff, numel, nccl_dtype, ncclSum, comm->comm(),
            stream));
      }
    }
    PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
    //    if (device_id == 0) {
    //        print_gpu_data("fuse_nccl", recvbuff, static_cast<int>(recv_len),
    //        device_id, stream);
    //    }
    box_ptr->DenseNcclTimer(device_id, true, 0x01);
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  }

 protected:
  void GetTensorMemSize(
      const std::vector<const framework::LoDTensor *> &lod_tensors,
      int64_t *numel) const {
    *numel = 0;
    for (size_t i = 0; i < lod_tensors.size(); ++i) {
      CHECK(lod_tensors[i]->IsInitialized());
      *numel += lod_tensors[i]->numel();
    }
  }
};

template <typename T>
class CMixAllGatherOpXPUKernel : public framework::OpKernel<T> {
  static const int NCCL_ALLREDUCE = 0;
  static const int NCCL_MIXALLGATHER = 1;
  static const int NCCL_ALLGATHER = 2;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#if defined(PADDLE_WITH_XPU_BKCL) && defined(PADDLE_WITH_BOX_PS)
    auto in_tensors = ctx.MultiInput<framework::LoDTensor>("Input");
    auto fused_tensor = ctx.Output<framework::LoDTensor>("Output");

//    int nranks = ctx.Attr<int>("nranks");
//    int rank_id = ctx.Attr<int>("rankid");
    int nccl_mode = ctx.Attr<int>("nccl_mode");
//    int ring_id = ctx.Attr<int>("ring_id");

    if (nccl_mode == NCCL_ALLGATHER || nccl_mode == NCCL_MIXALLGATHER) {
      PADDLE_THROW("PaddlePaddle xpu not support gather mode.");
      return;
    }

    auto place = ctx.GetPlace();

    int device_id = place.GetDeviceId();
    auto box_ptr = paddle::framework::BoxWrapper::GetInstance();

    box_ptr->DenseNcclTimer(device_id, false, 0x03);

    int64_t numel = 0;
    BKCLDataType bkcl_dtype =
            platform::ToBKCLDataType(framework::TransToProtoVarType(in_tensors[0]->dtype()));
    GetTensorMemSize(in_tensors, &numel);

    auto comm = platform::BKCLCommContext::Instance().Get(0, device_id);
//    int comm_rank_num = comm->nranks();
//    int device_num = platform::GetDeviceCount();

    T *recvbuff = fused_tensor->mutable_data<T>({numel, 1}, place);

    auto dev_ctx = paddle::platform::DeviceContextPool::Instance().Get(place);
    // copy input datas
    int64_t offset = 0;
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      int64_t len = in_tensors[i]->numel();
      auto sub_tensor = fused_tensor->Slice(offset, offset + len);
      framework::TensorCopy(*in_tensors[i], place, *dev_ctx, &sub_tensor);
      offset += len;
    }
    box_ptr->DenseNcclTimer(device_id, true, 0x02);
    XPUStream stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
                       ->x_context()
                       ->xpu_stream;
#ifdef TRACE_PROFILE
    TRACE_SCOPE_START("bkcl_all_reduce", xpu_wait(stream));
#endif

    PADDLE_ENFORCE_EQ(
        bkcl_all_reduce(comm->comm(),
                        recvbuff,
                        recvbuff,
                        numel,
                        bkcl_dtype,
                        BKCL_ADD,
                        stream),
            BKCL_SUCCESS,
            platform::errors::PreconditionNotMet("BKCL all reduce failed"));
    PADDLE_ENFORCE_XPU_SUCCESS(xpu_wait(stream));
#ifdef TRACE_PROFILE
    TRACE_SCOPE_END("bkcl_all_reduce",);
#endif
    box_ptr->DenseNcclTimer(device_id, true, 0x01);
#else
    PADDLE_THROW("PaddlePaddle should compile with XPU.");
#endif
  }

 protected:
  void GetTensorMemSize(
      const std::vector<const framework::LoDTensor *> &lod_tensors,
      int64_t *numel) const {
    *numel = 0;
    for (size_t i = 0; i < lod_tensors.size(); ++i) {
      CHECK(lod_tensors[i]->IsInitialized());
      *numel += lod_tensors[i]->numel();
    }
  }
};

class CMixAllGatherOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("Input",
             "(vector<LoDTensor>) The input tensors of mixallgather_tensor "
             "operator.")
        .AsDuplicable();
    AddOutput("Output",
              "(LoDTensor) The output tensor "
              "of mixallgather_tensor operator. And the tensors of"
              " Output is sliced from the tensor of FusedOutput.");
    AddAttr<int>("rankid", "(int default 0) communication node id.")
        .SetDefault(0);
    AddAttr<int>("nranks", "(int default 1) communication node num.")
        .SetDefault(1);
    AddAttr<int>("nccl_mode",
                 "(int default 0) one node 0 allreduce, 1 mixallgather mode , "
                 "2 allgather mode.")
        .SetDefault(0);
    AddAttr<int>("ring_id", "(int default -1) nccl ring id num.")
        .SetDefault(-1);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(true);
    AddAttr<bool>("use_boxps_nccl",
                  "(bool default false) used boxps nccl sync dense data.")
        .SetDefault(false);
    AddAttr<bool>("multi_nccl",
                  "(bool default false) used multi_nccl sync dense data.")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
MixAllGather %s Operator

Call collective MixAllGather with reduce type %s. If input and output are
the same variable, in-place allreduce will be used.
Reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce
)DOC",
                               GetName(), GetName()));
  }

 protected:
  virtual std::string GetName() { return "MixAllGather"; }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_mixallgather, ops::CMixAllGatherOp,
                  ops::CMixAllGatherOpMaker);
REGISTER_OP_CPU_KERNEL(c_mixallgather, ops::CMixAllGatherOpCPUKernel<float>,
                       ops::CMixAllGatherOpCPUKernel<double>,
                       ops::CMixAllGatherOpCPUKernel<int>,
                       ops::CMixAllGatherOpCPUKernel<int64_t>,
                       ops::CMixAllGatherOpCPUKernel<plat::float16>);
#if defined(PADDLE_WITH_BOX_PS)
#if defined(PADDLE_WITH_NCCL)
REGISTER_OP_CUDA_KERNEL(c_mixallgather, ops::CMixAllGatherOpCUDAKernel<float>,
                        ops::CMixAllGatherOpCUDAKernel<double>,
                        ops::CMixAllGatherOpCUDAKernel<int>,
                        ops::CMixAllGatherOpCUDAKernel<int64_t>,
                        ops::CMixAllGatherOpCUDAKernel<plat::float16>);
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
REGISTER_OP_XPU_KERNEL(c_mixallgather, ops::CMixAllGatherOpXPUKernel<float>,
                        ops::CMixAllGatherOpXPUKernel<double>,
                        ops::CMixAllGatherOpXPUKernel<int>,
                        ops::CMixAllGatherOpXPUKernel<int64_t>,
                        ops::CMixAllGatherOpXPUKernel<plat::float16>);
#endif
#endif
