#pragma once
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class FusedSeqTensorCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext&) const override {
    PADDLE_THROW(platform::errors::Unimplemented("fused_seq_tensor supports only GPU"));
  }
};

}  // namespace operators
}  // namespace paddle
