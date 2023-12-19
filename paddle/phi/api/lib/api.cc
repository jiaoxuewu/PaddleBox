
#include "paddle/phi/api/include/api.h"
#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"

#include "paddle/fluid/platform/profiler/event_tracing.h"

DECLARE_bool(conv2d_disable_cudnn);

namespace paddle {
namespace experimental {


PADDLE_API Tensor atan2(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "atan2 API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "atan2", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "atan2 kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::Atan2InferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("atan2 compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor bernoulli(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "bernoulli API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "bernoulli", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "bernoulli kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("bernoulli compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor cholesky(const Tensor& x, bool upper) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cholesky API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cholesky", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "cholesky kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CholeskyInferMeta(MakeMetaTensor(*input_x), upper, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cholesky compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, upper, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor cholesky_solve(const Tensor& x, const Tensor& y, bool upper) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cholesky_solve API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cholesky_solve", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "cholesky_solve kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CholeskySolveInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), upper, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cholesky_solve compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, upper, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor cross(const Tensor& x, const Tensor& y, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cross API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cross", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "cross kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CrossInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cross compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor diag(const Tensor& x, int offset, float padding_value) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "diag API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "diag", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "diag kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::DiagInferMeta(MakeMetaTensor(*input_x), offset, padding_value, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("diag compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, offset, padding_value, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor diagonal(const Tensor& x, int offset, int axis1, int axis2) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "diagonal API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "diagonal", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "diagonal kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::DiagonalInferMeta(MakeMetaTensor(*input_x), offset, axis1, axis2, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("diagonal compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, offset, axis1, axis2, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor digamma(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "digamma API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "digamma", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "digamma kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("digamma compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor dist(const Tensor& x, const Tensor& y, float p) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "dist API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "dist", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "dist kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::DistInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), p, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("dist compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, p, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor dot(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "dot API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "dot", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "dot kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::DotInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("dot compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor erf(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "erf API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "erf", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "erf kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("erf compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor lgamma(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "lgamma API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "lgamma", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "lgamma kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("lgamma compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor mv(const Tensor& x, const Tensor& vec) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, vec);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "mv API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "mv", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "mv kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_vec = PrepareData(vec, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::MvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_vec), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("mv compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_vec, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor poisson(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "poisson API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "poisson", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "poisson kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("poisson compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor solve(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "solve API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "solve", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "solve kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::SolveInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("solve compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor trace(const Tensor& x, int offset, int axis1, int axis2) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "trace API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "trace", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "trace kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::TraceInferMeta(MakeMetaTensor(*input_x), offset, axis1, axis2, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("trace compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, offset, axis1, axis2, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor trunc(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "trunc API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "trunc", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "trunc kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("trunc compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor abs(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "abs API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "abs", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "abs kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::RealAndImagInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("abs compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> accuracy(const Tensor& x, const Tensor& indices, const Tensor& label) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, indices, label);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "accuracy API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "accuracy", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "accuracy kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_indices = PrepareData(indices, kernel.InputAt(1), {});
  auto input_label = PrepareData(label, kernel.InputAt(2), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::AccuracyInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_indices), MakeMetaTensor(*input_label), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("accuracy compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_indices, *input_label, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API Tensor acos(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "acos API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "acos", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "acos kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("acos compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor acosh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "acosh API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "acosh", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "acosh kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("acosh compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> adadelta(const Tensor& param, const Tensor& grad, const Tensor& avg_squared_grad, const Tensor& avg_squared_update, float rho, float epsilon) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(param, grad, avg_squared_grad, avg_squared_update);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "adadelta API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "adadelta", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "adadelta kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_param = PrepareData(param, kernel.InputAt(0), {});
  auto input_grad = PrepareData(grad, kernel.InputAt(1), {});
  auto input_avg_squared_grad = PrepareData(avg_squared_grad, kernel.InputAt(2), {});
  auto input_avg_squared_update = PrepareData(avg_squared_update, kernel.InputAt(3), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::AdadeltaInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_grad), MakeMetaTensor(*input_avg_squared_grad), MakeMetaTensor(*input_avg_squared_update), rho, epsilon, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("adadelta compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_param, *input_grad, *input_avg_squared_grad, *input_avg_squared_update, rho, epsilon, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor&, Tensor&> adagrad_(Tensor& param, const Tensor& grad, Tensor& moment, const Tensor& learning_rate, float epsilon) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(param);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(param, grad, moment, learning_rate);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (param.is_dense_tensor() && grad.is_dense_tensor() && moment.is_dense_tensor() && learning_rate.is_dense_tensor()) {

    VLOG(6) << "adagrad_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "adagrad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "adagrad kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_param = PrepareData(param, kernel.InputAt(0), {});
    auto input_grad = PrepareData(grad, kernel.InputAt(1), {});
    auto input_moment = PrepareData(moment, kernel.InputAt(2), {});
    auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(3), {});

    std::tuple<Tensor&, Tensor&> api_output{param, moment};
    auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
    auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
    phi::MetaTensor meta_out_0(kernel_out_0);
    phi::MetaTensor meta_out_1(kernel_out_1);

    phi::AdagradInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_grad), MakeMetaTensor(*input_moment), MakeMetaTensor(*input_learning_rate), epsilon, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, phi::DenseTensor*, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("adagrad compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_param, *input_grad, *input_moment, *input_learning_rate, epsilon, kernel_out_0, kernel_out_1);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    }

    return api_output;
  }

  if (param.is_dense_tensor() && grad.is_selected_rows() && moment.is_dense_tensor() && learning_rate.is_dense_tensor()) {

    VLOG(6) << "adagrad_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "adagrad_dense_param_sparse_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "adagrad_dense_param_sparse_grad kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_param = PrepareData(param, kernel.InputAt(0), {});
    auto input_grad = TensorToSelectedRows(grad);
    auto input_moment = PrepareData(moment, kernel.InputAt(2), {});
    auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(3), {});

    std::tuple<Tensor&, Tensor&> api_output{param, moment};
    auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
    auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
    phi::MetaTensor meta_out_0(kernel_out_0);
    phi::MetaTensor meta_out_1(kernel_out_1);

    phi::AdagradInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_grad), MakeMetaTensor(*input_moment), MakeMetaTensor(*input_learning_rate), epsilon, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::SelectedRows&, const phi::DenseTensor&, const phi::DenseTensor&, float, phi::DenseTensor*, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("adagrad_dense_param_sparse_grad compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_param, *input_grad, *input_moment, *input_learning_rate, epsilon, kernel_out_0, kernel_out_1);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (adagrad_) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> adam_(Tensor& param, const Tensor& grad, const Tensor& learning_rate, Tensor& moment1, Tensor& moment2, Tensor& beta1_pow, Tensor& beta2_pow, paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(param);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(param, grad, learning_rate, moment1, moment2, beta1_pow, beta2_pow, master_param, skip_update);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (param.is_dense_tensor() && grad.is_dense_tensor() && learning_rate.is_dense_tensor() && moment1.is_dense_tensor() && moment2.is_dense_tensor() && beta1_pow.is_dense_tensor() && beta2_pow.is_dense_tensor() && (!master_param || master_param->is_dense_tensor()) && (!skip_update || skip_update->is_dense_tensor())) {

    VLOG(6) << "adam_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "adam", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "adam kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_param = PrepareData(param, kernel.InputAt(0), {});
    auto input_grad = PrepareData(grad, kernel.InputAt(1), {});
    auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(2), {});
    auto input_moment1 = PrepareData(moment1, kernel.InputAt(3), {});
    auto input_moment2 = PrepareData(moment2, kernel.InputAt(4), {});
    auto input_beta1_pow = PrepareData(beta1_pow, kernel.InputAt(5), {});
    auto input_beta2_pow = PrepareData(beta2_pow, kernel.InputAt(6), {});
    auto input_master_param = PrepareData(master_param, kernel.InputAt(7), {});
    auto input_skip_update = PrepareData(skip_update, kernel.InputAt(8), {});

    std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> api_output{param, moment1, moment2, beta1_pow, beta2_pow, master_param};
    auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
    auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
    auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
    auto kernel_out_3 = SetKernelOutput(kernel_backend, &std::get<3>(api_output));
    auto kernel_out_4 = SetKernelOutput(kernel_backend, &std::get<4>(api_output));
    auto kernel_out_5 = SetKernelOutput(kernel_backend, std::get<5>(api_output).get_ptr());
    phi::MetaTensor meta_out_0(kernel_out_0);
    phi::MetaTensor meta_out_1(kernel_out_1);
    phi::MetaTensor meta_out_2(kernel_out_2);
    phi::MetaTensor meta_out_3(kernel_out_3);
    phi::MetaTensor meta_out_4(kernel_out_4);
    phi::MetaTensor meta_out_5(kernel_out_5);

    phi::AdamInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_grad), MakeMetaTensor(*input_learning_rate), MakeMetaTensor(*input_moment1), MakeMetaTensor(*input_moment2), MakeMetaTensor(*input_beta1_pow), MakeMetaTensor(*input_beta2_pow), MakeMetaTensor(input_master_param), MakeMetaTensor(input_skip_update), beta1, beta2, epsilon, lazy_mode, min_row_size_to_use_multithread, multi_precision, use_global_beta_pow, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr, kernel_out_5 ? &meta_out_5 : nullptr);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::Scalar&, const phi::Scalar&, const phi::Scalar&, bool, int64_t, bool, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("adam compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_param, *input_grad, *input_learning_rate, *input_moment1, *input_moment2, *input_beta1_pow, *input_beta2_pow, input_master_param, input_skip_update, phi::Scalar(beta1), phi::Scalar(beta2), phi::Scalar(epsilon), lazy_mode, min_row_size_to_use_multithread, multi_precision, use_global_beta_pow, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4, kernel_out_5);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
      TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
      TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
      TransDataBackend(kernel_out_4, kernel_backend, kernel_out_4);
      TransDataBackend(kernel_out_5, kernel_backend, kernel_out_5);
    }

    return api_output;
  }

  if (param.is_dense_tensor() && grad.is_selected_rows() && learning_rate.is_dense_tensor() && moment1.is_dense_tensor() && moment2.is_dense_tensor() && beta1_pow.is_dense_tensor() && beta2_pow.is_dense_tensor() && (!master_param || master_param->is_dense_tensor()) && (!skip_update || skip_update->is_dense_tensor())) {

    VLOG(6) << "adam_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "adam_dense_param_sparse_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "adam_dense_param_sparse_grad kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_param = PrepareData(param, kernel.InputAt(0), {});
    auto input_grad = TensorToSelectedRows(grad);
    auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(2), {});
    auto input_moment1 = PrepareData(moment1, kernel.InputAt(3), {});
    auto input_moment2 = PrepareData(moment2, kernel.InputAt(4), {});
    auto input_beta1_pow = PrepareData(beta1_pow, kernel.InputAt(5), {});
    auto input_beta2_pow = PrepareData(beta2_pow, kernel.InputAt(6), {});
    auto input_master_param = PrepareData(master_param, kernel.InputAt(7), {});
    auto input_skip_update = PrepareData(skip_update, kernel.InputAt(8), {});

    std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, paddle::optional<Tensor>&> api_output{param, moment1, moment2, beta1_pow, beta2_pow, master_param};
    auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
    auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
    auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
    auto kernel_out_3 = SetKernelOutput(kernel_backend, &std::get<3>(api_output));
    auto kernel_out_4 = SetKernelOutput(kernel_backend, &std::get<4>(api_output));
    auto kernel_out_5 = SetKernelOutput(kernel_backend, std::get<5>(api_output).get_ptr());
    phi::MetaTensor meta_out_0(kernel_out_0);
    phi::MetaTensor meta_out_1(kernel_out_1);
    phi::MetaTensor meta_out_2(kernel_out_2);
    phi::MetaTensor meta_out_3(kernel_out_3);
    phi::MetaTensor meta_out_4(kernel_out_4);
    phi::MetaTensor meta_out_5(kernel_out_5);

    phi::AdamInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_grad), MakeMetaTensor(*input_learning_rate), MakeMetaTensor(*input_moment1), MakeMetaTensor(*input_moment2), MakeMetaTensor(*input_beta1_pow), MakeMetaTensor(*input_beta2_pow), MakeMetaTensor(input_master_param), MakeMetaTensor(input_skip_update), beta1, beta2, epsilon, lazy_mode, min_row_size_to_use_multithread, multi_precision, use_global_beta_pow, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr, kernel_out_5 ? &meta_out_5 : nullptr);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::SelectedRows&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const phi::Scalar&, const phi::Scalar&, const phi::Scalar&, bool, int64_t, bool, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("adam_dense_param_sparse_grad compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_param, *input_grad, *input_learning_rate, *input_moment1, *input_moment2, *input_beta1_pow, *input_beta2_pow, input_master_param, input_skip_update, phi::Scalar(beta1), phi::Scalar(beta2), phi::Scalar(epsilon), lazy_mode, min_row_size_to_use_multithread, multi_precision, use_global_beta_pow, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4, kernel_out_5);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
      TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
      TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
      TransDataBackend(kernel_out_4, kernel_backend, kernel_out_4);
      TransDataBackend(kernel_out_5, kernel_backend, kernel_out_5);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (adam_) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> adamax(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment, const Tensor& inf_norm, const Tensor& beta1_pow, float beta1, float beta2, float epsilon) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(param, grad, learning_rate, moment, inf_norm, beta1_pow);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "adamax API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "adamax", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "adamax kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_param = PrepareData(param, kernel.InputAt(0), {});
  auto input_grad = PrepareData(grad, kernel.InputAt(1), {});
  auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(2), {});
  auto input_moment = PrepareData(moment, kernel.InputAt(3), {});
  auto input_inf_norm = PrepareData(inf_norm, kernel.InputAt(4), {});
  auto input_beta1_pow = PrepareData(beta1_pow, kernel.InputAt(5), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::AdamaxInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_grad), MakeMetaTensor(*input_learning_rate), MakeMetaTensor(*input_moment), MakeMetaTensor(*input_inf_norm), MakeMetaTensor(*input_beta1_pow), beta1, beta2, epsilon, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("adamax compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_param, *input_grad, *input_learning_rate, *input_moment, *input_inf_norm, *input_beta1_pow, beta1, beta2, epsilon, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> adamw(const Tensor& param, const Tensor& grad, const Tensor& learning_rate, const Tensor& moment1, const Tensor& moment2, const Tensor& beta1_pow, const Tensor& beta2_pow, const paddle::optional<Tensor>& master_param, const paddle::optional<Tensor>& skip_update, const Scalar& beta1, const Scalar& beta2, const Scalar& epsilon, float lr_ratio, float coeff, bool with_decay, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow) {
  return adamw_impl(param, grad, learning_rate, moment1, moment2, beta1_pow, beta2_pow, master_param, skip_update, beta1, beta2, epsilon, lr_ratio, coeff, with_decay, lazy_mode, min_row_size_to_use_multithread, multi_precision, use_global_beta_pow);
}
PADDLE_API Tensor add(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "add API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "add", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "add kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("add compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor add_n(const std::vector<Tensor>& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "add_n API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "add_n", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "add_n kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x_vec = PrepareData(x, kernel.InputAt(0), {});
  std::vector<const phi::DenseTensor*> input_x(input_x_vec->size());
  for (size_t i = 0; i < input_x.size(); ++i) {
    input_x[i] = &input_x_vec->at(i);
  }

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);

  auto x_meta_vec = MakeMetaTensor(input_x);
  std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
  for (size_t i = 0; i < x_meta_vec.size(); ++i) {
    x_metas[i] = &x_meta_vec[i];
  }
  phi::MetaTensor meta_out(kernel_out);

  phi::AddNInferMeta(x_metas, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<const phi::DenseTensor*>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("add_n compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor addmm(const Tensor& input, const Tensor& x, const Tensor& y, float alpha, float beta) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "addmm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "addmm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "addmm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});
  auto input_x = PrepareData(x, kernel.InputAt(1), {});
  auto input_y = PrepareData(y, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::AddmmInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), alpha, beta, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("addmm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, *input_x, *input_y, alpha, beta, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor all(const Tensor& x, const std::vector<int64_t>& dims, bool keep_dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "all API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "all", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "all kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReduceInferMeta(MakeMetaTensor(*input_x), dims, keep_dim, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("all compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dims, keep_dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor allclose(const Tensor& x, const Tensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "allclose API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "allclose", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "allclose kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::AllValueCompareInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::Scalar&, const phi::Scalar&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("allclose compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, phi::Scalar(rtol), phi::Scalar(atol), equal_nan, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor amax(const Tensor& x, const std::vector<int64_t>& dims, bool keep_dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "amax API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "amax", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "amax kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReduceInferMeta(MakeMetaTensor(*input_x), dims, keep_dim, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("amax compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dims, keep_dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor amin(const Tensor& x, const std::vector<int64_t>& dims, bool keep_dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "amin API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "amin", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "amin kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReduceInferMeta(MakeMetaTensor(*input_x), dims, keep_dim, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("amin compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dims, keep_dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor angle(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "angle API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "angle", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "angle kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::RealAndImagInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("angle compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor any(const Tensor& x, const std::vector<int64_t>& dims, bool keep_dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "any API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "any", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "any kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReduceInferMeta(MakeMetaTensor(*input_x), dims, keep_dim, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("any compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dims, keep_dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor arange(const Tensor& start, const Tensor& end, const Tensor& step, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(start, end, step);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "arange API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "arange", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "arange kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_start = PrepareData(start, kernel.InputAt(0), {false, true});
  auto input_end = PrepareData(end, kernel.InputAt(1), {false, true});
  auto input_step = PrepareData(step, kernel.InputAt(2), {false, true});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ArangeInferMeta(MakeMetaTensor(*input_start), MakeMetaTensor(*input_end), MakeMetaTensor(*input_step), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("arange compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_start, *input_end, *input_step, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor argmax(const Tensor& x, int64_t axis, bool keepdims, bool flatten, int dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "argmax API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "arg_max", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "arg_max kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ArgMinMaxInferMeta(MakeMetaTensor(*input_x), axis, keepdims, flatten, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int64_t, bool, bool, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("arg_max compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, keepdims, flatten, dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor argmin(const Tensor& x, int64_t axis, bool keepdims, bool flatten, int dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "argmin API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "arg_min", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "arg_min kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ArgMinMaxInferMeta(MakeMetaTensor(*input_x), axis, keepdims, flatten, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int64_t, bool, bool, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("arg_min compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, keepdims, flatten, dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> argsort(const Tensor& x, int axis, bool descending) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "argsort API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "argsort", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "argsort kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::ArgsortInferMeta(MakeMetaTensor(*input_x), axis, descending, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("argsort compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, descending, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor as_complex(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "as_complex API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "as_complex", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "as_complex kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::AsComplexInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("as_complex compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor as_real(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "as_real API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "as_real", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "as_real kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::AsRealInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("as_real compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor asin(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "asin API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "asin", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "asin kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("asin compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor asinh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "asinh API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "asinh", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "asinh kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("asinh compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor assign(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "assign API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "assign", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "assign kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("assign compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor& assign_out_(const Tensor& x, Tensor& output) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, output);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "assign_out_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "assign", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "assign kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor& api_output = output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("assign compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor& assign_value_(Tensor& output, const std::vector<int>& shape, DataType dtype, const std::vector<phi::Scalar>& values, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackendWithInputOrder(place, output);

  kernel_data_type = ParseDataType(dtype);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(output);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "assign_value_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "assign_value", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "assign_value kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor& api_output = output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::AssignValueInferMeta(shape, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<int>&, DataType, const std::vector<phi::Scalar>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("assign_value compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, shape, dtype, values, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor atan(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "atan API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "atan", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "atan kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("atan compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor atanh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "atanh API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "atanh", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "atanh kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("atanh compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> auc(const Tensor& x, const Tensor& label, const Tensor& stat_pos, const Tensor& stat_neg, const std::string& curve, int num_thresholds, int slide_steps) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, label, stat_pos, stat_neg);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "auc API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "auc", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "auc kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_label = PrepareData(label, kernel.InputAt(1), {});
  auto input_stat_pos = PrepareData(stat_pos, kernel.InputAt(2), {});
  auto input_stat_neg = PrepareData(stat_neg, kernel.InputAt(3), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::AucInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_label), MakeMetaTensor(*input_stat_pos), MakeMetaTensor(*input_stat_neg), curve, num_thresholds, slide_steps, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, int, int, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("auc compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_label, *input_stat_pos, *input_stat_neg, curve, num_thresholds, slide_steps, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&> average_accumulates_(const Tensor& param, Tensor& in_sum_1, Tensor& in_sum_2, Tensor& in_sum_3, Tensor& in_num_accumulates, Tensor& in_old_num_accumulates, Tensor& in_num_updates, float average_window, int64_t max_average_window, int64_t min_average_window) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(param);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(param, in_sum_1, in_sum_2, in_sum_3, in_num_accumulates, in_old_num_accumulates, in_num_updates);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "average_accumulates_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "average_accumulates", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "average_accumulates kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_param = PrepareData(param, kernel.InputAt(0), {});
  auto input_in_sum_1 = PrepareData(in_sum_1, kernel.InputAt(1), {});
  auto input_in_sum_2 = PrepareData(in_sum_2, kernel.InputAt(2), {});
  auto input_in_sum_3 = PrepareData(in_sum_3, kernel.InputAt(3), {});
  auto input_in_num_accumulates = PrepareData(in_num_accumulates, kernel.InputAt(4), {});
  auto input_in_old_num_accumulates = PrepareData(in_old_num_accumulates, kernel.InputAt(5), {});
  auto input_in_num_updates = PrepareData(in_num_updates, kernel.InputAt(6), {});

  std::tuple<Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, Tensor&> api_output{in_sum_1, in_sum_2, in_sum_3, in_num_accumulates, in_old_num_accumulates, in_num_updates};
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(kernel_backend, &std::get<3>(api_output));
  auto kernel_out_4 = SetKernelOutput(kernel_backend, &std::get<4>(api_output));
  auto kernel_out_5 = SetKernelOutput(kernel_backend, &std::get<5>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);
  phi::MetaTensor meta_out_3(kernel_out_3);
  phi::MetaTensor meta_out_4(kernel_out_4);
  phi::MetaTensor meta_out_5(kernel_out_5);

  phi::AverageAccumulatesInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_in_sum_1), MakeMetaTensor(*input_in_sum_2), MakeMetaTensor(*input_in_sum_3), MakeMetaTensor(*input_in_num_accumulates), MakeMetaTensor(*input_in_old_num_accumulates), MakeMetaTensor(*input_in_num_updates), average_window, max_average_window, min_average_window, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr, kernel_out_5 ? &meta_out_5 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, int64_t, int64_t, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("average_accumulates compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_param, *input_in_sum_1, *input_in_sum_2, *input_in_sum_3, *input_in_num_accumulates, *input_in_old_num_accumulates, *input_in_num_updates, average_window, max_average_window, min_average_window, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4, kernel_out_5);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
    TransDataBackend(kernel_out_4, kernel_backend, kernel_out_4);
    TransDataBackend(kernel_out_5, kernel_backend, kernel_out_5);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> batch_norm(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& variance, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics, bool fuse_with_relu) {
  return batch_norm_impl(x, scale, bias, mean, variance, momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics, fuse_with_relu);
}
PADDLE_API Tensor bce_loss(const Tensor& input, const Tensor& label) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, label);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "bce_loss API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "bce_loss", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "bce_loss kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});
  auto input_label = PrepareData(label, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::BCELossInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_label), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("bce_loss compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, *input_label, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor bilinear_tensor_product(const Tensor& x, const Tensor& y, const Tensor& weight, const paddle::optional<Tensor>& bias) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, weight, bias);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "bilinear_tensor_product API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "bilinear_tensor_product", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "bilinear_tensor_product kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});
  auto input_weight = PrepareData(weight, kernel.InputAt(2), {});
  auto input_bias = PrepareData(bias, kernel.InputAt(3), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::BilinearTensorProductInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), MakeMetaTensor(*input_weight), MakeMetaTensor(input_bias), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("bilinear_tensor_product compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, *input_weight, input_bias, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor bitwise_and(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "bitwise_and API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "bitwise_and", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "bitwise_and kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("bitwise_and compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor bitwise_not(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "bitwise_not API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "bitwise_not", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "bitwise_not kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("bitwise_not compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor bitwise_or(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "bitwise_or API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "bitwise_or", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "bitwise_or kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("bitwise_or compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor bitwise_xor(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "bitwise_xor API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "bitwise_xor", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "bitwise_xor kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("bitwise_xor compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor bmm(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "bmm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "bmm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "bmm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::BmmInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("bmm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor box_coder(const Tensor& prior_box, const paddle::optional<Tensor>& prior_box_var, const Tensor& target_box, const std::string& code_type, bool box_normalized, int axis, const std::vector<float>& variance) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(prior_box, prior_box_var, target_box);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "box_coder API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "box_coder", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "box_coder kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_prior_box = PrepareData(prior_box, kernel.InputAt(0), {});
  auto input_prior_box_var = PrepareData(prior_box_var, kernel.InputAt(1), {});
  auto input_target_box = PrepareData(target_box, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::BoxCoderInferMeta(MakeMetaTensor(*input_prior_box), MakeMetaTensor(input_prior_box_var), MakeMetaTensor(*input_target_box), code_type, box_normalized, axis, variance, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const phi::DenseTensor&, const std::string&, bool, int, const std::vector<float>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("box_coder compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_prior_box, input_prior_box_var, *input_target_box, code_type, box_normalized, axis, variance, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor brelu(const Tensor& x, float t_min, float t_max) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "brelu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "brelu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "brelu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("brelu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, t_min, t_max, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor cast(const Tensor& x, DataType out_dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cast API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cast", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "cast kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CastInferMeta(MakeMetaTensor(*input_x), out_dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cast compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, out_dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor ceil(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "ceil API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "ceil", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "ceil kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("ceil compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor celu(const Tensor& x, float alpha) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "celu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "celu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "celu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("celu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, alpha, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor clip(const Tensor& x, const Scalar& min, const Scalar& max) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "clip API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "clip", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "clip kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::Scalar&, const phi::Scalar&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("clip compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(min), phi::Scalar(max), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor& clip_(Tensor& x, const Scalar& min, const Scalar& max) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "clip API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "clip", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "clip kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor& api_output = x;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::Scalar&, const phi::Scalar&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("clip compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(min), phi::Scalar(max), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor clip_by_norm(const Tensor& x, float max_norm) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "clip_by_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "clip_by_norm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "clip_by_norm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ClipByNormInferMeta(MakeMetaTensor(*input_x), max_norm, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("clip_by_norm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, max_norm, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor complex(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "complex API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "complex", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "complex kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ComplexInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("complex compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor concat(const std::vector<Tensor>& x, const Scalar& axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "concat API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "concat", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "concat kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x_vec = PrepareData(x, kernel.InputAt(0), {});
  std::vector<const phi::DenseTensor*> input_x(input_x_vec->size());
  for (size_t i = 0; i < input_x.size(); ++i) {
    input_x[i] = &input_x_vec->at(i);
  }

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);

  auto x_meta_vec = MakeMetaTensor(input_x);
  std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
  for (size_t i = 0; i < x_meta_vec.size(); ++i) {
    x_metas[i] = &x_meta_vec[i];
  }
  phi::MetaTensor meta_out(kernel_out);

  phi::ConcatInferMeta(x_metas, axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<const phi::DenseTensor*>&, const phi::Scalar&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("concat compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, input_x, phi::Scalar(axis), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor conj(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "conj API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "conj", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "conj kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("conj compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor conv2d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& paddding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, bool use_addto, int workspace_size_MB, bool exhaustive_search) {
  return conv2d_impl(input, filter, strides, paddings, paddding_algorithm, groups, dilations, data_format, use_addto, workspace_size_MB, exhaustive_search);
}
PADDLE_API Tensor conv2d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, filter);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "conv2d_transpose API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "conv2d_transpose", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "conv2d_transpose kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_filter = PrepareData(filter, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ConvTransposeInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_filter), strides, paddings, output_padding, output_size, padding_algorithm, groups, dilations, data_format, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::string&, int, const std::vector<int>&, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("conv2d_transpose compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_filter, strides, paddings, output_padding, output_size, padding_algorithm, groups, dilations, data_format, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor conv3d(const Tensor& input, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& paddding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, bool use_addto, int workspace_size_MB, bool exhaustive_search) {
  return conv3d_impl(input, filter, strides, paddings, paddding_algorithm, groups, dilations, data_format, use_addto, workspace_size_MB, exhaustive_search);
}
PADDLE_API Tensor conv3d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, filter);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "conv3d_transpose API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "conv3d_transpose", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "conv3d_transpose kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_filter = PrepareData(filter, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ConvTransposeInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_filter), strides, paddings, output_padding, output_size, padding_algorithm, groups, dilations, data_format, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::string&, int, const std::vector<int>&, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("conv3d_transpose compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_filter, strides, paddings, output_padding, output_size, padding_algorithm, groups, dilations, data_format, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor copy_to(const Tensor& x, const Place& place, bool blocking) {
  return copy_to_impl(x, place, blocking);
}
PADDLE_API Tensor cos(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cos API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cos", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "cos kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cos compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor cosh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cosh API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cosh", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "cosh kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cosh compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor crop_tensor(const Tensor& x, const IntArray& shape, const IntArray& offsets) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "crop_tensor API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "crop_tensor", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "crop_tensor kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CropTensorInferMeta(MakeMetaTensor(*input_x), shape, offsets, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, const phi::IntArray&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("crop_tensor compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(shape), phi::IntArray(offsets), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> cross_entropy_with_softmax(const Tensor& input, const Tensor& label, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(input);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, label);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cross_entropy_with_softmax API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cross_entropy_with_softmax", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "cross_entropy_with_softmax kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});
  auto input_label = PrepareData(label, kernel.InputAt(1), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::CrossEntropyWithSoftmaxInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_label), soft_label, use_softmax, numeric_stable_mode, ignore_index, axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, bool, bool, bool, int, int, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cross_entropy_with_softmax compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, *input_label, soft_label, use_softmax, numeric_stable_mode, ignore_index, axis, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor cumprod(const Tensor& x, int dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cumprod API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cumprod", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "cumprod kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cumprod compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor cumsum(const Tensor& x, int axis, bool flatten, bool exclusive, bool reverse) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "cumsum API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "cumsum", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "cumsum kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CumInferMeta(MakeMetaTensor(*input_x), axis, flatten, exclusive, reverse, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, bool, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("cumsum compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, flatten, exclusive, reverse, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor deformable_conv(const Tensor& x, const Tensor& offset, const Tensor& filter, const paddle::optional<Tensor>& mask, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations, int deformable_groups, int groups, int im2col_step) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, offset, filter, mask);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "deformable_conv API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "deformable_conv", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "deformable_conv kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_offset = PrepareData(offset, kernel.InputAt(1), {});
  auto input_filter = PrepareData(filter, kernel.InputAt(2), {});
  auto input_mask = PrepareData(mask, kernel.InputAt(3), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::DeformableConvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_offset), MakeMetaTensor(*input_filter), MakeMetaTensor(input_mask), strides, paddings, dilations, deformable_groups, groups, im2col_step, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, int, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("deformable_conv compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_offset, *input_filter, input_mask, strides, paddings, dilations, deformable_groups, groups, im2col_step, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor depthwise_conv2d(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, bool use_addto, int workspace_size_MB, bool exhaustive_search, bool fuse_relu, bool use_gpudnn) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, filter);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "depthwise_conv2d API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "depthwise_conv2d", {kernel_backend, kernel_layout, kernel_data_type}, use_gpudnn);
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "depthwise_conv2d kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_filter = PrepareData(filter, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ConvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_filter), strides, paddings, padding_algorithm, groups, dilations, data_format, use_addto, workspace_size_MB, exhaustive_search, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::string&, int, const std::vector<int>&, const std::string&, bool, int, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("depthwise_conv2d compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_filter, strides, paddings, padding_algorithm, groups, dilations, data_format, use_addto, workspace_size_MB, exhaustive_search, fuse_relu, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor depthwise_conv2d_transpose(const Tensor& x, const Tensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& output_padding, const std::vector<int>& output_size, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, filter);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "depthwise_conv2d_transpose API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "depthwise_conv2d_transpose", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "depthwise_conv2d_transpose kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_filter = PrepareData(filter, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ConvTransposeInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_filter), strides, paddings, output_padding, output_size, padding_algorithm, groups, dilations, data_format, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::string&, int, const std::vector<int>&, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("depthwise_conv2d_transpose compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_filter, strides, paddings, output_padding, output_size, padding_algorithm, groups, dilations, data_format, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor det(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "det API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "determinant", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "determinant kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("determinant compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor diag_embed(const Tensor& x, int offset, int dim1, int dim2) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "diag_embed API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "diag_embed", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "diag_embed kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::DiagEmbedInferMeta(MakeMetaTensor(*input_x), offset, dim1, dim2, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("diag_embed compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, offset, dim1, dim2, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor divide(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "divide API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "divide", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "divide kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("divide compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> dropout(const Tensor& x, const paddle::optional<Tensor>& seed_tensor, float p, bool is_test, const std::string& mode, int seed, bool fix_seed) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, seed_tensor);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "dropout API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "dropout", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "dropout kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_seed_tensor = PrepareData(seed_tensor, kernel.InputAt(1), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::DropoutInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_seed_tensor), p, is_test, mode, seed, fix_seed, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, float, bool, const std::string&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("dropout compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, input_seed_tensor, p, is_test, mode, seed, fix_seed, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> eigh(const Tensor& x, const std::string& uplo) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "eigh API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "eigh", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "eigh kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::EighInferMeta(MakeMetaTensor(*input_x), uplo, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::string&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("eigh compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, uplo, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor eigvals(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "eigvals API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "eigvals", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "eigvals kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::EigvalsInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("eigvals compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> einsum(const std::vector<Tensor>& x, const std::string& equation) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "einsum API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "einsum_raw", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "einsum_raw kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x_vec = PrepareData(x, kernel.InputAt(0), {});
  std::vector<const phi::DenseTensor*> input_x(input_x_vec->size());
  for (size_t i = 0; i < input_x.size(); ++i) {
    input_x[i] = &input_x_vec->at(i);
  }

  std::tuple<Tensor, std::vector<Tensor>, std::vector<Tensor>> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(x.size(), kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(x.size(), kernel_backend, &std::get<2>(api_output));

  auto x_meta_vec = MakeMetaTensor(input_x);
  std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
  for (size_t i = 0; i < x_meta_vec.size(); ++i) {
    x_metas[i] = &x_meta_vec[i];
  }
  phi::MetaTensor meta_out_0(kernel_out_0);

  auto kernel_out_1_meta_vec = MakeMetaTensor(kernel_out_1);
  std::vector<phi::MetaTensor*> kernel_out_1_metas(kernel_out_1_meta_vec.size());
  for (size_t i = 0; i < kernel_out_1_meta_vec.size(); ++i) {
    kernel_out_1_metas[i] = kernel_out_1[i] ? &kernel_out_1_meta_vec[i] : nullptr;
  }
  auto kernel_out_2_meta_vec = MakeMetaTensor(kernel_out_2);
  std::vector<phi::MetaTensor*> kernel_out_2_metas(kernel_out_2_meta_vec.size());
  for (size_t i = 0; i < kernel_out_2_meta_vec.size(); ++i) {
    kernel_out_2_metas[i] = kernel_out_2[i] ? &kernel_out_2_meta_vec[i] : nullptr;
  }
  phi::EinsumRawInferMeta(x_metas, equation, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1_metas, kernel_out_2_metas);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<const phi::DenseTensor*>&, const std::string&, phi::DenseTensor*, std::vector<phi::DenseTensor*>&, std::vector<phi::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("einsum_raw compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, input_x, equation, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API Tensor elementwise_pow(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "elementwise_pow API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "elementwise_pow", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "elementwise_pow kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("elementwise_pow compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor elu(const Tensor& x, float alpha) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "elu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "elu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "elu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("elu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, alpha, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor embedding(const Tensor& x, const Tensor& weight, int64_t padding_idx, bool sparse) {
  return embedding_impl(x, weight, padding_idx, sparse);
}
PADDLE_API Tensor empty(const IntArray& shape, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);


  VLOG(6) << "empty API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "empty", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "empty kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CreateInferMeta(shape, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::IntArray&, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("empty compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, phi::IntArray(shape), dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor empty_like(const Tensor& x, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackendWithInputOrder(place, x);

  kernel_data_type = ParseDataTypeWithInputOrder(dtype, x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "empty_like API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "empty_like", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "empty_like kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CreateLikeInferMeta(MakeMetaTensor(*input_x), dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("empty_like compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor equal(const Tensor& x, const Tensor& y, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "equal API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "equal", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "equal kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CompareInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("equal compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor equal_all(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "equal_all API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "equal_all", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "equal_all kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CompareAllInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("equal_all compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor erfinv(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "erfinv API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "erfinv", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "erfinv kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("erfinv compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor& erfinv_(Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "erfinv API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "erfinv", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "erfinv kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor& api_output = x;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("erfinv compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor exp(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "exp API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "exp", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "exp kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("exp compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor expand(const Tensor& x, const IntArray& shape) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "expand API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "expand", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "expand kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ExpandInferMeta(MakeMetaTensor(*input_x), shape, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("expand compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(shape), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor expand_as(const Tensor& x, const paddle::optional<Tensor>& y, const std::vector<int>& target_shape) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "expand_as API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "expand_as", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "expand_as kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ExpandAsInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_y), target_shape, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const std::vector<int>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("expand_as compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, input_y, target_shape, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor expm1(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "expm1 API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "expm1", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "expm1 kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("expm1 compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor& exponential_(Tensor& x, float lambda) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "exponential_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "exponential", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "exponential kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor& api_output = x;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("exponential compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, lambda, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor eye(int64_t num_rows, int64_t num_columns, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);


  VLOG(6) << "eye API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "eye", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "eye kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::EyeInferMeta(num_rows, num_columns, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, int64_t, int64_t, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("eye compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, num_rows, num_columns, dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor flatten(const Tensor& x, int start_axis, int stop_axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "flatten API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "flatten_with_xshape", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "flatten_with_xshape kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  kernel_out_0->ShareBufferWith(*input_x);
  kernel_out_0->ShareInplaceVersionCounterWith(*input_x);
  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::FlattenWithXShapeInferMeta(MakeMetaTensor(*input_x), start_axis, stop_axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("flatten_with_xshape compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, start_axis, stop_axis, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor& flatten_(Tensor& x, int start_axis, int stop_axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "flatten API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "flatten_with_xshape", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "flatten_with_xshape kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor&, Tensor> api_output{x, Tensor()};
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::FlattenWithXShapeInferMeta(MakeMetaTensor(*input_x), start_axis, stop_axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("flatten_with_xshape compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, start_axis, stop_axis, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor flip(const Tensor& x, const std::vector<int>& axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "flip API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "flip", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "flip kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::FlipInferMeta(MakeMetaTensor(*input_x), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("flip compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor floor(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "floor API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "floor", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "floor kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("floor compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor floor_divide(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "floor_divide API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "floor_divide", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "floor_divide kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("floor_divide compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor fmax(const Tensor& x, const Tensor& y, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "fmax API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fmax", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "fmax kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("fmax compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor fmin(const Tensor& x, const Tensor& y, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "fmin API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "fmin", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "fmin kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("fmin compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor frame(const Tensor& x, int frame_length, int hop_length, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "frame API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "frame", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "frame kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::FrameInferMeta(MakeMetaTensor(*input_x), frame_length, hop_length, axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("frame compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, frame_length, hop_length, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor frobenius_norm(const Tensor& x, const std::vector<int64_t>& axis, bool keep_dim, bool reduce_all) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "frobenius_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "frobenius_norm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "frobenius_norm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReduceInferMetaBase(MakeMetaTensor(*input_x), axis, keep_dim, reduce_all, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("frobenius_norm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, keep_dim, reduce_all, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor full(const IntArray& shape, const Scalar& value, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);


  VLOG(6) << "full API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "full", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "full kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CreateInferMeta(shape, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::IntArray&, const phi::Scalar&, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("full compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, phi::IntArray(shape), phi::Scalar(value), dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor full_batch_size_like(const Tensor& input, const std::vector<int>& shape, DataType dtype, const Scalar& value, int input_dim_idx, int output_dim_idx, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "full_batch_size_like API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "full_batch_size_like", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "full_batch_size_like kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::FullBatchSizeLikeInferMeta(MakeMetaTensor(*input_input), shape, value, dtype, input_dim_idx, output_dim_idx, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, const phi::Scalar&, DataType, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("full_batch_size_like compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, shape, phi::Scalar(value), dtype, input_dim_idx, output_dim_idx, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor full_like(const Tensor& x, const Scalar& value, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackendWithInputOrder(place, x);

  kernel_data_type = ParseDataTypeWithInputOrder(dtype, x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "full_like API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "full_like", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "full_like kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {true});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CreateLikeInferMeta(MakeMetaTensor(*input_x), dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::Scalar&, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("full_like compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(value), dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor gather(const Tensor& x, const Tensor& index, const Scalar& axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, index);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "gather API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "gather", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "gather kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_index = PrepareData(index, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::GatherInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_index), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::Scalar&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("gather compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_index, phi::Scalar(axis), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor gather_nd(const Tensor& x, const Tensor& index) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, index);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "gather_nd API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "gather_nd", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "gather_nd kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_index = PrepareData(index, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::GatherNdInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_index), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("gather_nd compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_index, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor gather_tree(const Tensor& ids, const Tensor& parents) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(ids, parents);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "gather_tree API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "gather_tree", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "gather_tree kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_ids = PrepareData(ids, kernel.InputAt(0), {});
  auto input_parents = PrepareData(parents, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::GatherTreeMeta(MakeMetaTensor(*input_ids), MakeMetaTensor(*input_parents), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("gather_tree compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_ids, *input_parents, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor gaussian_random(const IntArray& shape, float mean, float std, int seed, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);


  VLOG(6) << "gaussian_random API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "gaussian_random", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "gaussian_random kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::GaussianRandomInferMeta(shape, mean, std, seed, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::IntArray&, float, float, int, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("gaussian_random compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, phi::IntArray(shape), mean, std, seed, dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor gelu(const Tensor& x, bool approximate) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "gelu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "gelu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "gelu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("gelu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, approximate, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor graph_send_recv(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& pool_type, int64_t out_size) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, src_index, dst_index);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "graph_send_recv API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "graph_send_recv", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "graph_send_recv kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_src_index = PrepareData(src_index, kernel.InputAt(1), {});
  auto input_dst_index = PrepareData(dst_index, kernel.InputAt(2), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::GraphSendRecvInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_src_index), MakeMetaTensor(*input_dst_index), pool_type, out_size, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, int64_t, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("graph_send_recv compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_src_index, *input_dst_index, pool_type, out_size, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor greater_equal(const Tensor& x, const Tensor& y, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "greater_equal API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "greater_equal", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "greater_equal kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CompareInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("greater_equal compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor greater_than(const Tensor& x, const Tensor& y, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "greater_than API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "greater_than", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "greater_than kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CompareInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("greater_than compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor grid_sample(const Tensor& x, const Tensor& grid, const std::string& mode, const std::string& padding_mode, bool align_corners) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, grid);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "grid_sample API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "grid_sample", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "grid_sample kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_grid = PrepareData(grid, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::GridSampleBaseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_grid), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, const std::string&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("grid_sample compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_grid, mode, padding_mode, align_corners, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor group_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int groups, const std::string& data_layout) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, scale, bias);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "group_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "group_norm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "group_norm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_scale = PrepareData(scale, kernel.InputAt(1), {});
  auto input_bias = PrepareData(bias, kernel.InputAt(2), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::GroupNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_scale), MakeMetaTensor(input_bias), epsilon, groups, data_layout, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, int, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("group_norm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, input_scale, input_bias, epsilon, groups, data_layout, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor gumbel_softmax(const Tensor& x, float temperature, bool hard, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "gumbel_softmax API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "gumbel_softmax", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "gumbel_softmax kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::GumbelSoftmaxInferMeta(MakeMetaTensor(*input_x), temperature, hard, axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, bool, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("gumbel_softmax compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, temperature, hard, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor hard_shrink(const Tensor& x, float threshold) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "hard_shrink API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "hard_shrink", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "hard_shrink kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("hard_shrink compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, threshold, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor hard_sigmoid(const Tensor& x, float slope, float offset) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "hard_sigmoid API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "hard_sigmoid", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "hard_sigmoid kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("hard_sigmoid compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, slope, offset, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor hard_swish(const Tensor& x, float threshold, float scale, float offset) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "hard_swish API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "hard_swish", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "hard_swish kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, float, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("hard_swish compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, threshold, scale, offset, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> hierarchical_sigmoid(const Tensor& x, const Tensor& w, const Tensor& label, const paddle::optional<Tensor>& path, const paddle::optional<Tensor>& code, const paddle::optional<Tensor>& bias, int num_classes, bool remote_prefetch, int trainer_id, const std::vector<int64_t>& height_sections, const std::vector<std::string>& epmap, const std::vector<std::string>& table_names, bool is_sparse) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, w, label, path, code, bias);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "hierarchical_sigmoid API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "hierarchical_sigmoid", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "hierarchical_sigmoid kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_w = PrepareData(w, kernel.InputAt(1), {});
  auto input_label = PrepareData(label, kernel.InputAt(2), {});
  auto input_path = PrepareData(path, kernel.InputAt(3), {});
  auto input_code = PrepareData(code, kernel.InputAt(4), {});
  auto input_bias = PrepareData(bias, kernel.InputAt(5), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::HierarchicalSigmoidInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_w), MakeMetaTensor(*input_label), MakeMetaTensor(input_path), MakeMetaTensor(input_code), MakeMetaTensor(input_bias), num_classes, remote_prefetch, trainer_id, height_sections, epmap, table_names, is_sparse, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, int, bool, int, const std::vector<int64_t>&, const std::vector<std::string>&, const std::vector<std::string>&, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("hierarchical_sigmoid compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_w, *input_label, input_path, input_code, input_bias, num_classes, remote_prefetch, trainer_id, height_sections, epmap, table_names, is_sparse, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API Tensor histogram(const Tensor& x, int64_t bins, int min, int max) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "histogram API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "histogram", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "histogram kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::HistogramInferMeta(MakeMetaTensor(*input_x), bins, min, max, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int64_t, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("histogram compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, bins, min, max, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> huber_loss(const Tensor& input, const Tensor& label, float delta) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, label);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "huber_loss API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "huber_loss", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "huber_loss kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});
  auto input_label = PrepareData(label, kernel.InputAt(1), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::HuberLossInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_label), delta, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, float, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("huber_loss compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, *input_label, delta, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor imag(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "imag API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "imag", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "imag kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::RealAndImagInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("imag compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor increment(const Tensor& x, float value) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "increment API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "increment", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "increment kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::IncrementInferMeta(MakeMetaTensor(*input_x), value, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("increment compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, value, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor index_sample(const Tensor& x, const Tensor& index) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, index);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "index_sample API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "index_sample", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "index_sample kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_index = PrepareData(index, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::IndexSampleInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_index), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("index_sample compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_index, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor index_select(const Tensor& x, const Tensor& index, int dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, index);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "index_select API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "index_select", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "index_select kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_index = PrepareData(index, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::IndexSelectInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_index), dim, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("index_select compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_index, dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor instance_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, scale, bias);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "instance_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "instance_norm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "instance_norm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_scale = PrepareData(scale, kernel.InputAt(1), {});
  auto input_bias = PrepareData(bias, kernel.InputAt(2), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::InstanceNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_scale), MakeMetaTensor(input_bias), epsilon, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("instance_norm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, input_scale, input_bias, epsilon, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor inverse(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "inverse API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "inverse", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "inverse kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::InverseInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("inverse compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor is_empty(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "is_empty API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "is_empty", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "is_empty kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::IsEmptyInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("is_empty compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor isclose(const Tensor& x, const Tensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "isclose API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "isclose", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "isclose kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ValueCompareInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::Scalar&, const phi::Scalar&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("isclose compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, phi::Scalar(rtol), phi::Scalar(atol), equal_nan, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor isfinite(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (x.is_dense_tensor()) {

    VLOG(6) << "isfinite API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "isfinite", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "isfinite kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = PrepareData(x, kernel.InputAt(0), {});

    Tensor api_output;
    auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::IsfiniteInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("isfinite compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  if (x.is_selected_rows()) {

    VLOG(6) << "isfinite API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "infinite_sr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "infinite_sr kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = TensorToSelectedRows(x);

    Tensor api_output;
    auto kernel_out = SetSelectedRowsKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::IsfiniteInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::SelectedRows&, phi::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("infinite_sr compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (isfinite) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor isinf(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (x.is_dense_tensor()) {

    VLOG(6) << "isinf API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "isinf", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "isinf kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = PrepareData(x, kernel.InputAt(0), {});

    Tensor api_output;
    auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::IsfiniteInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("isinf compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  if (x.is_selected_rows()) {

    VLOG(6) << "isinf API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "isinf_sr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "isinf_sr kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = TensorToSelectedRows(x);

    Tensor api_output;
    auto kernel_out = SetSelectedRowsKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::IsfiniteInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::SelectedRows&, phi::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("isinf_sr compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (isinf) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor isnan(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (x.is_dense_tensor()) {

    VLOG(6) << "isnan API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "isnan", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "isnan kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = PrepareData(x, kernel.InputAt(0), {});

    Tensor api_output;
    auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::IsfiniteInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("isnan compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  if (x.is_selected_rows()) {

    VLOG(6) << "isnan API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "isnan_sr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "isnan_sr kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = TensorToSelectedRows(x);

    Tensor api_output;
    auto kernel_out = SetSelectedRowsKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::IsfiniteInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::SelectedRows&, phi::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("isnan_sr compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (isnan) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor kldiv_loss(const Tensor& x, const Tensor& label, const std::string& reduction) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, label);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "kldiv_loss API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "kldiv_loss", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "kldiv_loss kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_label = PrepareData(label, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::KLDivInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_label), reduction, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("kldiv_loss compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_label, reduction, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor kron(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "kron API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "kron", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "kron kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::KronInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("kron compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> kthvalue(const Tensor& x, int k, int axis, bool keepdim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "kthvalue API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "kthvalue", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "kthvalue kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::KthvalueInferMeta(MakeMetaTensor(*input_x), k, axis, keepdim, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("kthvalue compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, k, axis, keepdim, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor label_smooth(const Tensor& label, const paddle::optional<Tensor>& prior_dist, float epsilon) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(label);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(label, prior_dist);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "label_smooth API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "label_smooth", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "label_smooth kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_label = PrepareData(label, kernel.InputAt(0), {});
  auto input_prior_dist = PrepareData(prior_dist, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_label), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("label_smooth compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_label, input_prior_dist, epsilon, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> layer_norm(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int begin_norm_axis, bool is_test) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, scale, bias);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "layer_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "layer_norm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "layer_norm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_scale = PrepareData(scale, kernel.InputAt(1), {});
  auto input_bias = PrepareData(bias, kernel.InputAt(2), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::LayerNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(input_scale), MakeMetaTensor(input_bias), epsilon, begin_norm_axis, is_test, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, float, int, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("layer_norm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, input_scale, input_bias, epsilon, begin_norm_axis, is_test, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API Tensor leaky_relu(const Tensor& x, float alpha) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "leaky_relu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "leaky_relu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "leaky_relu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("leaky_relu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, alpha, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor lerp(const Tensor& x, const Tensor& y, const Tensor& weight) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, weight);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "lerp API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "lerp", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "lerp kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});
  auto input_weight = PrepareData(weight, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::LerpInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), MakeMetaTensor(*input_weight), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("lerp compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, *input_weight, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor less_equal(const Tensor& x, const Tensor& y, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "less_equal API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "less_equal", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "less_equal kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CompareInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("less_equal compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor less_than(const Tensor& x, const Tensor& y, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "less_than API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "less_than", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "less_than kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CompareInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("less_than compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor linspace(const Tensor& start, const Tensor& stop, const Tensor& number, DataType dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(dtype);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(start, stop, number);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "linspace API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "linspace", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "linspace kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_start = PrepareData(start, kernel.InputAt(0), {});
  auto input_stop = PrepareData(stop, kernel.InputAt(1), {});
  auto input_number = PrepareData(number, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::LinspaceInferMeta(MakeMetaTensor(*input_start), MakeMetaTensor(*input_stop), MakeMetaTensor(*input_number), dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("linspace compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_start, *input_stop, *input_number, dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor log(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "log API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "log", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "log kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("log compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor log10(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "log10 API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "log10", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "log10 kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("log10 compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor log1p(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "log1p API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "log1p", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "log1p kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("log1p compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor log2(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "log2 API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "log2", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "log2 kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("log2 compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor log_loss(const Tensor& input, const Tensor& label, float epsilon) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, label);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "log_loss API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "log_loss", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "log_loss kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});
  auto input_label = PrepareData(label, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::LogLossInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_label), epsilon, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("log_loss compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, *input_label, epsilon, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor log_softmax(const Tensor& x, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "log_softmax API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "log_softmax", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "log_softmax kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMetaCheckAxis(MakeMetaTensor(*input_x), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("log_softmax compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor logcumsumexp(const Tensor& x, int axis, bool flatten, bool exclusive, bool reverse) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "logcumsumexp API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "logcumsumexp", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "logcumsumexp kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CumInferMeta(MakeMetaTensor(*input_x), axis, flatten, exclusive, reverse, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, bool, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("logcumsumexp compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, flatten, exclusive, reverse, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor logical_and(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "logical_and API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "logical_and", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "logical_and kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("logical_and compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor logical_not(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "logical_not API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "logical_not", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "logical_not kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("logical_not compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor logical_or(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "logical_or API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "logical_or", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "logical_or kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("logical_or compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor logical_xor(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "logical_xor API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "logical_xor", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "logical_xor kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("logical_xor compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor logit(const Tensor& x, float eps) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "logit API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "logit", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "logit kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("logit compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, eps, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor logsigmoid(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "logsigmoid API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "logsigmoid", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "logsigmoid kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("logsigmoid compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor logsumexp(const Tensor& x, const std::vector<int64_t>& axis, bool keepdim, bool reduce_all) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "logsumexp API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "logsumexp", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "logsumexp kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::LogsumexpInferMeta(MakeMetaTensor(*input_x), axis, keepdim, reduce_all, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("logsumexp compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, keepdim, reduce_all, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> lstsq(const Tensor& x, const Tensor& y, const Scalar& rcond, const std::string& driver) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "lstsq API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "lstsq", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "lstsq kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  std::tuple<Tensor, Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(kernel_backend, &std::get<3>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);
  phi::MetaTensor meta_out_3(kernel_out_3);

  phi::LstsqInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), rcond, driver, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::Scalar&, const std::string&, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("lstsq compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, phi::Scalar(rcond), driver, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> lu(const Tensor& x, bool pivot) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "lu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "lu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "lu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::LUInferMeta(MakeMetaTensor(*input_x), pivot, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("lu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, pivot, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> lu_unpack(const Tensor& x, const Tensor& pivots, bool unpack_ludata, bool unpack_pivots) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, pivots);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "lu_unpack API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "lu_unpack", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "lu_unpack kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_pivots = PrepareData(pivots, kernel.InputAt(1), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::LUUnpackInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_pivots), unpack_ludata, unpack_pivots, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, bool, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("lu_unpack compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_pivots, unpack_ludata, unpack_pivots, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API Tensor masked_select(const Tensor& x, const Tensor& mask) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, mask);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "masked_select API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "masked_select", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "masked_select kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_mask = PrepareData(mask, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::MaskedSelectInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_mask), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("masked_select compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_mask, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor matmul(const Tensor& x, const Tensor& y, bool transpose_x, bool transpose_y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "matmul API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "matmul", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "matmul kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::MatmulInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), transpose_x, transpose_y, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("matmul compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, transpose_x, transpose_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor matrix_power(const Tensor& x, int n) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "matrix_power API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "matrix_power", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "matrix_power kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("matrix_power compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, n, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor matrix_rank(const Tensor& x, float tol, bool use_default_tol, bool hermitian) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "matrix_rank API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "matrix_rank", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "matrix_rank kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::MatrixRankInferMeta(MakeMetaTensor(*input_x), use_default_tol, hermitian, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("matrix_rank compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, tol, use_default_tol, hermitian, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor matrix_rank_tol(const Tensor& x, const Tensor& atol_tensor, bool use_default_tol, bool hermitian) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, atol_tensor);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "matrix_rank_tol API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "matrix_rank_tol", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "matrix_rank_tol kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_atol_tensor = PrepareData(atol_tensor, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::MatrixRankTolInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_atol_tensor), use_default_tol, hermitian, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("matrix_rank_tol compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_atol_tensor, use_default_tol, hermitian, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor max(const Tensor& x, const std::vector<int64_t>& dims, bool keep_dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "max API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "max", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "max kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReduceInferMeta(MakeMetaTensor(*input_x), dims, keep_dim, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("max compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dims, keep_dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> max_pool2d_with_index(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "max_pool2d_with_index API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "max_pool2d_with_index", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "max_pool2d_with_index kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::MaxPoolWithIndexInferMeta(MakeMetaTensor(*input_x), kernel_size, strides, paddings, global_pooling, adaptive, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, bool, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("max_pool2d_with_index compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_size, strides, paddings, global_pooling, adaptive, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> max_pool3d_with_index(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool global_pooling, bool adaptive) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "max_pool3d_with_index API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "max_pool3d_with_index", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "max_pool3d_with_index kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::MaxPoolWithIndexInferMeta(MakeMetaTensor(*input_x), kernel_size, strides, paddings, global_pooling, adaptive, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, bool, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("max_pool3d_with_index compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_size, strides, paddings, global_pooling, adaptive, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor maximum(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "maximum API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "maximum", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "maximum kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("maximum compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor maxout(const Tensor& x, int groups, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "maxout API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "maxout", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "maxout kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::MaxOutInferMeta(MakeMetaTensor(*input_x), groups, axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("maxout compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, groups, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor mean(const Tensor& x, const std::vector<int64_t>& dims, bool keep_dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "mean API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "mean", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "mean kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReduceInferMeta(MakeMetaTensor(*input_x), dims, keep_dim, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("mean compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dims, keep_dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor mean_all(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "mean_all API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "mean_all", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "mean_all kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::MeanAllInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("mean_all compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::vector<Tensor> meshgrid(const std::vector<Tensor>& inputs) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(inputs);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "meshgrid API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "meshgrid", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "meshgrid kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_inputs_vec = PrepareData(inputs, kernel.InputAt(0), {});
  std::vector<const phi::DenseTensor*> input_inputs(input_inputs_vec->size());
  for (size_t i = 0; i < input_inputs.size(); ++i) {
    input_inputs[i] = &input_inputs_vec->at(i);
  }

  std::vector<Tensor> api_output;
  auto kernel_out = SetKernelOutput(inputs.size(), kernel_backend, &api_output);

  auto inputs_meta_vec = MakeMetaTensor(input_inputs);
  std::vector<const phi::MetaTensor*> inputs_metas(inputs_meta_vec.size());
  for (size_t i = 0; i < inputs_meta_vec.size(); ++i) {
    inputs_metas[i] = &inputs_meta_vec[i];
  }

  auto kernel_out_meta_vec = MakeMetaTensor(kernel_out);
  std::vector<phi::MetaTensor*> kernel_out_metas(kernel_out_meta_vec.size());
  for (size_t i = 0; i < kernel_out_meta_vec.size(); ++i) {
    kernel_out_metas[i] = kernel_out[i] ? &kernel_out_meta_vec[i] : nullptr;
  }
  phi::MeshgridInferMeta(inputs_metas, kernel_out_metas);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<const phi::DenseTensor*>&, std::vector<phi::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("meshgrid compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, input_inputs, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor min(const Tensor& x, const std::vector<int64_t>& dims, bool keep_dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "min API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "min", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "min kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReduceInferMeta(MakeMetaTensor(*input_x), dims, keep_dim, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("min compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dims, keep_dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor minimum(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "minimum API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "minimum", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "minimum kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("minimum compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor mish(const Tensor& x, float lambda) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "mish API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "mish", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "mish kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("mish compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, lambda, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> mode(const Tensor& x, int axis, bool keepdim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "mode API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "mode", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "mode kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::ModeInferMeta(MakeMetaTensor(*input_x), axis, keepdim, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("mode compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, keepdim, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor modulo(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "modulo API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "modulo", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "modulo kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("modulo compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> momentum(const Tensor& param, const Tensor& grad, const Tensor& velocity, const Tensor& learning_rate, const paddle::optional<Tensor>& master_param, float mu, bool use_nesterov, const std::string& regularization_method, float regularization_coeff, bool multi_precision, float rescale_grad) {
  return momentum_impl(param, grad, velocity, learning_rate, master_param, mu, use_nesterov, regularization_method, regularization_coeff, multi_precision, rescale_grad);
}
PADDLE_API Tensor multi_dot(const std::vector<Tensor>& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "multi_dot API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "multi_dot", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "multi_dot kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x_vec = PrepareData(x, kernel.InputAt(0), {});
  std::vector<const phi::DenseTensor*> input_x(input_x_vec->size());
  for (size_t i = 0; i < input_x.size(); ++i) {
    input_x[i] = &input_x_vec->at(i);
  }

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);

  auto x_meta_vec = MakeMetaTensor(input_x);
  std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
  for (size_t i = 0; i < x_meta_vec.size(); ++i) {
    x_metas[i] = &x_meta_vec[i];
  }
  phi::MetaTensor meta_out(kernel_out);

  phi::MultiDotInferMeta(x_metas, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<const phi::DenseTensor*>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("multi_dot compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor multinomial(const Tensor& x, int num_samples, bool replacement) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "multinomial API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "multinomial", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "multinomial kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::MultinomialInferMeta(MakeMetaTensor(*input_x), num_samples, replacement, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("multinomial compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, num_samples, replacement, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor multiplex(const std::vector<Tensor>& ins, const Tensor& ids) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(ins);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(ins, ids);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "multiplex API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "multiplex", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "multiplex kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_ins_vec = PrepareData(ins, kernel.InputAt(0), {});
  std::vector<const phi::DenseTensor*> input_ins(input_ins_vec->size());
  for (size_t i = 0; i < input_ins.size(); ++i) {
    input_ins[i] = &input_ins_vec->at(i);
  }
  auto input_ids = PrepareData(ids, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);

  auto ins_meta_vec = MakeMetaTensor(input_ins);
  std::vector<const phi::MetaTensor*> ins_metas(ins_meta_vec.size());
  for (size_t i = 0; i < ins_meta_vec.size(); ++i) {
    ins_metas[i] = &ins_meta_vec[i];
  }
  phi::MetaTensor meta_out(kernel_out);

  phi::MultiplexInferMeta(ins_metas, MakeMetaTensor(*input_ids), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<const phi::DenseTensor*>&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("multiplex compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, input_ins, *input_ids, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "multiply API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "multiply", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "multiply kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("multiply compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> nll_loss(const Tensor& input, const Tensor& label, const paddle::optional<Tensor>& weight, int64_t ignore_index, const std::string& reduction) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(input);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, label, weight);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "nll_loss API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "nll_loss", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "nll_loss kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});
  auto input_label = PrepareData(label, kernel.InputAt(1), {});
  auto input_weight = PrepareData(weight, kernel.InputAt(2), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::NllLossRawInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_label), MakeMetaTensor(input_weight), ignore_index, reduction, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, int64_t, const std::string&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("nll_loss compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, *input_label, input_weight, ignore_index, reduction, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor nms(const Tensor& x, float threshold) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "nms API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "nms", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "nms kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::NMSInferMeta(MakeMetaTensor(*input_x), threshold, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("nms compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, threshold, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor norm(const Tensor& x, int axis, float epsilon, bool is_test) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "norm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "norm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::NormInferMeta(MakeMetaTensor(*input_x), axis, epsilon, is_test, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, float, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("norm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, epsilon, is_test, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor not_equal(const Tensor& x, const Tensor& y, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "not_equal API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "not_equal", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "not_equal kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::CompareInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("not_equal compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor one_hot(const Tensor& x, const Scalar& num_classes) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "one_hot API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "one_hot", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "one_hot kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::OneHotInferMeta(MakeMetaTensor(*input_x), num_classes, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::Scalar&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("one_hot compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(num_classes), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor ones_like(const Tensor& x, DataType dtype, const Place& place) {
  return full_like(x, 1, dtype, place);
}
PADDLE_API Tensor p_norm(const Tensor& x, float porder, int axis, float epsilon, bool keepdim, bool asvector) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "p_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "p_norm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "p_norm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::PNormInferMeta(MakeMetaTensor(*input_x), porder, axis, epsilon, keepdim, asvector, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, int, float, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("p_norm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, porder, axis, epsilon, keepdim, asvector, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor pad(const Tensor& x, const std::vector<int>& paddings, float pad_value) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "pad API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "pad", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "pad kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::PadInferMeta(MakeMetaTensor(*input_x), paddings, pad_value, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("pad compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, paddings, pad_value, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor pad3d(const Tensor& x, const IntArray& paddings, const std::string& mode, float pad_value, const std::string& data_format) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "pad3d API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "pad3d", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "pad3d kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::Pad3dInferMeta(MakeMetaTensor(*input_x), paddings, mode, pad_value, data_format, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, const std::string&, float, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("pad3d compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(paddings), mode, pad_value, data_format, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor pixel_shuffle(const Tensor& x, int upscale_factor, const std::string& data_format) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "pixel_shuffle API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "pixel_shuffle", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "pixel_shuffle kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::PixelShuffleInferMeta(MakeMetaTensor(*input_x), upscale_factor, data_format, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("pixel_shuffle compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, upscale_factor, data_format, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor pool2d(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "pool2d API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "pool2d", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "pool2d kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::PoolInferMeta(MakeMetaTensor(*input_x), kernel_size, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, bool, bool, const std::string&, const std::string&, bool, bool, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("pool2d compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_size, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor pool2d_gpudnn_unused(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "pool2d_gpudnn_unused API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "pool2d", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "pool2d kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::PoolInferMeta(MakeMetaTensor(*input_x), kernel_size, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, bool, bool, const std::string&, const std::string&, bool, bool, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("pool2d compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_size, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor pool3d(const Tensor& x, const std::vector<int>& kernel_size, const std::vector<int>& strides, const std::vector<int>& paddings, bool ceil_mode, bool exclusive, const std::string& data_format, const std::string& pooling_type, bool global_pooling, bool adaptive, const std::string& padding_algorithm) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "pool3d API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "pool3d", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "pool3d kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::PoolInferMeta(MakeMetaTensor(*input_x), kernel_size, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, bool, bool, const std::string&, const std::string&, bool, bool, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("pool3d compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_size, strides, paddings, ceil_mode, exclusive, data_format, pooling_type, global_pooling, adaptive, padding_algorithm, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor pow(const Tensor& x, const Scalar& s) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "pow API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "pow", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "pow kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::Scalar&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("pow compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(s), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor prelu(const Tensor& x, const Tensor& alpha, const std::string& data_format, const std::string& mode) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, alpha);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "prelu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "prelu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "prelu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_alpha = PrepareData(alpha, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::PReluInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_alpha), data_format, mode, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("prelu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_alpha, data_format, mode, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> prior_box(const Tensor& input, const Tensor& image, const std::vector<float>& min_sizes, const std::vector<float>& aspect_ratios, const std::vector<float>& variances, const std::vector<float>& max_sizes, bool flip, bool clip, float step_w, float step_h, float offset, bool min_max_aspect_ratios_order) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, image);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "prior_box API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "prior_box", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "prior_box kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});
  auto input_image = PrepareData(image, kernel.InputAt(1), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::PriorBoxInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_image), min_sizes, aspect_ratios, variances, max_sizes, flip, clip, step_w, step_h, offset, min_max_aspect_ratios_order, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, bool, bool, float, float, float, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("prior_box compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, *input_image, min_sizes, aspect_ratios, variances, max_sizes, flip, clip, step_w, step_h, offset, min_max_aspect_ratios_order, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor psroi_pool(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, int output_channels, float spatial_scale) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, boxes, boxes_num);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "psroi_pool API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "psroi_pool", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "psroi_pool kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_boxes = PrepareData(boxes, kernel.InputAt(1), {});
  auto input_boxes_num = PrepareData(boxes_num, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::PsroiPoolInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_boxes), MakeMetaTensor(input_boxes_num), pooled_height, pooled_width, output_channels, spatial_scale, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, int, int, int, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("psroi_pool compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_boxes, input_boxes_num, pooled_height, pooled_width, output_channels, spatial_scale, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor put_along_axis(const Tensor& x, const Tensor& index, const Tensor& value, int axis, const std::string& reduce) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, index, value);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "put_along_axis API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "put_along_axis", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "put_along_axis kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_index = PrepareData(index, kernel.InputAt(1), {});
  auto input_value = PrepareData(value, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_index), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, int, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("put_along_axis compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_index, *input_value, axis, reduce, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> qr(const Tensor& x, const std::string& mode) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "qr API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "qr", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "qr kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::QrInferMeta(MakeMetaTensor(*input_x), mode, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::string&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("qr compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, mode, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor randint(int low, int high, const IntArray& shape, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);


  VLOG(6) << "randint API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "randint", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "randint kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::RandintInferMeta(low, high, shape, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, int, int, const phi::IntArray&, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("randint compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, low, high, phi::IntArray(shape), dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor randperm(int n, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);


  VLOG(6) << "randperm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "randperm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "randperm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::RandpermInferMeta(n, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, int, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("randperm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, n, dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor real(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "real API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "real", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "real kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::RealAndImagInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("real compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor reciprocal(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "reciprocal API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "reciprocal", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "reciprocal kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("reciprocal compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor reduce_prod(const Tensor& x, const std::vector<int64_t>& dims, bool keep_dim, bool reduce_all) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "reduce_prod API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "prod_raw", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "prod_raw kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReduceInferMetaBase(MakeMetaTensor(*input_x), dims, keep_dim, reduce_all, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("prod_raw compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dims, keep_dim, reduce_all, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor relu(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "relu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "relu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "relu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("relu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor& relu_(Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "relu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "relu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "relu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor& api_output = x;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("relu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor relu6(const Tensor& x, float threshold) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "relu6 API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "relu6", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "relu6 kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("relu6 compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, threshold, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor reshape(const Tensor& x, const IntArray& shape) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "reshape API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "reshape_with_xshape", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "reshape_with_xshape kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  kernel_out_0->ShareBufferWith(*input_x);
  kernel_out_0->ShareInplaceVersionCounterWith(*input_x);
  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::ReshapeWithXShapeInferMeta(MakeMetaTensor(*input_x), shape, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("reshape_with_xshape compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(shape), kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor& reshape_(Tensor& x, const IntArray& shape) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "reshape API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "reshape_with_xshape", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "reshape_with_xshape kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor&, Tensor> api_output{x, Tensor()};
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::ReshapeWithXShapeInferMeta(MakeMetaTensor(*input_x), shape, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("reshape_with_xshape compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(shape), kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor reverse(const Tensor& x, const std::vector<int>& axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "reverse API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "reverse", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "reverse kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ReverseInferMeta(MakeMetaTensor(*input_x), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("reverse compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::vector<Tensor> reverse_array(const std::vector<Tensor>& x, const std::vector<int>& axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "reverse_array API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "reverse_array", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "reverse_array kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x_vec = PrepareData(x, kernel.InputAt(0), {});
  std::vector<const phi::DenseTensor*> input_x(input_x_vec->size());
  for (size_t i = 0; i < input_x.size(); ++i) {
    input_x[i] = &input_x_vec->at(i);
  }

  std::vector<Tensor> api_output;
  auto kernel_out = SetKernelOutput(x.size(), kernel_backend, &api_output);

  auto x_meta_vec = MakeMetaTensor(input_x);
  std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
  for (size_t i = 0; i < x_meta_vec.size(); ++i) {
    x_metas[i] = &x_meta_vec[i];
  }

  auto kernel_out_meta_vec = MakeMetaTensor(kernel_out);
  std::vector<phi::MetaTensor*> kernel_out_metas(kernel_out_meta_vec.size());
  for (size_t i = 0; i < kernel_out_meta_vec.size(); ++i) {
    kernel_out_metas[i] = kernel_out[i] ? &kernel_out_meta_vec[i] : nullptr;
  }
  phi::ReverseArrayInferMeta(x_metas, axis, kernel_out_metas);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<const phi::DenseTensor*>&, const std::vector<int>&, std::vector<phi::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("reverse_array compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, input_x, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> rmsprop_(Tensor& param, Tensor& mean_square, const Tensor& grad, Tensor& moment, const Tensor& learning_rate, Tensor& mean_grad, float epsilon, float decay, float momentum, bool centered) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(param, mean_square, grad, moment, learning_rate, mean_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (param.is_dense_tensor() && mean_square.is_dense_tensor() && grad.is_dense_tensor() && moment.is_dense_tensor() && learning_rate.is_dense_tensor() && mean_grad.is_dense_tensor()) {

    VLOG(6) << "rmsprop_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "rmsprop", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "rmsprop kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_param = PrepareData(param, kernel.InputAt(0), {});
    auto input_mean_square = PrepareData(mean_square, kernel.InputAt(1), {});
    auto input_grad = PrepareData(grad, kernel.InputAt(2), {});
    auto input_moment = PrepareData(moment, kernel.InputAt(3), {});
    auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(4), {});
    auto input_mean_grad = PrepareData(mean_grad, kernel.InputAt(5), {});

    std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> api_output{param, moment, mean_square, mean_grad};
    auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
    auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
    auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
    auto kernel_out_3 = SetKernelOutput(kernel_backend, &std::get<3>(api_output));
    phi::MetaTensor meta_out_0(kernel_out_0);
    phi::MetaTensor meta_out_1(kernel_out_1);
    phi::MetaTensor meta_out_2(kernel_out_2);
    phi::MetaTensor meta_out_3(kernel_out_3);

    phi::RmspropInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_mean_square), MakeMetaTensor(*input_grad), MakeMetaTensor(*input_moment), MakeMetaTensor(*input_learning_rate), MakeMetaTensor(*input_mean_grad), epsilon, decay, momentum, centered, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, float, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("rmsprop compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_param, *input_mean_square, *input_grad, *input_moment, *input_learning_rate, *input_mean_grad, epsilon, decay, momentum, centered, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
      TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
      TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
    }

    return api_output;
  }

  if (param.is_dense_tensor() && mean_square.is_dense_tensor() && grad.is_selected_rows() && moment.is_dense_tensor() && learning_rate.is_dense_tensor() && mean_grad.is_dense_tensor()) {

    VLOG(6) << "rmsprop_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "rmsprop_dense_param_sparse_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "rmsprop_dense_param_sparse_grad kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_param = PrepareData(param, kernel.InputAt(0), {});
    auto input_mean_square = PrepareData(mean_square, kernel.InputAt(1), {});
    auto input_grad = TensorToSelectedRows(grad);
    auto input_moment = PrepareData(moment, kernel.InputAt(3), {});
    auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(4), {});
    auto input_mean_grad = PrepareData(mean_grad, kernel.InputAt(5), {});

    std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> api_output{param, moment, mean_square, mean_grad};
    auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
    auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
    auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
    auto kernel_out_3 = SetKernelOutput(kernel_backend, &std::get<3>(api_output));
    phi::MetaTensor meta_out_0(kernel_out_0);
    phi::MetaTensor meta_out_1(kernel_out_1);
    phi::MetaTensor meta_out_2(kernel_out_2);
    phi::MetaTensor meta_out_3(kernel_out_3);

    phi::RmspropInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_mean_square), MakeMetaTensor(*input_grad), MakeMetaTensor(*input_moment), MakeMetaTensor(*input_learning_rate), MakeMetaTensor(*input_mean_grad), epsilon, decay, momentum, centered, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::SelectedRows&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, float, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("rmsprop_dense_param_sparse_grad compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_param, *input_mean_square, *input_grad, *input_moment, *input_learning_rate, *input_mean_grad, epsilon, decay, momentum, centered, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
      TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
      TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (rmsprop_) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor roi_align(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, boxes, boxes_num);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "roi_align API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "roi_align", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "roi_align kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_boxes = PrepareData(boxes, kernel.InputAt(1), {});
  auto input_boxes_num = PrepareData(boxes_num, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::RoiAlignInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_boxes), MakeMetaTensor(input_boxes_num), pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, int, int, float, int, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("roi_align compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_boxes, input_boxes_num, pooled_height, pooled_width, spatial_scale, sampling_ratio, aligned, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor roi_pool(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, float spatial_scale) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, boxes, boxes_num);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "roi_pool API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "roi_pool", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "roi_pool kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_boxes = PrepareData(boxes, kernel.InputAt(1), {});
  auto input_boxes_num = PrepareData(boxes_num, kernel.InputAt(2), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::RoiPoolInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_boxes), MakeMetaTensor(input_boxes_num), pooled_height, pooled_width, spatial_scale, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, int, int, float, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("roi_pool compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_boxes, input_boxes_num, pooled_height, pooled_width, spatial_scale, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor roll(const Tensor& x, const IntArray& shifts, const std::vector<int64_t>& axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "roll API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "roll", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "roll kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::RollInferMeta(MakeMetaTensor(*input_x), shifts, axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, const std::vector<int64_t>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("roll compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(shifts), axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor round(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "round API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "round", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "round kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("round compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor rsqrt(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "rsqrt API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "rsqrt", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "rsqrt kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("rsqrt compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor& rsqrt_(Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "rsqrt API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "rsqrt", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "rsqrt kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor& api_output = x;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("rsqrt compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor scale(const Tensor& x, const Scalar& scale, float bias, bool bias_after_scale) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (x.is_dense_tensor()) {

    VLOG(6) << "scale API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "scale kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = PrepareData(x, kernel.InputAt(0), {});

    Tensor api_output;
    auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::Scalar&, float, bool, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("scale compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(scale), bias, bias_after_scale, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  if (x.is_selected_rows()) {

    VLOG(6) << "scale API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale_sr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "scale_sr kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = TensorToSelectedRows(x);

    Tensor api_output;
    auto kernel_out = SetSelectedRowsKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::SelectedRows&, const phi::Scalar&, float, bool, phi::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("scale_sr compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(scale), bias, bias_after_scale, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (scale) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor& scale_(Tensor& x, const Scalar& scale, float bias, bool bias_after_scale) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (x.is_dense_tensor()) {

    VLOG(6) << "scale API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "scale kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = PrepareData(x, kernel.InputAt(0), {});

    Tensor& api_output = x;
    auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::Scalar&, float, bool, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("scale compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(scale), bias, bias_after_scale, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  if (x.is_selected_rows()) {

    VLOG(6) << "scale API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale_sr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "scale_sr kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_x = TensorToSelectedRows(x);

    Tensor& api_output = x;
    auto kernel_out = SetSelectedRowsKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::SelectedRows&, const phi::Scalar&, float, bool, phi::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("scale_sr compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(scale), bias, bias_after_scale, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (scale) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor scatter(const Tensor& x, const Tensor& index, const Tensor& updates, bool overwrite) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, index, updates);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "scatter API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "scatter", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "scatter kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_index = PrepareData(index, kernel.InputAt(1), {});
  auto input_updates = PrepareData(updates, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ScatterInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_index), MakeMetaTensor(*input_updates), overwrite, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("scatter compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_index, *input_updates, overwrite, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor scatter_nd_add(const Tensor& x, const Tensor& index, const Tensor& updates) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, index, updates);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "scatter_nd_add API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "scatter_nd_add", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "scatter_nd_add kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_index = PrepareData(index, kernel.InputAt(1), {});
  auto input_updates = PrepareData(updates, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ScatterNdAddInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_index), MakeMetaTensor(*input_updates), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("scatter_nd_add compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_index, *input_updates, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor searchsorted(const Tensor& sorted_sequence, const Tensor& value, bool out_int32, bool right) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(sorted_sequence);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(sorted_sequence, value);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "searchsorted API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "searchsorted", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "searchsorted kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_sorted_sequence = PrepareData(sorted_sequence, kernel.InputAt(0), {});
  auto input_value = PrepareData(value, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::SearchsortedInferMeta(MakeMetaTensor(*input_sorted_sequence), MakeMetaTensor(*input_value), out_int32, right, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("searchsorted compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_sorted_sequence, *input_value, out_int32, right, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> segment_pool(const Tensor& x, const Tensor& segment_ids, const std::string& pooltype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, segment_ids);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "segment_pool API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "segment_pool", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "segment_pool kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_segment_ids = PrepareData(segment_ids, kernel.InputAt(1), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::SegmentPoolInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_segment_ids), pooltype, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::string&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("segment_pool compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_segment_ids, pooltype, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor selu(const Tensor& x, float scale, float alpha) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "selu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "selu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "selu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("selu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, scale, alpha, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor&, paddle::optional<Tensor>&> sgd_(Tensor& param, const Tensor& learning_rate, const Tensor& grad, paddle::optional<Tensor>& master_param, bool multi_precision) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(param);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(param, learning_rate, grad, master_param);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (param.is_dense_tensor() && learning_rate.is_dense_tensor() && grad.is_dense_tensor() && (!master_param || master_param->is_dense_tensor())) {

    VLOG(6) << "sgd_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sgd", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "sgd kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_param = PrepareData(param, kernel.InputAt(0), {});
    auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(1), {false, true});
    auto input_grad = PrepareData(grad, kernel.InputAt(2), {});
    auto input_master_param = PrepareData(master_param, kernel.InputAt(3), {});

    std::tuple<Tensor&, paddle::optional<Tensor>&> api_output{param, master_param};
    auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
    auto kernel_out_1 = SetKernelOutput(kernel_backend, std::get<1>(api_output).get_ptr());
    phi::MetaTensor meta_out_0(kernel_out_0);
    phi::MetaTensor meta_out_1(kernel_out_1);

    phi::SgdInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_learning_rate), MakeMetaTensor(*input_grad), MakeMetaTensor(input_master_param), multi_precision, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, bool, phi::DenseTensor*, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("sgd compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_param, *input_learning_rate, *input_grad, input_master_param, multi_precision, kernel_out_0, kernel_out_1);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    }

    return api_output;
  }

  if (param.is_dense_tensor() && learning_rate.is_dense_tensor() && grad.is_selected_rows() && (!master_param || master_param->is_dense_tensor())) {

    VLOG(6) << "sgd_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sgd_dense_param_sparse_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "sgd_dense_param_sparse_grad kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_param = PrepareData(param, kernel.InputAt(0), {});
    auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(1), {false, true});
    auto input_grad = TensorToSelectedRows(grad);
    auto input_master_param = PrepareData(master_param, kernel.InputAt(3), {});

    std::tuple<Tensor&, paddle::optional<Tensor>&> api_output{param, master_param};
    auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
    auto kernel_out_1 = SetKernelOutput(kernel_backend, std::get<1>(api_output).get_ptr());
    phi::MetaTensor meta_out_0(kernel_out_0);
    phi::MetaTensor meta_out_1(kernel_out_1);

    phi::SgdInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_learning_rate), MakeMetaTensor(*input_grad), MakeMetaTensor(input_master_param), multi_precision, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::SelectedRows&, const paddle::optional<phi::DenseTensor>&, bool, phi::DenseTensor*, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("sgd_dense_param_sparse_grad compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_param, *input_learning_rate, *input_grad, input_master_param, multi_precision, kernel_out_0, kernel_out_1);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    }

    return api_output;
  }

  if (param.is_selected_rows() && learning_rate.is_dense_tensor() && grad.is_selected_rows() && (!master_param || master_param->is_selected_rows())) {

    VLOG(6) << "sgd_ API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sgd_sparse_param_sparse_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "sgd_sparse_param_sparse_grad kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_param = TensorToSelectedRows(param);
    auto input_learning_rate = PrepareData(learning_rate, kernel.InputAt(1), {false, true});
    auto input_grad = TensorToSelectedRows(grad);
    auto input_master_param = TensorToSelectedRows(master_param);

    std::tuple<Tensor&, paddle::optional<Tensor>&> api_output{param, master_param};
    auto kernel_out_0 = SetSelectedRowsKernelOutput(kernel_backend, &std::get<0>(api_output));
    auto kernel_out_1 = SetSelectedRowsKernelOutput(kernel_backend, std::get<1>(api_output).get_ptr());
    phi::MetaTensor meta_out_0(kernel_out_0);
    phi::MetaTensor meta_out_1(kernel_out_1);

    phi::SgdInferMeta(MakeMetaTensor(*input_param), MakeMetaTensor(*input_learning_rate), MakeMetaTensor(*input_grad), MakeMetaTensor(input_master_param), multi_precision, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::SelectedRows&, const phi::DenseTensor&, const phi::SelectedRows&, const paddle::optional<phi::SelectedRows>&, bool, phi::SelectedRows*, phi::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("sgd_sparse_param_sparse_grad compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_param, *input_learning_rate, *input_grad, input_master_param, multi_precision, kernel_out_0, kernel_out_1);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
      TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sgd_) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor shape(const Tensor& input) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }


  if (input.is_dense_tensor()) {

    VLOG(6) << "shape API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "shape", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "shape kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_input = PrepareData(input, kernel.InputAt(0), {true});

    Tensor api_output;
    auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::ShapeInferMeta(MakeMetaTensor(*input_input), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("shape compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_input, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  if (input.is_selected_rows()) {

    VLOG(6) << "shape API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "shape_sr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& kernel = kernel_result.kernel;
    VLOG(6) << "shape_sr kernel: " << kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

    auto input_input = TensorToSelectedRows(input);

    Tensor api_output;
    auto kernel_out = SetSelectedRowsKernelOutput(kernel_backend, &api_output);
    phi::MetaTensor meta_out(kernel_out);

    phi::ShapeInferMeta(MakeMetaTensor(*input_input), &meta_out);


    using kernel_signature = void(*)(const platform::DeviceContext&, const phi::SelectedRows&, phi::SelectedRows*);
    auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    {
      paddle::platform::RecordEvent kernel_record_event("shape_sr compute", paddle::platform::TracerEventType::OperatorInner, 1);
      (*kernel_fn)(*dev_ctx, *input_input, kernel_out);
    }
    if (kernel_result.has_fallback_cpu) {

      TransDataBackend(kernel_out, kernel_backend, kernel_out);
    }

    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (shape) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor shard_index(const Tensor& in, int index_num, int nshards, int shard_id, int ignore_value) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(in);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "shard_index API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "shard_index", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "shard_index kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_in = PrepareData(in, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ShardIndexInferMeta(MakeMetaTensor(*input_in), index_num, nshards, shard_id, ignore_value, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("shard_index compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_in, index_num, nshards, shard_id, ignore_value, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor sigmoid(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "sigmoid API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "sigmoid", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "sigmoid kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("sigmoid compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor sigmoid_cross_entropy_with_logits(const Tensor& x, const Tensor& label, bool normalize, int ignore_index) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, label);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "sigmoid_cross_entropy_with_logits API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "sigmoid_cross_entropy_with_logits", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "sigmoid_cross_entropy_with_logits kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_label = PrepareData(label, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::SigmoidCrossEntropyWithLogitsInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_label), normalize, ignore_index, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, bool, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("sigmoid_cross_entropy_with_logits compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_label, normalize, ignore_index, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor sign(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "sign API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "sign", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "sign kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("sign compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor silu(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "silu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "silu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "silu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("silu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor sin(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "sin API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "sin", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "sin kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("sin compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor sinh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "sinh API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "sinh", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "sinh kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("sinh compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor size(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "size API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "size", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "size kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {true});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::SizeInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("size compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor slice(const Tensor& input, const std::vector<int64_t>& axes, const IntArray& starts, const IntArray& ends, const std::vector<int64_t>& infer_flags, const std::vector<int64_t>& decrease_axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "slice API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "slice", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "slice kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::SliceRawInferMeta(MakeMetaTensor(*input_input), axes, starts, ends, infer_flags, decrease_axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, const phi::IntArray&, const phi::IntArray&, const std::vector<int64_t>&, const std::vector<int64_t>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("slice compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, axes, phi::IntArray(starts), phi::IntArray(ends), infer_flags, decrease_axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor slogdet(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "slogdet API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "slogdeterminant", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "slogdeterminant kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("slogdeterminant compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor soft_shrink(const Tensor& x, float lambda) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "soft_shrink API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "soft_shrink", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "soft_shrink kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("soft_shrink compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, lambda, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor softmax(const Tensor& x, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "softmax API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "softmax", {kernel_backend, kernel_layout, kernel_data_type}, true);
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "softmax kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::SoftmaxInferMeta(MakeMetaTensor(*input_x), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("softmax compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor softplus(const Tensor& x, float beta, float threshold) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "softplus API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "softplus", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "softplus kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("softplus compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, beta, threshold, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor softsign(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "softsign API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "softsign", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "softsign kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("softsign compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor spectral_norm(const Tensor& weight, const Tensor& u, const Tensor& v, int dim, int power_iters, float eps) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(weight);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(weight, u, v);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "spectral_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "spectralnorm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "spectralnorm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_weight = PrepareData(weight, kernel.InputAt(0), {});
  auto input_u = PrepareData(u, kernel.InputAt(1), {});
  auto input_v = PrepareData(v, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::SpectralNormInferMeta(MakeMetaTensor(*input_weight), MakeMetaTensor(*input_u), MakeMetaTensor(*input_v), dim, power_iters, eps, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, int, int, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("spectralnorm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_weight, *input_u, *input_v, dim, power_iters, eps, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::vector<Tensor> split(const Tensor& x, const IntArray& num_or_sections, const Scalar& axis) {
  return split_impl(x, num_or_sections, axis);
}
PADDLE_API Tensor sqrt(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "sqrt API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "sqrt", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "sqrt kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("sqrt compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor square(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "square API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "square", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "square kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("square compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor squared_l2_norm(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "squared_l2_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "squared_l2_norm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "squared_l2_norm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::SquaredL2NormInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("squared_l2_norm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor squeeze(const Tensor& x, const std::vector<int>& axes) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "squeeze API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "squeeze_with_xshape", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "squeeze_with_xshape kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  kernel_out_0->ShareBufferWith(*input_x);
  kernel_out_0->ShareInplaceVersionCounterWith(*input_x);
  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::SqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axes, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("squeeze_with_xshape compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axes, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor stack(const std::vector<Tensor>& x, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "stack API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "stack", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "stack kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x_vec = PrepareData(x, kernel.InputAt(0), {});
  std::vector<const phi::DenseTensor*> input_x(input_x_vec->size());
  for (size_t i = 0; i < input_x.size(); ++i) {
    input_x[i] = &input_x_vec->at(i);
  }

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);

  auto x_meta_vec = MakeMetaTensor(input_x);
  std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
  for (size_t i = 0; i < x_meta_vec.size(); ++i) {
    x_metas[i] = &x_meta_vec[i];
  }
  phi::MetaTensor meta_out(kernel_out);

  phi::StackInferMeta(x_metas, axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<const phi::DenseTensor*>&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("stack compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, input_x, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor strided_slice(const Tensor& x, const std::vector<int>& axes, const IntArray& starts, const IntArray& ends, const IntArray& strides) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "strided_slice API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "strided_slice", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "strided_slice kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::StridedSliceInferMeta(MakeMetaTensor(*input_x), axes, starts, ends, strides, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, const phi::IntArray&, const phi::IntArray&, const phi::IntArray&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("strided_slice compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axes, phi::IntArray(starts), phi::IntArray(ends), phi::IntArray(strides), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "subtract API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "subtract", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "subtract kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::ElementwiseInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("subtract compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor sum(const Tensor& x, const std::vector<int64_t>& dims, DataType out_dtype, bool keep_dim) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "sum API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "sum", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "sum kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::SumInferMeta(MakeMetaTensor(*input_x), dims, out_dtype, keep_dim, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int64_t>&, DataType, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("sum compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, dims, out_dtype, keep_dim, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& x, bool full_metrices) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "svd API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "svd", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "svd kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::SvdInferMeta(MakeMetaTensor(*input_x), full_metrices, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("svd compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, full_metrices, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API Tensor swish(const Tensor& x, float beta) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "swish API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "swish", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "swish kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("swish compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, beta, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> sync_batch_norm(const Tensor& x, const Tensor& scale, const Tensor& bias, const Tensor& mean, const Tensor& variance, float momentum, float epsilon, const std::string& data_layout, bool is_test, bool use_global_stats, bool trainable_statistics, bool fuse_with_relu) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, scale, bias, mean, variance);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "sync_batch_norm API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "sync_batch_norm", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "sync_batch_norm kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_scale = PrepareData(scale, kernel.InputAt(1), {});
  auto input_bias = PrepareData(bias, kernel.InputAt(2), {});
  auto input_mean = PrepareData(mean, kernel.InputAt(3), {});
  auto input_variance = PrepareData(variance, kernel.InputAt(4), {});

  std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(kernel_backend, &std::get<3>(api_output));
  auto kernel_out_4 = SetKernelOutput(kernel_backend, &std::get<4>(api_output));
  auto kernel_out_5 = SetKernelOutput(kernel_backend, &std::get<5>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);
  phi::MetaTensor meta_out_3(kernel_out_3);
  phi::MetaTensor meta_out_4(kernel_out_4);
  phi::MetaTensor meta_out_5(kernel_out_5);

  phi::BatchNormInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_scale), MakeMetaTensor(*input_bias), MakeMetaTensor(*input_mean), MakeMetaTensor(*input_variance), momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics, fuse_with_relu, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr, kernel_out_4 ? &meta_out_4 : nullptr, kernel_out_5 ? &meta_out_5 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, float, float, const std::string&, bool, bool, bool, bool, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("sync_batch_norm compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_scale, *input_bias, *input_mean, *input_variance, momentum, epsilon, data_layout, is_test, use_global_stats, trainable_statistics, fuse_with_relu, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3, kernel_out_4, kernel_out_5);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
    TransDataBackend(kernel_out_4, kernel_backend, kernel_out_4);
    TransDataBackend(kernel_out_5, kernel_backend, kernel_out_5);
  }

  return api_output;
}

PADDLE_API Tensor take_along_axis(const Tensor& x, const Tensor& index, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, index);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "take_along_axis API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "take_along_axis", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "take_along_axis kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_index = PrepareData(index, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_index), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("take_along_axis compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_index, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor tan(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "tan API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "tan", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "tan kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("tan compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor tanh(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "tanh API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "tanh", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "tanh kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("tanh compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor tanh_shrink(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "tanh_shrink API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "tanh_shrink", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "tanh_shrink kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("tanh_shrink compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor temporal_shift(const Tensor& x, int seg_num, float shift_ratio, const std::string& data_format_str) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "temporal_shift API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "temporal_shift", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "temporal_shift kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::TemporalShiftInferMeta(MakeMetaTensor(*input_x), seg_num, shift_ratio, data_format_str, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, float, const std::string&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("temporal_shift compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, seg_num, shift_ratio, data_format_str, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor thresholded_relu(const Tensor& x, float threshold) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "thresholded_relu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "thresholded_relu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "thresholded_relu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnchangedInferMeta(MakeMetaTensor(*input_x), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, float, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("thresholded_relu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, threshold, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor tile(const Tensor& x, const IntArray& repeat_times) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "tile API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "tile", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "tile kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::TileInferMeta(MakeMetaTensor(*input_x), repeat_times, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("tile compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(repeat_times), kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> top_k(const Tensor& x, const Scalar& k, int axis, bool largest, bool sorted) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "top_k API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "top_k", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "top_k kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::TopKInferMeta(MakeMetaTensor(*input_x), k, axis, largest, sorted, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::Scalar&, int, bool, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("top_k compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::Scalar(k), axis, largest, sorted, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor transpose(const Tensor& x, const std::vector<int>& axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "transpose API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "transpose", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "transpose kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::TransposeInferMeta(MakeMetaTensor(*input_x), axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("transpose compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor triangular_solve(const Tensor& x, const Tensor& y, bool upper, bool transpose, bool unitriangular) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "triangular_solve API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "triangular_solve", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "triangular_solve kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_y = PrepareData(y, kernel.InputAt(1), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::TriangularSolveInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), upper, transpose, unitriangular, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, bool, bool, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("triangular_solve compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_y, upper, transpose, unitriangular, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor tril_indices(int rows, int cols, int offset, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);


  VLOG(6) << "tril_indices API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "tril_indices", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "tril_indices kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::TrilIndicesInferMeta(rows, cols, offset, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, int, int, int, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("tril_indices compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, rows, cols, offset, dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor tril_triu(const Tensor& x, int diagonal, bool lower) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "tril_triu API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "tril_triu", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "tril_triu kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::TrilTriuInferMeta(MakeMetaTensor(*input_x), diagonal, lower, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, bool, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("tril_triu compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, diagonal, lower, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor truncated_gaussian_random(const std::vector<int>& shape, float mean, float std, int seed, DataType dtype, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);


  VLOG(6) << "truncated_gaussian_random API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "truncated_gaussian_random", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "truncated_gaussian_random kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::TruncatedGaussianRandomInferMeta(shape, mean, std, seed, dtype, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<int>&, float, float, int, DataType, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("truncated_gaussian_random compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, shape, mean, std, seed, dtype, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::vector<Tensor> unbind(const Tensor& input, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "unbind API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "unbind", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "unbind kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});

  std::vector<Tensor> api_output;
  auto kernel_out = SetKernelOutput(axis<0 ? input.dims()[input.dims().size()+axis]:input.dims()[axis], kernel_backend, &api_output);

  auto kernel_out_meta_vec = MakeMetaTensor(kernel_out);
  std::vector<phi::MetaTensor*> kernel_out_metas(kernel_out_meta_vec.size());
  for (size_t i = 0; i < kernel_out_meta_vec.size(); ++i) {
    kernel_out_metas[i] = kernel_out[i] ? &kernel_out_meta_vec[i] : nullptr;
  }
  phi::UnbindInferMeta(MakeMetaTensor(*input_input), axis, kernel_out_metas);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, std::vector<phi::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("unbind compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor unfold(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& strides, const std::vector<int>& paddings, const std::vector<int>& dilations) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "unfold API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "unfold", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "unfold kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UnfoldInferMeta(MakeMetaTensor(*input_x), kernel_sizes, strides, paddings, dilations, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("unfold compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_sizes, strides, paddings, dilations, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor uniform_random(const IntArray& shape, DataType dtype, float min, float max, int seed, const Place& place) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_backend = ParseBackend(place);

  kernel_data_type = ParseDataType(dtype);


  VLOG(6) << "uniform_random API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "uniform_random", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "uniform_random kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);


  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::UniformRandomInferMeta(shape, dtype, min, max, seed, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::IntArray&, DataType, float, float, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("uniform_random compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, phi::IntArray(shape), dtype, min, max, seed, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor, Tensor> unique(const Tensor& x, bool return_index, bool return_inverse, bool return_counts, const std::vector<int>& axis, DataType dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "unique API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "unique", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "unique kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  auto kernel_out_3 = SetKernelOutput(kernel_backend, &std::get<3>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);
  phi::MetaTensor meta_out_3(kernel_out_3);

  phi::UniqueInferMeta(MakeMetaTensor(*input_x), return_index, return_inverse, return_counts, axis, dtype, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr, kernel_out_3 ? &meta_out_3 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, bool, bool, bool, const std::vector<int>&, DataType, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("unique compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, return_index, return_inverse, return_counts, axis, dtype, kernel_out_0, kernel_out_1, kernel_out_2, kernel_out_3);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
    TransDataBackend(kernel_out_3, kernel_backend, kernel_out_3);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> unique_consecutive(const Tensor& x, bool return_inverse, bool return_counts, const std::vector<int>& axis, int dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "unique_consecutive API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "unique_consecutive", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "unique_consecutive kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  auto kernel_out_2 = SetKernelOutput(kernel_backend, &std::get<2>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);
  phi::MetaTensor meta_out_2(kernel_out_2);

  phi::UniqueConsecutiveInferMeta(MakeMetaTensor(*input_x), return_inverse, return_counts, axis, dtype, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr, kernel_out_2 ? &meta_out_2 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, bool, bool, const std::vector<int>&, int, phi::DenseTensor*, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("unique_consecutive compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, return_inverse, return_counts, axis, dtype, kernel_out_0, kernel_out_1, kernel_out_2);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
    TransDataBackend(kernel_out_2, kernel_backend, kernel_out_2);
  }

  return api_output;
}

PADDLE_API Tensor unsqueeze(const Tensor& x, const IntArray& axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "unsqueeze API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "unsqueeze_with_xshape", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "unsqueeze_with_xshape kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  kernel_out_0->ShareBufferWith(*input_x);
  kernel_out_0->ShareInplaceVersionCounterWith(*input_x);
  VLOG(3) << "Perform View between Output and Input Tensor, share allocation and inplace version.";
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::UnsqueezeWithXShapeInferMeta(MakeMetaTensor(*input_x), axis, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::IntArray&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("unsqueeze_with_xshape compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, phi::IntArray(axis), kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API std::vector<Tensor> unstack(const Tensor& x, int axis, int num) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "unstack API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "unstack", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "unstack kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::vector<Tensor> api_output;
  auto kernel_out = SetKernelOutput(num, kernel_backend, &api_output);

  auto kernel_out_meta_vec = MakeMetaTensor(kernel_out);
  std::vector<phi::MetaTensor*> kernel_out_metas(kernel_out_meta_vec.size());
  for (size_t i = 0; i < kernel_out_meta_vec.size(); ++i) {
    kernel_out_metas[i] = kernel_out[i] ? &kernel_out_meta_vec[i] : nullptr;
  }
  phi::UnStackInferMeta(MakeMetaTensor(*input_x), axis, num, kernel_out_metas);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, std::vector<phi::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("unstack compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, axis, num, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> viterbi_decode(const Tensor& input, const Tensor& transition, const Tensor& length, bool include_bos_eos_tag) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(input);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, transition, length);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "viterbi_decode API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "viterbi_decode", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "viterbi_decode kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_input = PrepareData(input, kernel.InputAt(0), {});
  auto input_transition = PrepareData(transition, kernel.InputAt(1), {});
  auto input_length = PrepareData(length, kernel.InputAt(2), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::ViterbiDecodeInferMeta(MakeMetaTensor(*input_input), MakeMetaTensor(*input_transition), MakeMetaTensor(*input_length), include_bos_eos_tag, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("viterbi_decode compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_input, *input_transition, *input_length, include_bos_eos_tag, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor warpctc(const Tensor& logits, const Tensor& label, const paddle::optional<Tensor>& logits_length, const paddle::optional<Tensor>& labels_length, int blank, bool norm_by_times) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(logits);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(logits, label, logits_length, labels_length);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "warpctc API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "warpctc", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "warpctc kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_logits = PrepareData(logits, kernel.InputAt(0), {});
  auto input_label = PrepareData(label, kernel.InputAt(1), {});
  auto input_logits_length = PrepareData(logits_length, kernel.InputAt(2), {});
  auto input_labels_length = PrepareData(labels_length, kernel.InputAt(3), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::WarpctcInferMeta(MakeMetaTensor(*input_logits), MakeMetaTensor(*input_label), MakeMetaTensor(input_logits_length), MakeMetaTensor(input_labels_length), blank, norm_by_times, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const paddle::optional<phi::DenseTensor>&, const paddle::optional<phi::DenseTensor>&, int, bool, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("warpctc compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_logits, *input_label, input_logits_length, input_labels_length, blank, norm_by_times, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return std::get<0>(api_output);
}

PADDLE_API Tensor where(const Tensor& condition, const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(condition, x, y);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "where API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "where", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "where kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_condition = PrepareData(condition, kernel.InputAt(0), {});
  auto input_x = PrepareData(x, kernel.InputAt(1), {});
  auto input_y = PrepareData(y, kernel.InputAt(2), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::WhereInferMeta(MakeMetaTensor(*input_condition), MakeMetaTensor(*input_x), MakeMetaTensor(*input_y), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("where compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_condition, *input_x, *input_y, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor where_index(const Tensor& condition) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(condition);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "where_index API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "where_index", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "where_index kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_condition = PrepareData(condition, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::WhereIndexInferMeta(MakeMetaTensor(*input_condition), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("where_index compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_condition, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> yolo_box(const Tensor& x, const Tensor& img_size, const std::vector<int>& anchors, int class_num, float conf_thresh, int downsample_ratio, bool clip_bbox, float scale_x_y, bool iou_aware, float iou_aware_factor) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, img_size);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "yolo_box API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "yolo_box", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "yolo_box kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});
  auto input_img_size = PrepareData(img_size, kernel.InputAt(1), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::YoloBoxInferMeta(MakeMetaTensor(*input_x), MakeMetaTensor(*input_img_size), anchors, class_num, conf_thresh, downsample_ratio, clip_bbox, scale_x_y, iou_aware, iou_aware_factor, kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, const phi::DenseTensor&, const std::vector<int>&, int, float, int, bool, float, bool, float, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("yolo_box compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, *input_img_size, anchors, class_num, conf_thresh, downsample_ratio, clip_bbox, scale_x_y, iou_aware, iou_aware_factor, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor zeros_like(const Tensor& x, DataType dtype, const Place& place) {
  return full_like(x, 0, dtype, place);
}
PADDLE_API std::vector<Tensor> broadcast_tensors(const std::vector<Tensor>& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "broadcast_tensors API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "broadcast_tensors", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "broadcast_tensors kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x_vec = PrepareData(x, kernel.InputAt(0), {});
  std::vector<const phi::DenseTensor*> input_x(input_x_vec->size());
  for (size_t i = 0; i < input_x.size(); ++i) {
    input_x[i] = &input_x_vec->at(i);
  }

  std::vector<Tensor> api_output;
  auto kernel_out = SetKernelOutput(x.size(), kernel_backend, &api_output);

  auto x_meta_vec = MakeMetaTensor(input_x);
  std::vector<const phi::MetaTensor*> x_metas(x_meta_vec.size());
  for (size_t i = 0; i < x_meta_vec.size(); ++i) {
    x_metas[i] = &x_meta_vec[i];
  }

  auto kernel_out_meta_vec = MakeMetaTensor(kernel_out);
  std::vector<phi::MetaTensor*> kernel_out_metas(kernel_out_meta_vec.size());
  for (size_t i = 0; i < kernel_out_meta_vec.size(); ++i) {
    kernel_out_metas[i] = kernel_out[i] ? &kernel_out_meta_vec[i] : nullptr;
  }
  phi::BroadcastTensorsInferMeta(x_metas, kernel_out_metas);


  using kernel_signature = void(*)(const platform::DeviceContext&, const std::vector<const phi::DenseTensor*>&, std::vector<phi::DenseTensor*>&);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("broadcast_tensors compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, input_x, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API Tensor dirichlet(const Tensor& alpha) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(alpha);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "dirichlet API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "dirichlet", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "dirichlet kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_alpha = PrepareData(alpha, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::DirichletInferMeta(MakeMetaTensor(*input_alpha), &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("dirichlet compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_alpha, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> eig(const Tensor& x) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "eig API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "eig", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "eig kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  std::tuple<Tensor, Tensor> api_output;
  auto kernel_out_0 = SetKernelOutput(kernel_backend, &std::get<0>(api_output));
  auto kernel_out_1 = SetKernelOutput(kernel_backend, &std::get<1>(api_output));
  phi::MetaTensor meta_out_0(kernel_out_0);
  phi::MetaTensor meta_out_1(kernel_out_1);

  phi::EigInferMeta(MakeMetaTensor(*input_x), kernel_out_0 ? &meta_out_0 : nullptr, kernel_out_1 ? &meta_out_1 : nullptr);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, phi::DenseTensor*, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("eig compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, kernel_out_0, kernel_out_1);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out_0, kernel_backend, kernel_out_0);
    TransDataBackend(kernel_out_1, kernel_backend, kernel_out_1);
  }

  return api_output;
}

PADDLE_API Tensor overlap_add(const Tensor& x, int hop_length, int axis) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  VLOG(6) << "overlap_add API kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
  auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
      "overlap_add", {kernel_backend, kernel_layout, kernel_data_type});
  const auto& kernel = kernel_result.kernel;
  VLOG(6) << "overlap_add kernel: " << kernel;

  auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);

  auto input_x = PrepareData(x, kernel.InputAt(0), {});

  Tensor api_output;
  auto kernel_out = SetKernelOutput(kernel_backend, &api_output);
  phi::MetaTensor meta_out(kernel_out);

  phi::OverlapAddInferMeta(MakeMetaTensor(*input_x), hop_length, axis, &meta_out);


  using kernel_signature = void(*)(const platform::DeviceContext&, const phi::DenseTensor&, int, int, phi::DenseTensor*);
  auto* kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
  {
    paddle::platform::RecordEvent kernel_record_event("overlap_add compute", paddle::platform::TracerEventType::OperatorInner, 1);
    (*kernel_fn)(*dev_ctx, *input_x, hop_length, axis, kernel_out);
  }
  if (kernel_result.has_fallback_cpu) {

    TransDataBackend(kernel_out, kernel_backend, kernel_out);
  }

  return api_output;
}


}  // namespace experimental
}  // namespace paddle
