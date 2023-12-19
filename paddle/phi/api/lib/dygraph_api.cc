#include "paddle/phi/api/lib/dygraph_api.h"

#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/lib/api_custom_impl.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/sparse_api_custom_impl.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"

#include "paddle/fluid/platform/profiler/event_tracing.h"

namespace paddle {
namespace experimental {


PADDLE_API std::tuple<Tensor, Tensor> flatten_intermediate(const Tensor& x, int start_axis, int stop_axis) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor&, Tensor> flatten_intermediate_(Tensor& x, int start_axis, int stop_axis) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> graph_send_recv_intermediate(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& pool_type, int64_t out_size) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> group_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int groups, const std::string& data_layout) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> instance_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> norm_intermediate(const Tensor& x, int axis, float epsilon, bool is_test) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> reshape_intermediate(const Tensor& x, const IntArray& shape) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor&, Tensor> reshape_intermediate_(Tensor& x, const IntArray& shape) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> roi_pool_intermediate(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, float spatial_scale) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> squeeze_intermediate(const Tensor& x, const std::vector<int>& axes) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> unsqueeze_intermediate(const Tensor& x, const IntArray& axis) {

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

  return api_output;
}

PADDLE_API std::tuple<Tensor, Tensor> warpctc_intermediate(const Tensor& logits, const Tensor& label, const paddle::optional<Tensor>& logits_length, const paddle::optional<Tensor>& labels_length, int blank, bool norm_by_times) {

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

  return api_output;
}

namespace sparse {

PADDLE_API std::tuple<Tensor, Tensor, Tensor> conv3d_intermediate(const Tensor& x, const Tensor& kernel, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, kernel);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(kernel.impl().get())) {

    VLOG(6) << "conv3d api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "conv3d_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "conv3d api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor, Tensor> api_output;
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::SPARSE_COO);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_2 = SetSparseKernelOutput(&std::get<2>(api_output), TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(kernel.impl().get());
    kernel_context.EmplaceBackAttr(paddings);
    kernel_context.EmplaceBackAttr(dilations);
    kernel_context.EmplaceBackAttr(strides);
    kernel_context.EmplaceBackAttr(groups);
    kernel_context.EmplaceBackAttr(subm);
    kernel_context.EmplaceBackAttr(key);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (conv3d) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API std::tuple<Tensor, Tensor> fused_attention_intermediate(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& sparse_mask, const paddle::optional<Tensor>& key_padding_mask, const paddle::optional<Tensor>& attn_mask) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(sparse_mask);

  kernel_data_type = ParseDataType(query);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(query, key, value, sparse_mask, key_padding_mask, attn_mask);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (phi::DenseTensor::classof(query.impl().get()) && phi::DenseTensor::classof(key.impl().get()) && phi::DenseTensor::classof(value.impl().get()) && sparse_mask.layout() == DataLayout::SPARSE_CSR && (!key_padding_mask || phi::DenseTensor::classof(key_padding_mask->impl().get())) && (!attn_mask || phi::DenseTensor::classof(attn_mask->impl().get()))) {

    VLOG(6) << "fused_attention api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "fused_attention_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "fused_attention api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor> api_output;
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(query.impl().get());
    kernel_context.EmplaceBackInput(key.impl().get());
    kernel_context.EmplaceBackInput(value.impl().get());
    kernel_context.EmplaceBackInput(sparse_mask.impl().get());
    kernel_context.EmplaceBackInput(key_padding_mask ? key_padding_mask->impl().get() : nullptr);
    kernel_context.EmplaceBackInput(attn_mask ? attn_mask->impl().get() : nullptr);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (fused_attention) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API std::tuple<Tensor, Tensor, Tensor> maxpool_intermediate(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

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

  if (x.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "maxpool api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "maxpool_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "maxpool api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    std::tuple<Tensor, Tensor, Tensor> api_output;
    auto* kernel_out_0 = SetSparseKernelOutput(&std::get<0>(api_output), TensorType::SPARSE_COO);
    auto* kernel_out_1 = SetSparseKernelOutput(&std::get<1>(api_output), TensorType::DENSE_TENSOR);
    auto* kernel_out_2 = SetSparseKernelOutput(&std::get<2>(api_output), TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(kernel_sizes);
    kernel_context.EmplaceBackAttr(paddings);
    kernel_context.EmplaceBackAttr(dilations);
    kernel_context.EmplaceBackAttr(strides);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (maxpool) for input tensors is unimplemented, please check the type of input tensors."));
}

}  // namespace sparse


}  // namespace experimental
}  // namespace paddle
