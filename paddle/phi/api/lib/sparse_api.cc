
#include "paddle/phi/api/include/sparse_api.h"
#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/sparse_api_custom_impl.h"
#include "paddle/phi/core/kernel_registry.h"

namespace paddle {
namespace experimental {
namespace sparse {


PADDLE_API Tensor abs(const Tensor& x) {

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

    VLOG(6) << "abs api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "abs_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "abs api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "abs api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "abs_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "abs api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (abs) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor acos(const Tensor& x) {

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

    VLOG(6) << "acos api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acos_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "acos api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "acos api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acos_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "acos api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (acos) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor acosh(const Tensor& x) {

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

    VLOG(6) << "acosh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acosh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "acosh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "acosh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acosh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "acosh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (acosh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor add(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

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

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "add api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "add api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "add api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "add api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (add) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor asin(const Tensor& x) {

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

    VLOG(6) << "asin api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asin_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "asin api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "asin api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asin_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "asin api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (asin) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor asinh(const Tensor& x) {

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

    VLOG(6) << "asinh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asinh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "asinh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "asinh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asinh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "asinh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (asinh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor atan(const Tensor& x) {

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

    VLOG(6) << "atan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atan_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "atan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "atan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atan_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "atan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (atan) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor atanh(const Tensor& x) {

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

    VLOG(6) << "atanh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atanh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "atanh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "atanh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atanh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "atanh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (atanh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor cast(const Tensor& x, DataType index_dtype, DataType value_dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

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

  if (x.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "cast api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "cast_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "cast api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(index_dtype);
    kernel_context.EmplaceBackAttr(value_dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "cast api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "cast_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "cast api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(index_dtype);
    kernel_context.EmplaceBackAttr(value_dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (cast) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor conv3d(const Tensor& x, const Tensor& kernel, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key) {

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
    return std::get<0>(api_output);
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (conv3d) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor coo_to_dense(const Tensor& x) {
  return to_dense_impl(x);
}
PADDLE_API Tensor create_sparse_coo_tensor(const Tensor& values, const Tensor& indices, const IntArray& dense_shape) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(values);

  kernel_data_type = ParseDataType(values);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(values, indices);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (phi::DenseTensor::classof(values.impl().get()) && phi::DenseTensor::classof(indices.impl().get())) {

    VLOG(6) << "create_sparse_coo_tensor api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sparse_coo_tensor", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "create_sparse_coo_tensor api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(values.impl().get());
    kernel_context.EmplaceBackInput(indices.impl().get());
    kernel_context.EmplaceBackAttr(phi::IntArray(dense_shape));
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (create_sparse_coo_tensor) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor dense_to_coo(const Tensor& x, int64_t sparse_dim) {
  return to_sparse_coo_impl(x, sparse_dim);
}
PADDLE_API Tensor divide(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

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

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "divide api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "divide api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "divide api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "divide api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (divide) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor divide_scalar(const Tensor& x, float scalar) {

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

  if (x.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "divide_scalar api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_coo_scalar", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "divide_scalar api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(scalar);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "divide_scalar api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_csr_scalar", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "divide_scalar api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(scalar);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (divide_scalar) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor expm1(const Tensor& x) {

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

    VLOG(6) << "expm1 api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "expm1_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "expm1 api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "expm1 api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "expm1_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "expm1 api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (expm1) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor leaky_relu(const Tensor& x, float alpha) {

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

    VLOG(6) << "leaky_relu api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "leaky_relu_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "leaky_relu api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "leaky_relu api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "leaky_relu_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "leaky_relu api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (leaky_relu) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor log1p(const Tensor& x) {

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

    VLOG(6) << "log1p api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "log1p_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "log1p api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "log1p api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "log1p_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "log1p api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (log1p) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor multiply(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

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

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "multiply api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "multiply_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "multiply api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "multiply api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "multiply_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "multiply api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (multiply) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor pow(const Tensor& x, float factor) {

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

    VLOG(6) << "pow api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "pow_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "pow api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(factor);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "pow api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "pow_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "pow api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(factor);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (pow) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor relu(const Tensor& x) {

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

    VLOG(6) << "relu api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "relu api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "relu api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "relu api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (relu) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor relu6(const Tensor& x, float threshold) {

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

    VLOG(6) << "relu6 api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu6_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "relu6 api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(threshold);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "relu6 api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu6_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "relu6 api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(threshold);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (relu6) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor scale(const Tensor& x, float scale, float bias, bool bias_after_scale) {

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

  if (x.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "scale api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "scale api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(scale);
    kernel_context.EmplaceBackAttr(bias);
    kernel_context.EmplaceBackAttr(bias_after_scale);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "scale api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "scale_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "scale api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(scale);
    kernel_context.EmplaceBackAttr(bias);
    kernel_context.EmplaceBackAttr(bias_after_scale);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (scale) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor sin(const Tensor& x) {

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

    VLOG(6) << "sin api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sin_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sin api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "sin api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sin_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sin api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sin) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor sinh(const Tensor& x) {

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

    VLOG(6) << "sinh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sinh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sinh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "sinh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sinh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sinh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sinh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor softmax(const Tensor& x, int axis) {

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

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "softmax api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "softmax_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "softmax api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(axis);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (softmax) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor sqrt(const Tensor& x) {

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

    VLOG(6) << "sqrt api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sqrt_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sqrt api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "sqrt api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sqrt_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sqrt api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sqrt) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor square(const Tensor& x) {

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

    VLOG(6) << "square api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "square_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "square api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "square api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "square_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "square api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (square) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor subtract(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

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

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "subtract api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "subtract_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "subtract api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "subtract api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "subtract_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "subtract api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (subtract) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor tan(const Tensor& x) {

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

    VLOG(6) << "tan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tan_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "tan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "tan api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tan_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "tan api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (tan) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor tanh(const Tensor& x) {

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

    VLOG(6) << "tanh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tanh_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "tanh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "tanh api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tanh_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "tanh api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (tanh) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor to_dense(const Tensor& x) {
  return to_dense_impl(x);
}
PADDLE_API Tensor to_sparse_coo(const Tensor& x, int64_t sparse_dim) {
  return to_sparse_coo_impl(x, sparse_dim);
}
PADDLE_API Tensor to_sparse_csr(const Tensor& x) {
  return to_sparse_csr_impl(x);
}
PADDLE_API Tensor values(const Tensor& x) {

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

    VLOG(6) << "values api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "coo_values", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "values api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "values api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "csr_values", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "values api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (values) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor addmm(const Tensor& input, const Tensor& x, const Tensor& y, float alpha, float beta) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

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

  if (phi::DenseTensor::classof(input.impl().get()) && x.layout() == DataLayout::SPARSE_CSR && phi::DenseTensor::classof(y.impl().get())) {

    VLOG(6) << "addmm api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_csr_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "addmm api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(input.impl().get());
    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (input.layout() == DataLayout::SPARSE_CSR && x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "addmm api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "addmm api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(input.impl().get());
    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (phi::DenseTensor::classof(input.impl().get()) && x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(y.impl().get())) {

    VLOG(6) << "addmm api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_coo_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "addmm api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(input.impl().get());
    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (input.layout() == DataLayout::SPARSE_COO && x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "addmm api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "addmm api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(input.impl().get());
    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (addmm) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor coalesce(const Tensor& x) {

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

    VLOG(6) << "coalesce api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "coalesce", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "coalesce api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (coalesce) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor full_like(const Tensor& x, const Scalar& value, DataType dtype) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  kernel_data_type = ParseDataType(dtype);

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

    VLOG(6) << "full_like api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "coo_full_like", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "full_like api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(phi::Scalar(value));
    kernel_context.EmplaceBackAttr(dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "full_like api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "csr_full_like", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "full_like api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackAttr(phi::Scalar(value));
    kernel_context.EmplaceBackAttr(dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (full_like) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor fused_attention(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& sparse_mask, const paddle::optional<Tensor>& key_padding_mask, const paddle::optional<Tensor>& attn_mask) {

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
    return std::get<0>(api_output);
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (fused_attention) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor masked_matmul(const Tensor& x, const Tensor& y, const Tensor& mask) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, mask);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (phi::DenseTensor::classof(x.impl().get()) && phi::DenseTensor::classof(y.impl().get()) && mask.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "masked_matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "masked_matmul_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "masked_matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(mask.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (masked_matmul) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor matmul(const Tensor& x, const Tensor& y) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

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

  if (x.layout() == DataLayout::SPARSE_CSR && phi::DenseTensor::classof(y.impl().get())) {

    VLOG(6) << "matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_csr_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_csr_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(y.impl().get())) {

    VLOG(6) << "matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_coo_dense", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "matmul api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_coo_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "matmul api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (matmul) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor maxpool(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides) {

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
    return std::get<0>(api_output);
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (maxpool) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API Tensor mv(const Tensor& x, const Tensor& vec) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(x);

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

  if (x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(vec.impl().get())) {

    VLOG(6) << "mv api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "mv_coo", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "mv api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(vec.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && phi::DenseTensor::classof(vec.impl().get())) {

    VLOG(6) << "mv api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "mv_csr", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "mv api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    Tensor api_output;
    auto* kernel_out = SetSparseKernelOutput(&api_output, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(vec.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return api_output;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (mv) for input tensors is unimplemented, please check the type of input tensors."));
}


}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
