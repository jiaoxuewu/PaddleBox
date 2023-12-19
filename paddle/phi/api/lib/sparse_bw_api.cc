
#include "paddle/phi/api/backward/sparse_bw_api.h"
#include <memory>

#include "glog/logging.h"

#include "paddle/phi/api/include/sparse_api.h"
#include "paddle/phi/api/lib/api_gen_utils.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/api/lib/sparse_api_custom_impl.h"
#include "paddle/phi/core/kernel_registry.h"

namespace paddle {
namespace experimental {
namespace sparse {


PADDLE_API void abs_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "abs_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "abs_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "abs_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "abs_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "abs_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "abs_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (abs_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void acos_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "acos_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acos_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "acos_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "acos_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acos_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "acos_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (acos_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void acosh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "acosh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acosh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "acosh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "acosh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "acosh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "acosh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (acosh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void add_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "add_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "add_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "add_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "add_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "add_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (add_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void addmm_grad(const Tensor& input, const Tensor& x, const Tensor& y, const Tensor& out_grad, float alpha, float beta, Tensor* input_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(input, x, y, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (phi::DenseTensor::classof(input.impl().get()) && x.layout() == DataLayout::SPARSE_CSR && phi::DenseTensor::classof(y.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "addmm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_csr_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "addmm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(input_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_1 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_2 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(input.impl().get());
    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  if (input.layout() == DataLayout::SPARSE_CSR && x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "addmm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "addmm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(input_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_2 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(input.impl().get());
    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  if (phi::DenseTensor::classof(input.impl().get()) && x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(y.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "addmm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_coo_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "addmm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(input_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_1 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_2 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(input.impl().get());
    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  if (input.layout() == DataLayout::SPARSE_COO && x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "addmm_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "addmm_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "addmm_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(input_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_2 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(input.impl().get());
    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackAttr(beta);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (addmm_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void asin_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "asin_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asin_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "asin_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "asin_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asin_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "asin_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (asin_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void asinh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "asinh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asinh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "asinh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "asinh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "asinh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "asinh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (asinh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void atan_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "atan_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atan_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "atan_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "atan_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atan_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "atan_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (atan_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void atanh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "atanh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atanh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "atanh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "atanh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "atanh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "atanh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (atanh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void cast_grad(const Tensor& x, const Tensor& out_grad, DataType value_dtype, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_data_type = ParseDataType(out_grad);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "cast_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "cast_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "cast_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(value_dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "cast_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "cast_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "cast_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(value_dtype);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (cast_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void conv3d_coo_grad(const Tensor& x, const Tensor& kernel, const Tensor& out, const Tensor& rulebook, const Tensor& counter, const Tensor& out_grad, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key, Tensor* x_grad, Tensor* kernel_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, kernel, out, rulebook, counter, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(kernel.impl().get()) && out.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(rulebook.impl().get()) && phi::DenseTensor::classof(counter.impl().get()) && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "conv3d_coo_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "conv3d_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "conv3d_coo_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(kernel_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(kernel.impl().get());
    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(rulebook.impl().get());
    kernel_context.EmplaceBackInput(counter.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(paddings);
    kernel_context.EmplaceBackAttr(dilations);
    kernel_context.EmplaceBackAttr(strides);
    kernel_context.EmplaceBackAttr(groups);
    kernel_context.EmplaceBackAttr(subm);
    kernel_context.EmplaceBackAttr(key);
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (conv3d_coo_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void coo_to_dense_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "coo_to_dense_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sparse_coo_to_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "coo_to_dense_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (coo_to_dense_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void create_sparse_coo_tensor_grad(const Tensor& indices, const Tensor& out_grad, Tensor* values_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(indices, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (phi::DenseTensor::classof(indices.impl().get()) && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "create_sparse_coo_tensor_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sparse_coo_tensor_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "create_sparse_coo_tensor_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(values_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(indices.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (create_sparse_coo_tensor_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void dense_to_coo_grad(const Tensor& out_grad, Tensor* x_grad) {
  *x_grad = to_dense_impl(out_grad);
}
PADDLE_API void divide_grad(const Tensor& x, const Tensor& y, const Tensor& out, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO && out.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "divide_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "divide_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR && out.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "divide_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "divide_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "divide_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (divide_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void divide_scalar_grad(const Tensor& out_grad, float scalar, Tensor* x_grad) {
  *x_grad = divide_scalar(out_grad, scalar);
}
PADDLE_API void expm1_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (out.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "expm1_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "expm1_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "expm1_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "expm1_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "expm1_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "expm1_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (expm1_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void leaky_relu_grad(const Tensor& x, const Tensor& out_grad, float alpha, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "leaky_relu_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "leaky_relu_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "leaky_relu_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "leaky_relu_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "leaky_relu_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "leaky_relu_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(alpha);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (leaky_relu_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void log1p_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "log1p_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "log1p_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "log1p_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "log1p_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "log1p_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "log1p_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (log1p_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void masked_matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (phi::DenseTensor::classof(x.impl().get()) && phi::DenseTensor::classof(y.impl().get()) && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "masked_matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "masked_matmul_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "masked_matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (masked_matmul_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void matmul_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_CSR && phi::DenseTensor::classof(y.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_csr_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(y.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_coo_dense_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "matmul_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "matmul_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "matmul_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (matmul_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void maxpool_grad(const Tensor& x, const Tensor& rulebook, const Tensor& counter, const Tensor& out, const Tensor& out_grad, const std::vector<int>& kernel_sizes, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, rulebook, counter, out, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(rulebook.impl().get()) && phi::DenseTensor::classof(counter.impl().get()) && out.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "maxpool_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "maxpool_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "maxpool_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(rulebook.impl().get());
    kernel_context.EmplaceBackInput(counter.impl().get());
    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(kernel_sizes);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (maxpool_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void multiply_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "multiply_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "multiply_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "multiply_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "multiply_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "multiply_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "multiply_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (multiply_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void mv_grad(const Tensor& x, const Tensor& vec, const Tensor& out_grad, Tensor* x_grad, Tensor* vec_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, vec, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(vec.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "mv_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "mv_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "mv_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(vec_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(vec.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && phi::DenseTensor::classof(vec.impl().get()) && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "mv_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "mv_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "mv_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(vec_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(vec.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (mv_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void pow_grad(const Tensor& x, const Tensor& out_grad, float factor, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "pow_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "pow_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "pow_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(factor);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "pow_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "pow_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "pow_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(factor);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (pow_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void relu6_grad(const Tensor& out, const Tensor& out_grad, float threshold, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (out.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "relu6_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu6_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "relu6_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(threshold);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "relu6_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu6_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "relu6_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(threshold);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (relu6_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void relu_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (out.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "relu_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "relu_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "relu_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "relu_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "relu_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (relu_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void scale_grad(const Tensor& out_grad, float scale_val, Tensor* x_grad) {
  *x_grad = scale(out_grad, scale_val, 0.0, true);
}
PADDLE_API void sin_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "sin_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sin_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sin_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "sin_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sin_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sin_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sin_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void sinh_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "sinh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sinh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sinh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "sinh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sinh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sinh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sinh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void softmax_grad(const Tensor& out, const Tensor& out_grad, int axis, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (out.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "softmax_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "softmax_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "softmax_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackAttr(axis);
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (softmax_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void sqrt_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (out.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "sqrt_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sqrt_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sqrt_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "sqrt_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "sqrt_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "sqrt_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (sqrt_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void square_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "square_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "square_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "square_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "square_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "square_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "square_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (square_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void subtract_grad(const Tensor& x, const Tensor& y, const Tensor& out_grad, Tensor* x_grad, Tensor* y_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, y, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && y.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "subtract_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "subtract_coo_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "subtract_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && y.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "subtract_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "subtract_csr_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "subtract_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);
    auto kernel_out_1 = SetSparseKernelOutput(y_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(y.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (subtract_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void tan_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "tan_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tan_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "tan_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (x.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "tan_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tan_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "tan_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (tan_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void tanh_grad(const Tensor& out, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(out, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (out.layout() == DataLayout::SPARSE_COO && out_grad.layout() == DataLayout::SPARSE_COO) {

    VLOG(6) << "tanh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tanh_coo_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "tanh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  if (out.layout() == DataLayout::SPARSE_CSR && out_grad.layout() == DataLayout::SPARSE_CSR) {

    VLOG(6) << "tanh_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "tanh_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "tanh_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_CSR);

    kernel_context.EmplaceBackInput(out.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (tanh_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void values_grad(const Tensor& x, const Tensor& out_grad, Tensor* x_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(x, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (x.layout() == DataLayout::SPARSE_COO && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "values_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "coo_values_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "values_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out = SetSparseKernelOutput(x_grad, TensorType::SPARSE_COO);

    kernel_context.EmplaceBackInput(x.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (values_grad) for input tensors is unimplemented, please check the type of input tensors."));
}

PADDLE_API void fused_attention_grad(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& softmax, const Tensor& out_grad, Tensor* query_grad, Tensor* key_grad, Tensor* value_grad) {

  Backend kernel_backend = Backend::UNDEFINED;
  DataLayout kernel_layout = DataLayout::UNDEFINED;
  DataType kernel_data_type = DataType::UNDEFINED;

  kernel_layout = ParseLayout(softmax);

  kernel_data_type = ParseDataType(query);

  if (kernel_backend == Backend::UNDEFINED
        || kernel_layout == DataLayout::UNDEFINED
        || kernel_data_type == DataType::UNDEFINED ) {
    auto kernel_key_set = ParseKernelKeyByInputArgs(query, key, value, softmax, out_grad);
    auto kernel_key = kernel_key_set.GetHighestPriorityKernelKey();
    if (kernel_backend == Backend::UNDEFINED) {
      kernel_backend = kernel_key.backend();
    }
    if (kernel_layout == DataLayout::UNDEFINED) {
      kernel_layout = kernel_key.layout();
    }
    if (kernel_data_type == DataType::UNDEFINED) {
      kernel_data_type = kernel_key.dtype();
    }
  }

  if (phi::DenseTensor::classof(query.impl().get()) && phi::DenseTensor::classof(key.impl().get()) && phi::DenseTensor::classof(value.impl().get()) && softmax.layout() == DataLayout::SPARSE_CSR && phi::DenseTensor::classof(out_grad.impl().get())) {

    VLOG(6) << "fused_attention_grad api sparse kernel key: [" << kernel_backend << ", " << kernel_layout << ", "<< kernel_data_type << "]";
    auto kernel_result = phi::KernelFactory::Instance().SelectKernelOrThrowError(
        "fused_attention_csr_grad", {kernel_backend, kernel_layout, kernel_data_type});
    const auto& phi_kernel = kernel_result.kernel;
    VLOG(6) << "fused_attention_grad api sparse kernel: " << phi_kernel;

    auto* dev_ctx = GetDeviceContextByBackend(kernel_result.has_fallback_cpu ? Backend::CPU : kernel_backend);
    auto kernel_context = phi::KernelContext(dev_ctx);

    auto kernel_out_0 = SetSparseKernelOutput(query_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_1 = SetSparseKernelOutput(key_grad, TensorType::DENSE_TENSOR);
    auto kernel_out_2 = SetSparseKernelOutput(value_grad, TensorType::DENSE_TENSOR);

    kernel_context.EmplaceBackInput(query.impl().get());
    kernel_context.EmplaceBackInput(key.impl().get());
    kernel_context.EmplaceBackInput(value.impl().get());
    kernel_context.EmplaceBackInput(softmax.impl().get());
    kernel_context.EmplaceBackInput(out_grad.impl().get());
    kernel_context.EmplaceBackOutput(kernel_out_0);
    kernel_context.EmplaceBackOutput(kernel_out_1);
    kernel_context.EmplaceBackOutput(kernel_out_2);
    phi_kernel(&kernel_context);
    return;
  }

  PADDLE_THROW(phi::errors::Unimplemented(
          "The kernel of (fused_attention_grad) for input tensors is unimplemented, please check the type of input tensors."));
}


}  // namespace sparse
}  // namespace experimental
}  // namespace paddle
