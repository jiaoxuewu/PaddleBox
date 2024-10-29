#pragma once

#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"


namespace paddle {
namespace framework {

template<typename T>
int fused_concat(xpu::Context* ctx,
                          const std::vector<const T*>& x_list,
                          T* y,
                          int batch_size,
                          int dim_size,
                          int length,
                          int offset);

template<typename T>
int fused_concat_grad(xpu::Context* ctx,
                               const T* dy,
                               std::vector<T*>& dx_vec,
                               int batch_size,
                               int dim_size,
                               int length,
                               int offset);

}  // namespace framework
}  // namespace paddle
