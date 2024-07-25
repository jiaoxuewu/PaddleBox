#pragma once

#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"


namespace paddle {
namespace framework {

template <typename T>
int add_bias(xpu::Context* xpu_ctx,
              T* data,
              int slot_pairs_num,
              int ins_num,
              int out_dim,
              const T* bias);

template <typename T>
int add_bias_grad(xpu::Context* xpu_ctx,
                   const T* dout_data,
                   int slot_pairs_num,
                   int ins_num,
                   int out_dim,
                   T* db_data);

}  // namespace framework
}  // namespace paddle
