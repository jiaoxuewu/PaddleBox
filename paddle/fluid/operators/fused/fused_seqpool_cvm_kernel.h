#pragma once

#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace framework {

template<typename T, typename TID = int>
int sequence_sum_pool_cvm(xpu::Context* ctx,
                          const std::vector<const T*>& x,
                          const std::vector<T*>& y,
                          const std::vector<TID>& lods,
                          uint32_t batch,
                          uint32_t dim,
                          uint32_t slot_num,
                          bool use_cvm,
                          bool clk_filter,
                          bool need_filter,
                          float padding_value,
                          int quant_ratio,
                          float show_coeff,
                          float clk_coeff,
                          float threshold,
                          int cvm_offset,
                          bool embed_threshold_filter,
                          float embed_threshold,
                          int embed_thres_size,
                          int embedx_concate_size,
                          bool embedx_concate_filter,
                          bool fix_ctr_to_click);

template<typename T, typename TID = int>
int sequence_sum_pool_cvm_grad(xpu::Context* ctx,
                               std::vector<const float*>& dy_vec,
                               float *cvm,
                               std::vector<float*>& dx_vec,
                               std::vector<int>& seq_vec,
                               bool use_cvm,
                               int cvm_offset,
                               bool clk_filter,//split
                               uint32_t item_width,
                               uint32_t batch_size,
                               uint32_t slot_num,
                               int embed_thres_size,
                               int embedx_concate_size);

template<typename T, typename TID = int>
int sequence_sum_pool_cvm_with_diff_thres(xpu::Context* ctx,
                          const std::vector<const T*>& x,
                          const std::vector<T*>& y,
                          const std::vector<TID>& lods,
                          uint32_t batch,
                          uint32_t dim,
                          uint32_t slot_num,
                          bool use_cvm,
                          bool clk_filter,
                          bool need_filter,
                          float padding_value,
                          int quant_ratio,
                          float show_coeff,
                          float clk_coeff,
                          float threshold,
                          int cvm_offset,
                          bool xbox_diff_thres_filter,
                          const std::vector<T>& diff_threshold_vec);

template<typename T, typename TID = int>
int sequence_sum_pool_cvm_with_diff_thres_grad(xpu::Context* ctx,
                               std::vector<const float*>& dy_vec,
                               float *cvm,
                               std::vector<float*>& dx_vec,
                               std::vector<int>& seq_vec,
                               bool use_cvm,
                               int cvm_offset,
                               bool clk_filter,//split
                               uint32_t item_width,
                               uint32_t batch_size,
                               uint32_t slot_num);

template<typename T, typename TID = int>
int sequence_sum_pool_cvm_with_conv(xpu::Context* ctx,
                          const std::vector<const T*>& x,
                          const std::vector<T*>& y,
                          const std::vector<TID>& lods,
                          uint32_t batch,
                          uint32_t dim,
                          uint32_t slot_num,
                          bool use_cvm,
                          bool need_filter,
                          float show_coeff,
                          float clk_coeff,
                          float threshold,
                          bool show_filter,
                          float padding_value,
                          int cvm_offset,
                          int embedx_concate_size);
template<typename T, typename TID = int>
int sequence_sum_pool_cvm_with_conv_grad(xpu::Context* ctx,
                               std::vector<const float*>& dy_vec,
                               float *cvm,
                               std::vector<float*>& dx_vec,
                               std::vector<int>& seq_vec,
                               bool use_cvm,
                               int cvm_offset,
                               bool show_filter,
                               uint32_t item_width,
                               uint32_t batch_size,
                               uint32_t slot_num,
                               int embedx_concate_size);
}  // end namespace framework
}  // end namespace paddle
