#pragma once

#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"

namespace paddle {
namespace framework {
template <typename T>
void cal_ad_slot_session(const T* input, 
                         const T* ad_input,
                         T* din_output,
                         T* ad_slot_session_output,
                         const uint32_t batch_num, 
                         const uint32_t ins_num, 
                         const uint32_t slot_num,
                         const uint32_t max_length,
                         const uint32_t fea_emb_dim,
                         const uint32_t ad_slot_num,
                         const uint32_t ad_slot_offset,
                         xpu::Context* ctx);

template <typename T>
void cal_sideinfo(const T* input,
                  T* side_info_output,
                  const uint32_t batch_num,
                  const uint32_t ins_num, 
                  const uint32_t slot_num,
                  const uint32_t max_length,
                  const uint32_t fea_emb_dim,
                  const uint32_t sideinfo_slot_num,
                  const uint32_t sideinfo_slot_offset,
                  xpu::Context* ctx);

template <typename T>
void cal_ad_mask(const T* input,
                 T* mask_output,
                 const uint32_t batch_count,
                 const uint32_t ins_num,
                 const uint32_t slot_num,
                 const uint32_t max_length,
                 const uint32_t fea_emb_dim,
                 xpu::Context* ctx);

}
}