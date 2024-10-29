#pragma once
#if defined(PADDLE_WITH_XPU_KP)

#include "xpu/runtime.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

struct UsedSlotGpuType {
  int is_uint64_value;
  int slot_value_idx;
};

class DataFeedPdboxXpuKernelHelper {
  public:
    static void CopyRankOffset(const paddle::platform::Place& place, int *dest, const int ins_num, const int pv_num, const int max_rank, 
                   const int *ranks, const int *cmatchs, const int *ad_offsets, const int cols);
    
    static void FillSlotValueOffset(const paddle::platform::Place& place, const int ins_num, const int used_slot_num, unsigned long long* slot_value_offsets,
                             const int* uint64_offsets, const int uint64_slot_size, const int* float_offsets,
                             const int float_slot_size, const UsedSlotGpuType* used_slots); 

    static void CopyForTensor(const paddle::platform::Place& place, const int ins_num, const int used_slot_num, unsigned long long* dest, 
                       const unsigned long long* slot_value_offsets, const unsigned long long* uint64_feas,
                       const int* uint64_offsets, const int* uint64_ins_lens, const int uint64_slot_size,
                       const float* float_feas, const int* float_offsets, const int* float_ins_lens,
                       const int float_slot_size, const UsedSlotGpuType* used_slots);
};

}  // namespace framework
}  // namespace paddle
#endif