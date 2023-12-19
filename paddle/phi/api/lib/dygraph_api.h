#pragma once

#include <tuple>

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/utils/optional.h"

namespace paddle {
namespace experimental {


PADDLE_API std::tuple<Tensor, Tensor> flatten_intermediate(const Tensor& x, int start_axis, int stop_axis);

PADDLE_API std::tuple<Tensor&, Tensor> flatten_intermediate_(Tensor& x, int start_axis, int stop_axis);

PADDLE_API std::tuple<Tensor, Tensor> graph_send_recv_intermediate(const Tensor& x, const Tensor& src_index, const Tensor& dst_index, const std::string& pool_type = "SUM", int64_t out_size = 0);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> group_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon, int groups, const std::string& data_layout);

PADDLE_API std::tuple<Tensor, Tensor, Tensor> instance_norm_intermediate(const Tensor& x, const paddle::optional<Tensor>& scale, const paddle::optional<Tensor>& bias, float epsilon);

PADDLE_API std::tuple<Tensor, Tensor> norm_intermediate(const Tensor& x, int axis, float epsilon, bool is_test);

PADDLE_API std::tuple<Tensor, Tensor> reshape_intermediate(const Tensor& x, const IntArray& shape);

PADDLE_API std::tuple<Tensor&, Tensor> reshape_intermediate_(Tensor& x, const IntArray& shape);

PADDLE_API std::tuple<Tensor, Tensor> roi_pool_intermediate(const Tensor& x, const Tensor& boxes, const paddle::optional<Tensor>& boxes_num, int pooled_height, int pooled_width, float spatial_scale);

PADDLE_API std::tuple<Tensor, Tensor> squeeze_intermediate(const Tensor& x, const std::vector<int>& axes);

PADDLE_API std::tuple<Tensor, Tensor> unsqueeze_intermediate(const Tensor& x, const IntArray& axis);

PADDLE_API std::tuple<Tensor, Tensor> warpctc_intermediate(const Tensor& logits, const Tensor& label, const paddle::optional<Tensor>& logits_length, const paddle::optional<Tensor>& labels_length, int blank, bool norm_by_times);

namespace sparse {

// out, rulebook, counter

PADDLE_API std::tuple<Tensor, Tensor, Tensor> conv3d_intermediate(const Tensor& x, const Tensor& kernel, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides, int groups, bool subm, const std::string& key);


// out, softmax

PADDLE_API std::tuple<Tensor, Tensor> fused_attention_intermediate(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& sparse_mask, const paddle::optional<Tensor>& key_padding_mask, const paddle::optional<Tensor>& attn_mask);


// out, rulebook, counter

PADDLE_API std::tuple<Tensor, Tensor, Tensor> maxpool_intermediate(const Tensor& x, const std::vector<int>& kernel_sizes, const std::vector<int>& paddings, const std::vector<int>& dilations, const std::vector<int>& strides);


}  // namespace sparse


}  // namespace experimental
}  // namespace paddle
