
#include "paddle/fluid/framework/attribute.h"
namespace paddle {{
    const static std::unordered_map<std::string, paddle::framework::AttributeMap> extra_attrs_map = {{{{"conv2d", {{ {{"use_cudnn", bool{{false}}}}, {{"fuse_relu_before_depthwise_conv", bool{{false}}}}, {{"use_mkldnn", bool{{false}}}}, {{"use_quantizer", bool{{false}}}}, {{"mkldnn_data_type", std::string{{"float32"}}}}, {{"fuse_relu", bool{{false}}}}, {{"fuse_activation", std::string{{""}}}}, {{"fuse_alpha", bool{{false}}}}, {{"fuse_beta", bool{{false}}}}, {{"use_addto", bool{{false}}}}, {{"fuse_residual_connection", bool{{false}}}}, {{"Scale_in", float{{1.0f}}}}, {{"Scale_out", float{{1.0f}}}}, {{"Scale_in_eltwise", float{{1.0f}}}}, {"Scale_weights", std::vector<float>{1.0f}}, {{"force_fp32_output", bool{{false}}}}, {{"workspace_size_MB", int{{512}}}}, {{"exhaustive_search", bool{{false}}}} }}}}}};
}}  // namespace paddle
