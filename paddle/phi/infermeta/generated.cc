
#include "paddle/phi/infermeta/generated.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/infermeta/nullary.h"
#include "paddle/phi/infermeta/unary.h"
#include "paddle/phi/infermeta/ternary.h"

namespace phi {

void AllcloseInferMeta(const MetaTensor& x, const MetaTensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan, MetaTensor* out) {
  AllValueCompareInferMeta(x, y, out);
}

void Assign_valueInferMeta(const std::vector<int>& shape, DataType dtype, const std::vector<phi::Scalar>& values, MetaTensor* out) {
  AssignValueInferMeta(shape, dtype, out);
}

void BreluInferMeta(const MetaTensor& x, float t_min, float t_max, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void CeluInferMeta(const MetaTensor& x, float alpha, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void ClipInferMeta(const MetaTensor& x, const Scalar& min, const Scalar& max, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void CumprodInferMeta(const MetaTensor& x, int dim, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void Depthwise_conv2dInferMeta(const MetaTensor& x, const MetaTensor& filter, const std::vector<int>& strides, const std::vector<int>& paddings, const std::string& padding_algorithm, int groups, const std::vector<int>& dilations, const std::string& data_format, bool use_addto, int workspace_size_MB, bool exhaustive_search, bool fuse_relu, MetaTensor* out) {
  ConvInferMeta(x, filter, strides, paddings, padding_algorithm, groups, dilations, data_format, use_addto, workspace_size_MB, exhaustive_search, out);
}

void EluInferMeta(const MetaTensor& x, float alpha, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void ExponentialInferMeta(const MetaTensor& x, float lambda, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void FmaxInferMeta(const MetaTensor& x, const MetaTensor& y, int axis, MetaTensor* out) {
  ElementwiseInferMeta(x, y, out);
}

void FminInferMeta(const MetaTensor& x, const MetaTensor& y, int axis, MetaTensor* out) {
  ElementwiseInferMeta(x, y, out);
}

void FullInferMeta(const IntArray& shape, const Scalar& value, DataType dtype, MetaTensor* out) {
  CreateInferMeta(shape, dtype, out);
}

void Full_likeInferMeta(const MetaTensor& x, const Scalar& value, DataType dtype, MetaTensor* out) {
  CreateLikeInferMeta(x, dtype, out);
}

void GeluInferMeta(const MetaTensor& x, bool approximate, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void Grid_sampleInferMeta(const MetaTensor& x, const MetaTensor& grid, const std::string& mode, const std::string& padding_mode, bool align_corners, MetaTensor* out) {
  GridSampleBaseInferMeta(x, grid, out);
}

void Hard_shrinkInferMeta(const MetaTensor& x, float threshold, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void Hard_sigmoidInferMeta(const MetaTensor& x, float slope, float offset, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void Hard_swishInferMeta(const MetaTensor& x, float threshold, float scale, float offset, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void IscloseInferMeta(const MetaTensor& x, const MetaTensor& y, const Scalar& rtol, const Scalar& atol, bool equal_nan, MetaTensor* out) {
  ValueCompareInferMeta(x, y, out);
}

void Label_smoothInferMeta(const MetaTensor& label, const MetaTensor& prior_dist, float epsilon, MetaTensor* out) {
  UnchangedInferMeta(label, out);
}

void Leaky_reluInferMeta(const MetaTensor& x, float alpha, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void LogitInferMeta(const MetaTensor& x, float eps, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void Matrix_powerInferMeta(const MetaTensor& x, int n, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void Matrix_rankInferMeta(const MetaTensor& x, float tol, bool use_default_tol, bool hermitian, MetaTensor* out) {
  MatrixRankInferMeta(x, use_default_tol, hermitian, out);
}

void MishInferMeta(const MetaTensor& x, float lambda, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void PowInferMeta(const MetaTensor& x, const Scalar& s, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void Put_along_axisInferMeta(const MetaTensor& x, const MetaTensor& index, const MetaTensor& value, int axis, const std::string& reduce, MetaTensor* out) {
  UnchangedInferMeta(index, out);
}

void Relu6InferMeta(const MetaTensor& x, float threshold, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void ScaleInferMeta(const MetaTensor& x, const Scalar& scale, float bias, bool bias_after_scale, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void SeluInferMeta(const MetaTensor& x, float scale, float alpha, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void Soft_shrinkInferMeta(const MetaTensor& x, float lambda, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void SoftplusInferMeta(const MetaTensor& x, float beta, float threshold, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void SwishInferMeta(const MetaTensor& x, float beta, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

void Take_along_axisInferMeta(const MetaTensor& x, const MetaTensor& index, int axis, MetaTensor* out) {
  UnchangedInferMeta(index, out);
}

void Thresholded_reluInferMeta(const MetaTensor& x, float threshold, MetaTensor* out) {
  UnchangedInferMeta(x, out);
}

}  // namespace phi

PD_REGISTER_INFER_META_FN(atan2, phi::Atan2InferMeta);
PD_REGISTER_INFER_META_FN(bernoulli, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(cholesky, phi::CholeskyInferMeta);
PD_REGISTER_INFER_META_FN(cholesky_solve, phi::CholeskySolveInferMeta);
PD_REGISTER_INFER_META_FN(cross, phi::CrossInferMeta);
PD_REGISTER_INFER_META_FN(diag, phi::DiagInferMeta);
PD_REGISTER_INFER_META_FN(diagonal, phi::DiagonalInferMeta);
PD_REGISTER_INFER_META_FN(digamma, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(dist, phi::DistInferMeta);
PD_REGISTER_INFER_META_FN(dot, phi::DotInferMeta);
PD_REGISTER_INFER_META_FN(erf, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(lgamma, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(mv, phi::MvInferMeta);
PD_REGISTER_INFER_META_FN(poisson, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(solve, phi::SolveInferMeta);
PD_REGISTER_INFER_META_FN(trace, phi::TraceInferMeta);
PD_REGISTER_INFER_META_FN(trunc, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(abs, phi::RealAndImagInferMeta);
PD_REGISTER_INFER_META_FN(accuracy, phi::AccuracyInferMeta);
PD_REGISTER_INFER_META_FN(acos, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(acosh, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(adadelta, phi::AdadeltaInferMeta);
PD_REGISTER_INFER_META_FN(adagrad, phi::AdagradInferMeta);
PD_REGISTER_INFER_META_FN(adam, phi::AdamInferMeta);
PD_REGISTER_INFER_META_FN(adamax, phi::AdamaxInferMeta);
PD_REGISTER_INFER_META_FN(add, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(add_n, phi::AddNInferMeta);
PD_REGISTER_INFER_META_FN(addmm, phi::AddmmInferMeta);
PD_REGISTER_INFER_META_FN(all, phi::ReduceInferMeta);
PD_REGISTER_INFER_META_FN(allclose, phi::AllcloseInferMeta);
PD_REGISTER_INFER_META_FN(amax, phi::ReduceInferMeta);
PD_REGISTER_INFER_META_FN(amin, phi::ReduceInferMeta);
PD_REGISTER_INFER_META_FN(angle, phi::RealAndImagInferMeta);
PD_REGISTER_INFER_META_FN(any, phi::ReduceInferMeta);
PD_REGISTER_INFER_META_FN(arange, phi::ArangeInferMeta);
PD_REGISTER_INFER_META_FN(arg_max, phi::ArgMinMaxInferMeta);
PD_REGISTER_INFER_META_FN(arg_min, phi::ArgMinMaxInferMeta);
PD_REGISTER_INFER_META_FN(argsort, phi::ArgsortInferMeta);
PD_REGISTER_INFER_META_FN(as_complex, phi::AsComplexInferMeta);
PD_REGISTER_INFER_META_FN(as_real, phi::AsRealInferMeta);
PD_REGISTER_INFER_META_FN(asin, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(asinh, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(assign, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(assign_value, phi::Assign_valueInferMeta);
PD_REGISTER_INFER_META_FN(atan, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(atanh, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(auc, phi::AucInferMeta);
PD_REGISTER_INFER_META_FN(average_accumulates, phi::AverageAccumulatesInferMeta);
PD_REGISTER_INFER_META_FN(bce_loss, phi::BCELossInferMeta);
PD_REGISTER_INFER_META_FN(bilinear_tensor_product, phi::BilinearTensorProductInferMeta);
PD_REGISTER_INFER_META_FN(bitwise_and, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(bitwise_not, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(bitwise_or, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(bitwise_xor, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(bmm, phi::BmmInferMeta);
PD_REGISTER_INFER_META_FN(box_coder, phi::BoxCoderInferMeta);
PD_REGISTER_INFER_META_FN(brelu, phi::BreluInferMeta);
PD_REGISTER_INFER_META_FN(cast, phi::CastInferMeta);
PD_REGISTER_INFER_META_FN(ceil, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(celu, phi::CeluInferMeta);
PD_REGISTER_INFER_META_FN(clip, phi::ClipInferMeta);
PD_REGISTER_INFER_META_FN(clip_by_norm, phi::ClipByNormInferMeta);
PD_REGISTER_INFER_META_FN(complex, phi::ComplexInferMeta);
PD_REGISTER_INFER_META_FN(concat, phi::ConcatInferMeta);
PD_REGISTER_INFER_META_FN(conj, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(conv2d_transpose, phi::ConvTransposeInferMeta);
PD_REGISTER_INFER_META_FN(conv3d_transpose, phi::ConvTransposeInferMeta);
PD_REGISTER_INFER_META_FN(cos, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(cosh, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(crop_tensor, phi::CropTensorInferMeta);
PD_REGISTER_INFER_META_FN(cross_entropy_with_softmax, phi::CrossEntropyWithSoftmaxInferMeta);
PD_REGISTER_INFER_META_FN(cumprod, phi::CumprodInferMeta);
PD_REGISTER_INFER_META_FN(cumsum, phi::CumInferMeta);
PD_REGISTER_INFER_META_FN(deformable_conv, phi::DeformableConvInferMeta);
PD_REGISTER_INFER_META_FN(depthwise_conv2d, phi::Depthwise_conv2dInferMeta);
PD_REGISTER_INFER_META_FN(depthwise_conv2d_transpose, phi::ConvTransposeInferMeta);
PD_REGISTER_INFER_META_FN(determinant, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(diag_embed, phi::DiagEmbedInferMeta);
PD_REGISTER_INFER_META_FN(divide, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(dropout, phi::DropoutInferMeta);
PD_REGISTER_INFER_META_FN(eigh, phi::EighInferMeta);
PD_REGISTER_INFER_META_FN(eigvals, phi::EigvalsInferMeta);
PD_REGISTER_INFER_META_FN(einsum_raw, phi::EinsumRawInferMeta);
PD_REGISTER_INFER_META_FN(elementwise_pow, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(elu, phi::EluInferMeta);
PD_REGISTER_INFER_META_FN(empty, phi::CreateInferMeta);
PD_REGISTER_INFER_META_FN(empty_like, phi::CreateLikeInferMeta);
PD_REGISTER_INFER_META_FN(equal, phi::CompareInferMeta);
PD_REGISTER_INFER_META_FN(equal_all, phi::CompareAllInferMeta);
PD_REGISTER_INFER_META_FN(erfinv, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(exp, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(expand, phi::ExpandInferMeta);
PD_REGISTER_INFER_META_FN(expand_as, phi::ExpandAsInferMeta);
PD_REGISTER_INFER_META_FN(expm1, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(exponential, phi::ExponentialInferMeta);
PD_REGISTER_INFER_META_FN(eye, phi::EyeInferMeta);
PD_REGISTER_INFER_META_FN(flip, phi::FlipInferMeta);
PD_REGISTER_INFER_META_FN(floor, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(floor_divide, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(fmax, phi::FmaxInferMeta);
PD_REGISTER_INFER_META_FN(fmin, phi::FminInferMeta);
PD_REGISTER_INFER_META_FN(frame, phi::FrameInferMeta);
PD_REGISTER_INFER_META_FN(frobenius_norm, phi::ReduceInferMetaBase);
PD_REGISTER_INFER_META_FN(full, phi::FullInferMeta);
PD_REGISTER_INFER_META_FN(full_batch_size_like, phi::FullBatchSizeLikeInferMeta);
PD_REGISTER_INFER_META_FN(full_like, phi::Full_likeInferMeta);
PD_REGISTER_INFER_META_FN(gather, phi::GatherInferMeta);
PD_REGISTER_INFER_META_FN(gather_nd, phi::GatherNdInferMeta);
PD_REGISTER_INFER_META_FN(gather_tree, phi::GatherTreeMeta);
PD_REGISTER_INFER_META_FN(gaussian_random, phi::GaussianRandomInferMeta);
PD_REGISTER_INFER_META_FN(gelu, phi::GeluInferMeta);
PD_REGISTER_INFER_META_FN(greater_equal, phi::CompareInferMeta);
PD_REGISTER_INFER_META_FN(greater_than, phi::CompareInferMeta);
PD_REGISTER_INFER_META_FN(grid_sample, phi::Grid_sampleInferMeta);
PD_REGISTER_INFER_META_FN(gumbel_softmax, phi::GumbelSoftmaxInferMeta);
PD_REGISTER_INFER_META_FN(hard_shrink, phi::Hard_shrinkInferMeta);
PD_REGISTER_INFER_META_FN(hard_sigmoid, phi::Hard_sigmoidInferMeta);
PD_REGISTER_INFER_META_FN(hard_swish, phi::Hard_swishInferMeta);
PD_REGISTER_INFER_META_FN(hierarchical_sigmoid, phi::HierarchicalSigmoidInferMeta);
PD_REGISTER_INFER_META_FN(histogram, phi::HistogramInferMeta);
PD_REGISTER_INFER_META_FN(huber_loss, phi::HuberLossInferMeta);
PD_REGISTER_INFER_META_FN(imag, phi::RealAndImagInferMeta);
PD_REGISTER_INFER_META_FN(increment, phi::IncrementInferMeta);
PD_REGISTER_INFER_META_FN(index_sample, phi::IndexSampleInferMeta);
PD_REGISTER_INFER_META_FN(index_select, phi::IndexSelectInferMeta);
PD_REGISTER_INFER_META_FN(inverse, phi::InverseInferMeta);
PD_REGISTER_INFER_META_FN(is_empty, phi::IsEmptyInferMeta);
PD_REGISTER_INFER_META_FN(isclose, phi::IscloseInferMeta);
PD_REGISTER_INFER_META_FN(isfinite, phi::IsfiniteInferMeta);
PD_REGISTER_INFER_META_FN(isinf, phi::IsfiniteInferMeta);
PD_REGISTER_INFER_META_FN(isnan, phi::IsfiniteInferMeta);
PD_REGISTER_INFER_META_FN(kldiv_loss, phi::KLDivInferMeta);
PD_REGISTER_INFER_META_FN(kron, phi::KronInferMeta);
PD_REGISTER_INFER_META_FN(kthvalue, phi::KthvalueInferMeta);
PD_REGISTER_INFER_META_FN(label_smooth, phi::Label_smoothInferMeta);
PD_REGISTER_INFER_META_FN(layer_norm, phi::LayerNormInferMeta);
PD_REGISTER_INFER_META_FN(leaky_relu, phi::Leaky_reluInferMeta);
PD_REGISTER_INFER_META_FN(lerp, phi::LerpInferMeta);
PD_REGISTER_INFER_META_FN(less_equal, phi::CompareInferMeta);
PD_REGISTER_INFER_META_FN(less_than, phi::CompareInferMeta);
PD_REGISTER_INFER_META_FN(linspace, phi::LinspaceInferMeta);
PD_REGISTER_INFER_META_FN(log, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(log10, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(log1p, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(log2, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(log_loss, phi::LogLossInferMeta);
PD_REGISTER_INFER_META_FN(log_softmax, phi::UnchangedInferMetaCheckAxis);
PD_REGISTER_INFER_META_FN(logcumsumexp, phi::CumInferMeta);
PD_REGISTER_INFER_META_FN(logical_and, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(logical_not, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(logical_or, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(logical_xor, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(logit, phi::LogitInferMeta);
PD_REGISTER_INFER_META_FN(logsigmoid, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(logsumexp, phi::LogsumexpInferMeta);
PD_REGISTER_INFER_META_FN(lstsq, phi::LstsqInferMeta);
PD_REGISTER_INFER_META_FN(lu, phi::LUInferMeta);
PD_REGISTER_INFER_META_FN(lu_unpack, phi::LUUnpackInferMeta);
PD_REGISTER_INFER_META_FN(masked_select, phi::MaskedSelectInferMeta);
PD_REGISTER_INFER_META_FN(matmul, phi::MatmulInferMeta);
PD_REGISTER_INFER_META_FN(matrix_power, phi::Matrix_powerInferMeta);
PD_REGISTER_INFER_META_FN(matrix_rank, phi::Matrix_rankInferMeta);
PD_REGISTER_INFER_META_FN(matrix_rank_tol, phi::MatrixRankTolInferMeta);
PD_REGISTER_INFER_META_FN(max, phi::ReduceInferMeta);
PD_REGISTER_INFER_META_FN(max_pool2d_with_index, phi::MaxPoolWithIndexInferMeta);
PD_REGISTER_INFER_META_FN(max_pool3d_with_index, phi::MaxPoolWithIndexInferMeta);
PD_REGISTER_INFER_META_FN(maximum, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(maxout, phi::MaxOutInferMeta);
PD_REGISTER_INFER_META_FN(mean, phi::ReduceInferMeta);
PD_REGISTER_INFER_META_FN(mean_all, phi::MeanAllInferMeta);
PD_REGISTER_INFER_META_FN(meshgrid, phi::MeshgridInferMeta);
PD_REGISTER_INFER_META_FN(min, phi::ReduceInferMeta);
PD_REGISTER_INFER_META_FN(minimum, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(mish, phi::MishInferMeta);
PD_REGISTER_INFER_META_FN(mode, phi::ModeInferMeta);
PD_REGISTER_INFER_META_FN(modulo, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(multi_dot, phi::MultiDotInferMeta);
PD_REGISTER_INFER_META_FN(multinomial, phi::MultinomialInferMeta);
PD_REGISTER_INFER_META_FN(multiplex, phi::MultiplexInferMeta);
PD_REGISTER_INFER_META_FN(multiply, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(nll_loss, phi::NllLossRawInferMeta);
PD_REGISTER_INFER_META_FN(nms, phi::NMSInferMeta);
PD_REGISTER_INFER_META_FN(not_equal, phi::CompareInferMeta);
PD_REGISTER_INFER_META_FN(one_hot, phi::OneHotInferMeta);
PD_REGISTER_INFER_META_FN(p_norm, phi::PNormInferMeta);
PD_REGISTER_INFER_META_FN(pad, phi::PadInferMeta);
PD_REGISTER_INFER_META_FN(pad3d, phi::Pad3dInferMeta);
PD_REGISTER_INFER_META_FN(pixel_shuffle, phi::PixelShuffleInferMeta);
PD_REGISTER_INFER_META_FN(pool2d, phi::PoolInferMeta);
PD_REGISTER_INFER_META_FN(pool3d, phi::PoolInferMeta);
PD_REGISTER_INFER_META_FN(pow, phi::PowInferMeta);
PD_REGISTER_INFER_META_FN(prelu, phi::PReluInferMeta);
PD_REGISTER_INFER_META_FN(prior_box, phi::PriorBoxInferMeta);
PD_REGISTER_INFER_META_FN(psroi_pool, phi::PsroiPoolInferMeta);
PD_REGISTER_INFER_META_FN(put_along_axis, phi::Put_along_axisInferMeta);
PD_REGISTER_INFER_META_FN(qr, phi::QrInferMeta);
PD_REGISTER_INFER_META_FN(randint, phi::RandintInferMeta);
PD_REGISTER_INFER_META_FN(randperm, phi::RandpermInferMeta);
PD_REGISTER_INFER_META_FN(real, phi::RealAndImagInferMeta);
PD_REGISTER_INFER_META_FN(reciprocal, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(prod_raw, phi::ReduceInferMetaBase);
PD_REGISTER_INFER_META_FN(relu, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(relu6, phi::Relu6InferMeta);
PD_REGISTER_INFER_META_FN(reverse, phi::ReverseInferMeta);
PD_REGISTER_INFER_META_FN(reverse_array, phi::ReverseArrayInferMeta);
PD_REGISTER_INFER_META_FN(rmsprop, phi::RmspropInferMeta);
PD_REGISTER_INFER_META_FN(roi_align, phi::RoiAlignInferMeta);
PD_REGISTER_INFER_META_FN(roll, phi::RollInferMeta);
PD_REGISTER_INFER_META_FN(round, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(rsqrt, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(scale, phi::ScaleInferMeta);
PD_REGISTER_INFER_META_FN(scatter, phi::ScatterInferMeta);
PD_REGISTER_INFER_META_FN(scatter_nd_add, phi::ScatterNdAddInferMeta);
PD_REGISTER_INFER_META_FN(searchsorted, phi::SearchsortedInferMeta);
PD_REGISTER_INFER_META_FN(segment_pool, phi::SegmentPoolInferMeta);
PD_REGISTER_INFER_META_FN(selu, phi::SeluInferMeta);
PD_REGISTER_INFER_META_FN(sgd, phi::SgdInferMeta);
PD_REGISTER_INFER_META_FN(shape, phi::ShapeInferMeta);
PD_REGISTER_INFER_META_FN(shard_index, phi::ShardIndexInferMeta);
PD_REGISTER_INFER_META_FN(sigmoid, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(sigmoid_cross_entropy_with_logits, phi::SigmoidCrossEntropyWithLogitsInferMeta);
PD_REGISTER_INFER_META_FN(sign, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(silu, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(sin, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(sinh, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(size, phi::SizeInferMeta);
PD_REGISTER_INFER_META_FN(slice, phi::SliceRawInferMeta);
PD_REGISTER_INFER_META_FN(slogdeterminant, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(soft_shrink, phi::Soft_shrinkInferMeta);
PD_REGISTER_INFER_META_FN(softmax, phi::SoftmaxInferMeta);
PD_REGISTER_INFER_META_FN(softplus, phi::SoftplusInferMeta);
PD_REGISTER_INFER_META_FN(softsign, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(spectralnorm, phi::SpectralNormInferMeta);
PD_REGISTER_INFER_META_FN(sqrt, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(square, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(squared_l2_norm, phi::SquaredL2NormInferMeta);
PD_REGISTER_INFER_META_FN(stack, phi::StackInferMeta);
PD_REGISTER_INFER_META_FN(strided_slice, phi::StridedSliceInferMeta);
PD_REGISTER_INFER_META_FN(subtract, phi::ElementwiseInferMeta);
PD_REGISTER_INFER_META_FN(sum, phi::SumInferMeta);
PD_REGISTER_INFER_META_FN(svd, phi::SvdInferMeta);
PD_REGISTER_INFER_META_FN(swish, phi::SwishInferMeta);
PD_REGISTER_INFER_META_FN(sync_batch_norm, phi::BatchNormInferMeta);
PD_REGISTER_INFER_META_FN(take_along_axis, phi::Take_along_axisInferMeta);
PD_REGISTER_INFER_META_FN(tan, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(tanh, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(tanh_shrink, phi::UnchangedInferMeta);
PD_REGISTER_INFER_META_FN(temporal_shift, phi::TemporalShiftInferMeta);
PD_REGISTER_INFER_META_FN(thresholded_relu, phi::Thresholded_reluInferMeta);
PD_REGISTER_INFER_META_FN(tile, phi::TileInferMeta);
PD_REGISTER_INFER_META_FN(top_k, phi::TopKInferMeta);
PD_REGISTER_INFER_META_FN(transpose, phi::TransposeInferMeta);
PD_REGISTER_INFER_META_FN(triangular_solve, phi::TriangularSolveInferMeta);
PD_REGISTER_INFER_META_FN(tril_indices, phi::TrilIndicesInferMeta);
PD_REGISTER_INFER_META_FN(tril_triu, phi::TrilTriuInferMeta);
PD_REGISTER_INFER_META_FN(truncated_gaussian_random, phi::TruncatedGaussianRandomInferMeta);
PD_REGISTER_INFER_META_FN(unbind, phi::UnbindInferMeta);
PD_REGISTER_INFER_META_FN(unfold, phi::UnfoldInferMeta);
PD_REGISTER_INFER_META_FN(uniform_random, phi::UniformRandomInferMeta);
PD_REGISTER_INFER_META_FN(unique, phi::UniqueInferMeta);
PD_REGISTER_INFER_META_FN(unique_consecutive, phi::UniqueConsecutiveInferMeta);
PD_REGISTER_INFER_META_FN(unstack, phi::UnStackInferMeta);
PD_REGISTER_INFER_META_FN(viterbi_decode, phi::ViterbiDecodeInferMeta);
PD_REGISTER_INFER_META_FN(where, phi::WhereInferMeta);
PD_REGISTER_INFER_META_FN(where_index, phi::WhereIndexInferMeta);
PD_REGISTER_INFER_META_FN(yolo_box, phi::YoloBoxInferMeta);
PD_REGISTER_INFER_META_FN(broadcast_tensors, phi::BroadcastTensorsInferMeta);
PD_REGISTER_INFER_META_FN(dirichlet, phi::DirichletInferMeta);
PD_REGISTER_INFER_META_FN(eig, phi::EigInferMeta);
PD_REGISTER_INFER_META_FN(overlap_add, phi::OverlapAddInferMeta);