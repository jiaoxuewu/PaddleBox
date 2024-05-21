// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/details/nan_inf_utils_detail.h"

#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/scope.h"

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#endif
#include "paddle/fluid/framework/convert_utils.h"

#ifdef PADDLE_WITH_XPU
#include "xpu/refactor/math.h"
#endif

namespace paddle {
namespace framework {
namespace details {

static std::once_flag white_list_init_flag;

static int op_role_nan_inf_white_list = 0;

static bool g_check_nan_inf_ret = false;

static constexpr int FORWARD = 0x10000;

// lazy init
static const std::unordered_map<std::string, int>& role_str2int() {
  /* In op_proto_maker.h
   * framework::OpRole::kForward      = 0x0000,
   * framework::OpRole::kBackward     = 0x0001,
   * framework::OpRole::kOptimize     = 0x0002,
   * framework::OpRole::kRPC          = 0x0004,
   * framework::OpRole::kDist         = 0x0008,
   * framework::OpRole::kLRSched      = 0x0010,
   * framework::OpRole::kLoss         = 0x0100,
   * framework::OpRole::kNotSpecified = 0x1000,
   */
  static const std::unordered_map<std::string, int> _role_str2int = {
      {"forward", FORWARD}, /* kForward=0, can't filter */
      {"backward", static_cast<int>(framework::OpRole::kBackward)},
      {"optimize", static_cast<int>(framework::OpRole::kOptimize)},
      {"rpc", static_cast<int>(framework::OpRole::kRPC)},
      {"dist", static_cast<int>(framework::OpRole::kDist)},
      {"lrsched", static_cast<int>(framework::OpRole::kLRSched)},
      {"loss", static_cast<int>(framework::OpRole::kLoss)},
      {"default", static_cast<int>(framework::OpRole::kNotSpecified)},
  };
  return _role_str2int;
}

static std::unordered_set<std::string>& op_type_nan_inf_white_list() {
  static std::unordered_set<std::string> _op_type_nan_inf_white_list = {
      "coalesce_tensor", /* This Op will alloc tensor, and may not init space */
      "rank_attention",   // This Op input param too large init spent time too
                          // long not zero
      "c_mixallgather",   // This Op alloc more space not need init
      "scaled_fc",        // This Op padding same not init data
  };
  return _op_type_nan_inf_white_list;
}

static std::unordered_map<std::string, std::vector<std::string>>&
op_var_nan_inf_white_list() {
  static std::unordered_map<std::string, std::vector<std::string>>
      _op_var_nan_inf_white_list = {
          /* encoded & gather var consist of idx&val, can't judge directly */
          {"dgc", {"__dgc_encoded__", "__dgc_gather__"}},
      };
  return _op_var_nan_inf_white_list;
}

static void InitWhiteListFormEnv() {
  // op_type_skip and op_var_skip may be NULL.
  // So need init static value in there, prevent thread competition.
  // NOTE. role_str2int needn't do this for it only used in this func.
  op_type_nan_inf_white_list();
  op_var_nan_inf_white_list();

  // export PADDLE_INF_NAN_SKIP_OP="op0,op1,op2"
  // export PADDLE_INF_NAN_SKIP_ROLE="role1,role2,role3"
  // export PADDLE_INF_NAN_SKIP_VAR="op0:var0,op0:var1,op1:var0"
  const char* op_type_skip = std::getenv("PADDLE_INF_NAN_SKIP_OP");
  const char* op_role_skip = std::getenv("PADDLE_INF_NAN_SKIP_ROLE");
  const char* op_var_skip = std::getenv("PADDLE_INF_NAN_SKIP_VAR");

  if (op_type_skip != NULL) {
    std::stringstream ss(op_type_skip);
    std::string op_type;
    while (std::getline(ss, op_type, ',')) {
      op_type_nan_inf_white_list().emplace(op_type);
    }
  }

  if (op_role_skip != NULL) {
    std::stringstream ss(op_role_skip);
    std::string op_role;
    while (std::getline(ss, op_role, ',')) {
      PADDLE_ENFORCE_EQ(role_str2int().find(op_role) != role_str2int().end(),
                        true,
                        platform::errors::InvalidArgument(
                            "Skip role must be one of "
                            "{forward,backward,optimize,rpc,dist,lrsched,loss,"
                            "default}, instead of %s",
                            op_role));
      op_role_nan_inf_white_list |= role_str2int().at(op_role);
    }
  }

  if (op_var_skip != NULL) {
    std::stringstream ss(op_var_skip);
    std::string op_var;
    while (std::getline(ss, op_var, ',')) {
      auto pos = op_var.find(":");
      PADDLE_ENFORCE_EQ(
          pos != std::string::npos,
          true,
          platform::errors::InvalidArgument(
              "Skip var format must be op:var, instead of %s", op_var));
      std::string op = op_var.substr(0, pos);
      std::string var = op_var.substr(pos + 1);

      op_var_nan_inf_white_list()[op].emplace_back(var);
    }
  }
  const char* nan_inf_ret = std::getenv("PADDLE_CHECK_INF_NAN_RET");
  if (nan_inf_ret != NULL) {
    g_check_nan_inf_ret = (atoi(nan_inf_ret) > 0);
  }
}

template <typename T>
static void PrintNanInf(const T* value,
                        const size_t numel,
                        int print_num,
                        const std::string& op_type,
                        const std::string& var_name,
                        bool abort = true) {
  T min_value = std::numeric_limits<T>::max();
  T max_value = std::numeric_limits<T>::min();
  size_t nan_count, inf_count, num_count;
  nan_count = inf_count = num_count = 0;

  // CPU print num value
  for (size_t i = 0; i < numel; ++i) {
    size_t count = 0;
    if (std::isnan(value[i])) {
      count = nan_count++;
    } else if (std::isinf(value[i])) {
      count = inf_count++;
    } else {
      count = num_count++;
      min_value = std::min(min_value, value[i]);
      max_value = std::max(max_value, value[i]);
    }

    if (count < static_cast<size_t>(print_num)) {
      printf("numel:%lu index:%lu value:%f\n",
             static_cast<uint64_t>(numel),
             static_cast<uint64_t>(i),
             static_cast<float>(value[i]));
    }
  }
  printf(
      "In cpu, there has %lu,%lu,%lu nan,inf,num. "
      "And in num, min_value is %f, max_value is %f\n",
      static_cast<uint64_t>(nan_count),
      static_cast<uint64_t>(inf_count),
      static_cast<uint64_t>(num_count),
      static_cast<double>(min_value),
      static_cast<double>(max_value));
  if (abort) {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "There are `nan` or `inf` in tensor (%s) of operator (%s).",
        var_name,
        op_type));
  }
}

// openmp 4.0, reduction with fp16
#if defined _OPENMP && _OPENMP >= 201307
// more detail see: 180 page of
// https://www.openmp.org/wp-content/uploads/OpenMP4.0.0.pdf
#pragma omp declare reduction(+ : paddle::platform::float16 : omp_out += omp_in)
#pragma omp declare reduction(+ : paddle::platform::bfloat16 : omp_out += \
                              omp_in)
#pragma omp declare reduction(+ : paddle::platform::complex < \
                                  float > : omp_out += omp_in)
#pragma omp declare reduction(+ : paddle::platform::complex < \
                                  double > : omp_out += omp_in)

#endif

template <typename T>
static void CheckNanInf(const T* value,
                        const size_t numel,
                        int print_num,
                        const std::string& op_type,
                        const std::string& var_name) {
  T sum = static_cast<T>(0.0);
#if defined _OPENMP && _OPENMP >= 201307
#pragma omp parallel for simd reduction(+ : sum)
#elif defined _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
  for (size_t i = 0; i < numel; ++i) {
    sum += (value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    PrintNanInf(value, numel, print_num, op_type, var_name);
  }
}

#if defined _OPENMP && _OPENMP >= 201307
// openmp4.0 not need to specialization fp16
#elif defined _OPENMP
template <>
void CheckNanInf<paddle::platform::float16>(
    const paddle::platform::float16* value,
    const size_t numel,
    int print_num,
    const std::string& op_type,
    const std::string& var_name) {
  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < numel; ++i) {
    sum += static_cast<float>(value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    PrintNanInf(value, numel, print_num, op_type, var_name);
  }
}

template <>
void CheckNanInf<paddle::platform::bfloat16>(
    const paddle::platform::bfloat16* value,
    const size_t numel,
    int print_num,
    const std::string& op_type,
    const std::string& var_name) {
  float sum = 0.0f;
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < numel; ++i) {
    sum += static_cast<float>(value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    PrintNanInf(value, numel, print_num, op_type, var_name);
  }
}

template <>
void CheckNanInf<paddle::platform::complex<float>>(
    const paddle::platform::complex<float>* value,
    const size_t numel,
    int print_num,
    const std::string& op_type,
    const std::string& var_name) {
  float real_sum = 0.0f;
#pragma omp parallel for reduction(+ : real_sum)
  for (size_t i = 0; i < numel; ++i) {
    real_sum += (value[i].real - value[i].real);
  }

  float imag_sum = 0.0f;
#pragma omp parallel for reduction(+ : imag_sum)
  for (size_t i = 0; i < numel; ++i) {
    imag_sum += (value[i].imag - value[i].imag);
  }

  if (std::isnan(real_sum) || std::isinf(real_sum) || std::isnan(imag_sum) ||
      std::isinf(imag_sum)) {
    // hot fix for compile failed in gcc4.8
    // here also need print detail info of nan or inf later
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "There are `nan` or `inf` in tensor (%s) of operator (%s).",
        var_name,
        op_type));
  }
}

template <>
    void CheckNanInf < paddle::platform::complex < double >>>
    (const paddle::platform::complex<double>* value,
     const size_t numel,
     int print_num,
     const std::string& op_type,
     const std::string& var_name) {
  double real_sum = 0.0;
#pragma omp parallel for reduction(+ : real_sum)
  for (size_t i = 0; i < numel; ++i) {
    real_sum += (value[i].real - value[i].real);
  }

  double imag_sum = 0.0;
#pragma omp parallel for reduction(+ : imag_sum)
  for (size_t i = 0; i < numel; ++i) {
    imag_sum += (value[i].imag - value[i].imag);
  }

  if (std::isnan(real_sum) || std::isinf(real_sum) || std::isnan(imag_sum) ||
      std::isinf(imag_sum)) {
    // hot fix for compile failed in gcc4.8
    // here also need print detail info of nan or inf later
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "There are `nan` or `inf` in tensor (%s) of operator (%s).",
        var_name,
        op_type));
  }
}

#endif

template <>
template <typename T>
void TensorCheckerVisitor<phi::CPUContext>::apply(
    typename std::enable_if<
        std::is_floating_point<T>::value ||
        std::is_same<T, ::paddle::platform::complex<float>>::value ||
        std::is_same<T, ::paddle::platform::complex<double>>::value>::type*)
    const {
  // use env strategy control in future, -1=print_all.
  int print_num = 3;
  CheckNanInf(
      tensor_.data<T>(), tensor_.numel(), print_num, op_type_, var_name_);
}

template <>
void tensor_check<phi::CPUContext>(const std::string& op_type,
                                   const std::string& var_name,
                                   const framework::Tensor& tensor,
                                   const platform::Place& place) {
  TensorCheckerVisitor<phi::CPUContext> vistor(
      op_type, var_name, tensor, place);
  VisitDataType(framework::TransToProtoVarType(tensor.dtype()), vistor);
}

void CheckVarHasNanOrInf(const std::string& op_type,
                         const std::string& var_name,
                         const framework::Variable* var,
                         const platform::Place& place) {
  PADDLE_ENFORCE_NOT_NULL(
      var,
      platform::errors::NotFound(
          "Cannot find var: `%s` in op `%s`.", var_name, op_type));

  const Tensor* tensor{nullptr};
  if (var->IsType<framework::LoDTensor>()) {
    tensor = &var->Get<framework::LoDTensor>();
  } else if (var->IsType<phi::SelectedRows>()) {
    tensor = &var->Get<phi::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check " << op_type << " var_name:" << var_name
           << ", place:" << tensor->place() << ", numel:" << tensor->numel();

  if (platform::is_gpu_place(tensor->place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    tensor_check<phi::GPUContext>(op_type, var_name, *tensor, place);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Tensor[%s] use gpu place. PaddlePaddle must compile with GPU.",
        var_name));
#endif
    return;
  } else if (platform::is_xpu_place(tensor->place()) || platform::is_xpul3_place(tensor->place())) {
#ifdef PADDLE_WITH_XPU
    if (framework::TransToProtoVarType(tensor->dtype()) !=
        proto::VarType::FP32) {
      return;
    }

    // float* cpu_data = new float[tensor->numel()];
    // memory::Copy(platform::CPUPlace(),
    //              static_cast<void*>(cpu_data),
    //              tensor->place(),
    //              static_cast<const void*>(tensor->data<float>()),
    //              tensor->numel() * sizeof(float));
    // bool flag = false;
    // for (int i = 0; i < tensor->numel(); i++) {
    //   if (isnan(cpu_data[i]) || isinf(cpu_data[i])) {
    //     flag = true;
    //     break;
    //   }
    // }
    // delete[] cpu_data;

    using XPUType = typename XPUTypeTrait<float>::Type;
    platform::XPUDeviceContext* dev_ctx = dynamic_cast<platform::XPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(tensor->place()));
    const XPUType* x = reinterpret_cast<const XPUType*>(tensor->data<float>());

    Tensor y_tensor;
    bool* y_ptr = y_tensor.mutable_data<bool>({1}, place);
    int r = xpu::check_nan_or_inf<XPUType>(dev_ctx->x_context(),
                              x,
                              y_ptr,
                              tensor->numel());
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
            platform::errors::External(
               "The check_nan_or_inf XPU OP return wrong value[%d %s]",
               r, XPUAPIErrorMsg[r]));
    dev_ctx->Wait();

    bool check_res = false;
    bool* res_ptr = &check_res;
    memory::Copy(platform::CPUPlace(),
                 static_cast<void*>(res_ptr),
                 y_tensor.place(),
                 static_cast<const void*>(y_tensor.data<bool>()),
                 y_tensor.numel() * sizeof(bool));
    VLOG(3) << "CheckVarHasNanOrInfRet check_res = " << check_res;
    PADDLE_ENFORCE_NE(
        check_res,
        true,
        platform::errors::Fatal(
            "Operator %s output Tensor %s contains Inf.", op_type, var_name));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Tensor[%s] use xpu place. PaddlePaddle must compile with XPU.",
        var_name));
#endif
    return;
  } else if (platform::is_npu_place(tensor->place())) {
#ifdef PADDLE_WITH_ASCEND_CL
    if (framework::TransToProtoVarType(tensor->dtype()) !=
        proto::VarType::FP32) {
      return;
    }

    framework::LoDTensor cpu_tensor;
    cpu_tensor.Resize(tensor->dims());
    float* cpu_data = static_cast<float*>(
        cpu_tensor.mutable_data(platform::CPUPlace(), tensor->dtype()));

    framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
    bool flag = false;
    for (int i = 0; i < cpu_tensor.numel(); i++) {
      if (isnan(cpu_data[i]) || isinf(cpu_data[i])) {
        flag = true;
        break;
      }
    }
    PADDLE_ENFORCE_NE(
        flag,
        true,
        platform::errors::Fatal(
            "Operator %s output Tensor %s contains Inf.", op_type, var_name));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Tensor[%s] use npu place. PaddlePaddle must compile with NPU.",
        var_name));
#endif
    return;
  }
  tensor_check<phi::CPUContext>(op_type, var_name, *tensor, place);
}

void CheckVarHasNanOrInf(const std::string& op_type,
                         const framework::ScopeBase& scope,
                         const std::string& var_name,
                         const platform::Place& place) {
  auto* var = scope.FindVar(var_name);
  CheckVarHasNanOrInf(op_type, var_name, var, place);
}

bool IsSkipOp(const framework::OperatorBase& op) {
  if (op_type_nan_inf_white_list().count(op.Type()) != 0) return true;

  int op_role = 0;
  if (op.HasAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName())) {
    op_role = op.template Attr<int>(
        framework::OpProtoAndCheckerMaker::OpRoleAttrName());
  }

  // kForward=0, can't filter
  if (op_role == static_cast<int>(framework::OpRole::kForward)) {
    op_role = FORWARD;
  }
  if (op_role_nan_inf_white_list & op_role) return true;

  return false;
}

#ifdef PADDLE_WITH_ASCEND_CL
using NpuOpRunner = paddle::operators::NpuOpRunner;

constexpr int FLOAT_STATUS_SIZE = 8;

static framework::Tensor& npu_float_status() {
  static framework::Tensor float_status;
  return float_status;
}

void NPUAllocAndClearFloatStatus(const framework::OperatorBase& op,
                                 const framework::ScopeBase& scope,
                                 const platform::Place& place) {
  if (!platform::is_npu_place(place)) return;

  std::call_once(white_list_init_flag, InitWhiteListFormEnv);
  if (IsSkipOp(op)) return;

  auto* dev_ctx = reinterpret_cast<platform::NPUDeviceContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto stream = dev_ctx->stream();

  auto& flag = npu_float_status();
  flag.mutable_data<float>({FLOAT_STATUS_SIZE}, place);
  NpuOpRunner("NPUAllocFloatStatus", {}, {flag}).Run(stream);

  framework::Tensor tmp;
  tmp.mutable_data<float>({FLOAT_STATUS_SIZE}, place);
  NpuOpRunner("NPUClearFloatStatus", {tmp}, {flag}).Run(stream);
}

void PrintNpuVarInfo(const std::string& op_type,
                     const std::string& var_name,
                     const framework::Variable* var,
                     const platform::Place& place) {
  const Tensor* tensor{nullptr};
  if (var->IsType<framework::LoDTensor>()) {
    tensor = &var->Get<framework::LoDTensor>();
  } else if (var->IsType<phi::SelectedRows>()) {
    tensor = &var->Get<phi::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if ((framework::TransToProtoVarType(tensor->dtype()) !=
       proto::VarType::FP32) &&
      (framework::TransToProtoVarType(tensor->dtype()) !=
       proto::VarType::FP16)) {
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check " << op_type << " var_name:" << var_name
           << ", place:" << tensor->place() << ", numel:" << tensor->numel();

  framework::Tensor cpu_tensor;
  cpu_tensor.Resize(tensor->dims());
  cpu_tensor.mutable_data(platform::CPUPlace(), tensor->dtype());
  framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);

  LOG(WARNING) << "print [" << var_name << "] tensor info:";
  // use env strategy control in future, -1=print_all.
  int print_num = 3;
  if (framework::TransToProtoVarType(tensor->dtype()) == proto::VarType::FP32) {
    const float* value = cpu_tensor.data<float>();
    PrintNanInf(value, tensor->numel(), print_num, op_type, var_name, false);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::FP16) {
    const paddle::platform::float16* value =
        cpu_tensor.data<paddle::platform::float16>();
    PrintNanInf(value, tensor->numel(), print_num, op_type, var_name, false);
  }
}

void PrintNPUOpValueInfo(const framework::OperatorBase& op,
                         const framework::ScopeBase& scope,
                         const platform::Place& place) {
  LOG(WARNING) << "There are `nan` or `inf` in operator (" << op.Type()
               << "), here we print some tensor value info of this op.";
  for (auto& vname : op.InputVars()) {
    auto* var = scope.FindVar(vname);
    if (var == nullptr) continue;
    PrintNpuVarInfo(op.Type(), vname, var, place);
  }

  for (auto& vname : op.OutputVars(true)) {
    auto* var = scope.FindVar(vname);
    if (var == nullptr) continue;
    PrintNpuVarInfo(op.Type(), vname, var, place);
  }
}

static void NPUCheckOpHasNanOrInf(const framework::OperatorBase& op,
                                  const framework::ScopeBase& scope,
                                  const platform::Place& place) {
  if (!platform::is_npu_place(place)) return;

  auto* dev_ctx = reinterpret_cast<platform::NPUDeviceContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto stream = dev_ctx->stream();

  auto& flag = npu_float_status();
  Tensor tmp;
  tmp.mutable_data<float>({FLOAT_STATUS_SIZE}, place);
  // NPUGetFloatStatus updates data on input in-place.
  // tmp is only placeholder.
  NpuOpRunner("NPUGetFloatStatus", {flag}, {tmp}).Run(stream);

  framework::Tensor cpu_tensor;
  auto cpu_place = platform::CPUPlace();
  float* cpu_data = static_cast<float*>(
      cpu_tensor.mutable_data<float>({FLOAT_STATUS_SIZE}, cpu_place));

  framework::TensorCopySync(flag, cpu_place, &cpu_tensor);
  float sum = 0.0;
  for (int i = 0; i < FLOAT_STATUS_SIZE; ++i) {
    sum += cpu_data[i];
  }

  if (sum >= 1.0) PrintNPUOpValueInfo(op, scope, place);

  PADDLE_ENFORCE_LT(sum,
                    1.0,
                    platform::errors::PreconditionNotMet(
                        "Operator %s contains Nan/Inf.", op.Type()));
}
#endif

void CheckOpHasNanOrInf(const framework::OperatorBase& op,
                        const framework::ScopeBase& exec_scope,
                        const platform::Place& place) {
  std::call_once(white_list_init_flag, InitWhiteListFormEnv);

  if (IsSkipOp(op)) return;

#ifdef PADDLE_WITH_ASCEND_CL
  if (platform::is_npu_place(place)) {
    NPUCheckOpHasNanOrInf(op, exec_scope, place);
    return;
  }
#endif

  if (op_var_nan_inf_white_list().count(op.Type()) == 0) {
    // NOTE. vname may destruct in the end of this func.
    for (auto& vname : op.OutputVars(true)) {
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      CheckVarHasNanOrInf(op.Type(), exec_scope, vname, place);
    }
  } else {
    for (auto& vname : op.OutputVars(true)) {
      bool need_check = true;
      for (auto& white_vname : op_var_nan_inf_white_list().at(op.Type())) {
        if (vname.find(white_vname) != std::string::npos) {
          need_check = false;
          break;
        }
      }
      if (!need_check) continue;
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      CheckVarHasNanOrInf(op.Type(), exec_scope, vname, place);
    }
  }
}
inline unsigned int& get_cpu_nan_inf_num(void) {
  thread_local unsigned int nan_inf_num = 0;
  return nan_inf_num;
}
static unsigned int* get_device_num_ptr(const platform::Place& place) {
#ifdef PADDLE_WITH_CUDA
  thread_local paddle::memory::AllocationPtr gpu_tensor = nullptr;
  if (gpu_tensor == nullptr) {
    auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(
          platform::DeviceContextPool::Instance().Get(place));
    gpu_tensor = paddle::memory::Alloc(*dev_ctx, sizeof(unsigned int));
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(
        gpu_tensor->ptr(), 0, sizeof(unsigned int), dev_ctx->stream()));
  }
  return reinterpret_cast<unsigned int*>(gpu_tensor->ptr());
#else
  PADDLE_THROW(platform::errors::Unimplemented("get_device_num_ptr not support."));
  return nullptr;
#endif
}
template <typename T>
static int CheckNanInfRet(const T* value,
                        const size_t numel) {
  T sum = static_cast<T>(0.0);
#ifndef PADDLE_WITH_XPU

#if defined _OPENMP && _OPENMP >= 201307
#pragma omp parallel for simd reduction(+ : sum)
#elif defined _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif

#endif // end PADDLE_WITH_XPU
  for (size_t i = 0; i < numel; ++i) {
    sum += (value[i] - value[i]);
  }

  if (std::isnan(sum) || std::isinf(sum)) {
    return 1;
  }
  return 0;
}
void CheckVarHasNanOrInfRet(const std::string& op_type,
                            const framework::Variable* var,
                            const std::string& var_name,
                            const platform::Place& place) {
  const Tensor* tensor{nullptr};
  if (var->IsType<framework::LoDTensor>()) {
    tensor = &var->Get<framework::LoDTensor>();
  } else if (var->IsType<phi::SelectedRows>()) {
    tensor = &var->Get<phi::SelectedRows>().value();
  } else {
    LOG(WARNING) << "op_type: " << op_type << ", var_name: " << var_name << " var->IsType invalid";
    return;
  }
  if (tensor->memory_size() == 0) {
    return;
  }
  if (tensor->numel() == 0) {
    LOG(WARNING) << "op_type: " << op_type << ", var_name: " << var_name << " tensor->numel() is zero";
    return;
  }
  if (platform::is_cpu_place(tensor->place())) {
    int64_t numel = tensor->numel();
    auto dtype = framework::TransToProtoVarType(tensor->type());
    if (dtype == proto::VarType::FP32) {
      const float *val = tensor->data<float>();
      get_cpu_nan_inf_num() += CheckNanInfRet(val, numel);
    } else if (dtype == proto::VarType::FP64) {
      const double *val = tensor->data<double>();
      get_cpu_nan_inf_num() += CheckNanInfRet(val, numel);
    }
    return;
  } else if (platform::is_xpu_place(tensor->place()) || platform::is_xpul3_place(tensor->place())) {
#ifdef PADDLE_WITH_XPU
    if (framework::TransToProtoVarType(tensor->dtype()) !=
        proto::VarType::FP32) {
      // LOG(WARNING) << "skip op_type: " << op_type << "check_nan_inf, tensor type:" << tensor->dtype() << " not float32!";
      return;
    }

    // float* cpu_data = new float[tensor->numel()];
    // memory::Copy(platform::CPUPlace(),
    //              static_cast<void*>(cpu_data),
    //              tensor->place(),
    //              static_cast<const void*>(tensor->data<float>()),
    //              tensor->numel() * sizeof(float));
    // // bool flag = false;
    // for (int64_t i = 0; i < tensor->numel(); i++) {
    //   if (isnan(cpu_data[i]) || isinf(cpu_data[i])) {
    //     get_cpu_nan_inf_num() ++;
    //     break;
    //   }
    // }
    // delete[] cpu_data;

    using XPUType = typename XPUTypeTrait<float>::Type;
    platform::XPUDeviceContext* dev_ctx = dynamic_cast<platform::XPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(tensor->place()));
    const XPUType* x = reinterpret_cast<const XPUType*>(tensor->data<float>());

    Tensor y_tensor;
    bool* y_ptr = y_tensor.mutable_data<bool>({1}, place);
    VLOG(1) << "Check its output indeed:" << var_name;
    int r = xpu::check_nan_or_inf<XPUType>(dev_ctx->x_context(),
                              x,
                              y_ptr,
                              tensor->numel());
    dev_ctx->Wait();

    if (r != xpu::Error_t::SUCCESS) {
        LOG(WARNING) << "op_type: " << op_type << ", var_name: " << var_name << "check_failed!";
        return;
    }
    bool check_res = false;
    bool* res_ptr = &check_res;
    memory::Copy(platform::CPUPlace(),
                 static_cast<void*>(res_ptr),
                 y_tensor.place(),
                 static_cast<const void*>(y_tensor.data<bool>()),
                 y_tensor.numel() * sizeof(bool));
    VLOG(1) << "CheckVarHasNanOrInfRet check_res = " << check_res;
    if (check_res) {
      get_cpu_nan_inf_num() ++;
      VLOG(0) << "op_type: " << op_type << ", var_name: " << var_name << "check nan faild!";
    }
    return;
#endif
  }
#if defined(PADDLE_WITH_CUDA)
  unsigned int* dnum = get_device_num_ptr(place);
  CudaTensorCheckNanInf(*tensor, dnum);
#endif
}

bool CheckBatchNanOrInfRet(const platform::Place& place) {
  if (!platform::is_gpu_place(place)) {
    return (get_cpu_nan_inf_num() > 0);
  }
#ifdef PADDLE_WITH_CUDA
  auto* dev_ctx = reinterpret_cast<phi::GPUContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto stream = dev_ctx->stream();
  unsigned int* num_ptr = get_device_num_ptr(place);
  thread_local unsigned int nan_inf_num = 0;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(&nan_inf_num, num_ptr,
                                              sizeof(unsigned int),
                                              cudaMemcpyDeviceToHost, stream));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaStreamSynchronize(stream));
  if ((nan_inf_num + get_cpu_nan_inf_num()) > 0) {
    return true;
  }
#endif
  return false;
}
bool CheckOpHasNanOrInfRet(const framework::OperatorBase& op,
                           const framework::Scope& exec_scope,
                           const platform::Place& place) {
  std::call_once(white_list_init_flag, InitWhiteListFormEnv);

  if (IsSkipOp(op)) return false;
  if (op_var_nan_inf_white_list().count(op.Type()) == 0) {
    VLOG(1) << "Check op:" << op.Type();
    // NOTE. vname may destruct in the end of this func.
    for (auto& vname : op.OutputVars(true)) {
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      VLOG(1) << "Check its output:" << vname;
      CheckVarHasNanOrInfRet(op.Type(), var, vname, place);
    }
  } else {
    VLOG(1) << "Check op:" << op.Type();
    for (auto& vname : op.OutputVars(true)) {
      bool need_check = true;
      for (auto& white_vname : op_var_nan_inf_white_list().at(op.Type())) {
        if (vname.find(white_vname) != std::string::npos) {
          need_check = false;
          break;
        }
      }
      if (!need_check) continue;
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      VLOG(1) << "Check its output:" << vname;
      CheckVarHasNanOrInfRet(op.Type(), var, vname, place);
    }
  }
  if (g_check_nan_inf_ret) {
    // every op check
    return CheckBatchNanOrInfRet(place);
  }
  return false;
}

void DumpTensorToFile(const std::string& path, const std::string& prefix,
                      const std::string& iname, const Scope& exec_scope) {
  auto* var = exec_scope.FindVar(iname);
  if (var == nullptr) {
    return;
  }
  if (!var->IsInitialized()) {
    return;
  }
  auto& tensor = var->Get<framework::LoDTensor>();
  if (!tensor.IsInitialized()) {
    return;
  }

  std::ostringstream os;
  if (var->IsType<phi::DenseTensor>()) {
    os << var->Get<phi::DenseTensor>();
  } else if (var->IsType<phi::SelectedRows>()) {
    os << var->Get<phi::SelectedRows>().value();
  }
  os << "\n";
  std::string s = os.str();

  char file_name[2048] = {0};
  snprintf(file_name, sizeof(file_name), "%s/%s_%s", path.c_str(),
           prefix.c_str(), iname.c_str());

  std::ofstream out(file_name, std::ios::binary);
  out.write(s.c_str(), s.length());
  out.close();
}

void DumpAllScope(const Scope& exec_scope, const platform::Place& place) {
  int device_id = 0;
#if (defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_XPU)) && !defined(_WIN32)
  device_id = place.GetDeviceId();
#endif
  VLOG(0) << "begin dump scope all tensor data, device id=" << device_id;
  std::string log_path = "./nan_inf";
  if (!PathExists(log_path)) {
    MkDirRecursively(log_path.c_str());
  }
  // dump scope all data
  char prefix[128] = {0};
  snprintf(prefix, sizeof(prefix), "gpu%d", device_id);
  for (auto& iname : exec_scope.LocalVarNames()) {
    DumpTensorToFile(log_path, prefix, iname, exec_scope);
  }
  VLOG(0) << "end dump scope all tensor data, device id=" << device_id;
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
