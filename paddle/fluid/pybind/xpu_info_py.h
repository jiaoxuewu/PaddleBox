#pragma once
#ifdef PADDLE_WITH_XPU
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include "paddle/fluid/platform/device/xpu/xpu_info.h"
#include "paddle/fluid/platform/device/xpu/xpu_op_list.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

void BindXPUInfo(py::module* m);

}  // namespace pybind
}  // namespace paddle
#endif // PADDLE_WITH_XPU
