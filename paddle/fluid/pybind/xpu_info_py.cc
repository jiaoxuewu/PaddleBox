#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/pybind/xpu_info_py.h"
#include "tuple"

namespace paddle {
namespace pybind {

void BindXPUInfo(py::module* m) {
    py::class_<platform::XPUMLHandler>(*m, "XPUMLHandler")
      .def(py::init<>())
      .def("getMemoryUsageTuple", &platform::XPUMLHandler::getMemoryUsageTuple)
      .def("getL3UsageTuple", &platform::XPUMLHandler::getL3UsageTuple);
}

}  // namespace pybind
}  // namespace paddle
#endif // PADDLE_WITH_XPU
