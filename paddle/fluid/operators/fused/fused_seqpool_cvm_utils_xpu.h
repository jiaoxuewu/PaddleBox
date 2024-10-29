#include "paddle/fluid/framework/tensor_util.h"
#pragma once


namespace paddle {
namespace operators {

void check_negative(Place& place,
                    xpu::Context* xpu_context,
                    float* cvm_data,
                    int cvm_len) {
  framework::LoDTensor result;
  result.Resize(phi::make_ddim({1}));
  bool* d_result = result.mutable_data<bool>(place);
  int r = xpu::check_negative(xpu_context, cvm_data, d_result, cvm_len);
  PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
          platform::errors::External(
            "The check_negative return wrong value[%d %s]",
            r, XPUAPIErrorMsg[r]));
  xpu_wait(xpu_context->xpu_stream);
  // std::vector<bool> h_result(1);//vector<bool> have no data()
  bool h_result = 0;
  xpu_memcpy(&h_result, d_result, sizeof(bool), XPU_DEVICE_TO_HOST);
  if(h_result != 0) {
    std::vector<float> h_cvm(cvm_len);
    xpu_memcpy(h_cvm.data(), cvm_data, cvm_len * sizeof(float), XPU_DEVICE_TO_HOST);

    auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm* ptm = localtime(&now_time);
    char date[100] = {0};
    snprintf(date, 100, "%d%02d%02d%02d%02d%02d",
            (int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
            (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
    std::stringstream name_ss;
    name_ss << "cvm-val.dev-" << (int)place.GetDeviceId() << "-" << date << ".dump";
    std::ofstream ofs;
    ofs.open(name_ss.str(), std::ios::app);

    for (uint32_t i = 0; i < h_cvm.size(); ++i) {
      ofs << h_cvm[i] << "\n";
    }

    ofs.close();
  }
  PADDLE_ENFORCE_EQ(h_result, 0,
          platform::errors::PreconditionNotMet(
            "The check_negative found negative in cvm, check file cvm-val.dev.dump"));
}

template <typename T>
void check_tensors_nan(Place& place,
                       xpu::Context* xpu_context,
                       T& x,  //  std::vector<paddle::framework::LoDTensor*>& x,
                       std::string name) {
  int slot_num = x.size();
  for (int i = 0; i < slot_num; i++) {
    if(x[i]->numel()==0)
      continue;

    const float* ptr = x[i]->template data<float>();
    int len = x[i]->numel();
    int line_len = x[0]->dims()[1];
    // float nan = log(-1.0);
    // xpu_memcpy(const_cast<T*>(x_ptr), &nan, sizeof(float), XPU_HOST_TO_DEVICE);

    framework::LoDTensor result;
    bool* d_result = result.mutable_data<bool>({1}, place);
    int r = xpu::check_nan_or_inf<float>(xpu_context,
                              ptr,
                              d_result,
                              len);
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
            platform::errors::External(
              "The check_nan_or_inf return wrong value[%d %s]",
              r, XPUAPIErrorMsg[r]));
    xpu_wait(xpu_context->xpu_stream);
    bool h_result = 0;
    xpu_memcpy(&h_result, d_result, sizeof(bool), XPU_DEVICE_TO_HOST);
    if(h_result != 0) {
      std::vector<float> h_x(len);
      xpu_memcpy(h_x.data(), ptr, len * sizeof(float), XPU_DEVICE_TO_HOST);

      auto now_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      struct tm* ptm = localtime(&now_time);
      char date[100] = {0};
      snprintf(date, 100, "%d%02d%02d%02d%02d%02d",
              (int)ptm->tm_year + 1900, (int)ptm->tm_mon + 1, (int)ptm->tm_mday,
              (int)ptm->tm_hour, (int)ptm->tm_min, (int)ptm->tm_sec);
      std::stringstream name_ss;
      name_ss << name << "[" << i << "].dev-" << (int)place.GetDeviceId() << "-" << date << ".dump";
      std::ofstream ofs;
      ofs.open(name_ss.str(), std::ios::app);

      for (int j = 0; j < (int)h_x.size(); ++j) {
        ofs << h_x[j] << " ";
        if (j % line_len == (line_len-1)) {
          ofs << "\n";
        }
      }

      ofs.close();
    }
    PADDLE_ENFORCE_EQ(h_result, 0,
            platform::errors::PreconditionNotMet(
              "The check_nan_or_inf found something, check file " + name + ".dev.dump"));
  }
}

}  // namespace operators
}  // namespace paddle