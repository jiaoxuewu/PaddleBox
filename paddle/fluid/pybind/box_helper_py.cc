/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/pybind/box_helper_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindBoxHelper(py::module* m) {
  py::class_<framework::BoxHelper, std::shared_ptr<framework::BoxHelper>>(
      *m, "BoxPS")
      .def(py::init([](paddle::framework::Dataset* dataset) {
        return std::make_shared<paddle::framework::BoxHelper>(dataset);
      }))
      .def("set_date",
           &framework::BoxHelper::SetDate,
           py::call_guard<py::gil_scoped_release>())
      .def("begin_pass",
           &framework::BoxHelper::BeginPass,
           py::call_guard<py::gil_scoped_release>())
      .def("end_pass",
           &framework::BoxHelper::EndPass,
           py::call_guard<py::gil_scoped_release>())
      .def("wait_feed_pass_done",
           &framework::BoxHelper::WaitFeedPassDone,
           py::call_guard<py::gil_scoped_release>())
      .def("preload_into_memory",
           &framework::BoxHelper::PreLoadIntoMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("load_into_memory",
           &framework::BoxHelper::LoadIntoMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("slots_shuffle",
           &framework::BoxHelper::SlotsShuffle,
           py::call_guard<py::gil_scoped_release>())
      .def("read_ins_into_memory",
           &framework::BoxHelper::ReadData2Memory,
           py::call_guard<py::gil_scoped_release>());
}  // end BoxHelper

#ifdef PADDLE_WITH_BOX_PS
void BindBoxWrapper(py::module* m) {
  py::class_<framework::BoxWrapper, std::shared_ptr<framework::BoxWrapper>>(
      *m, "BoxWrapper")
      .def(py::init([](int embedx_dim,
                       int expand_embed_dim,
                       int feature_type,
                       float pull_embedx_scale) {
        // return std::make_shared<paddle::framework::BoxHelper>(dataset);
        return framework::BoxWrapper::SetInstance(
            embedx_dim, expand_embed_dim, feature_type, pull_embedx_scale);
      }))
      .def("save_base",
           &framework::BoxWrapper::SaveBase,
           py::call_guard<py::gil_scoped_release>())
      .def("feed_pass",
           &framework::BoxWrapper::FeedPass,
           py::call_guard<py::gil_scoped_release>())
      .def("set_test_mode",
           &framework::BoxWrapper::SetTestMode,
           py::call_guard<py::gil_scoped_release>())
      .def("save_delta",
           &framework::BoxWrapper::SaveDelta,
           py::call_guard<py::gil_scoped_release>())
      .def("initialize_gpu_and_load_model",
           &framework::BoxWrapper::InitializeGPUAndLoadModel,
           py::call_guard<py::gil_scoped_release>())
      .def("initialize_auc_runner",
           &framework::BoxWrapper::InitializeAucRunner,
           py::call_guard<py::gil_scoped_release>())
      .def("init_metric",
           &framework::BoxWrapper::InitMetric,
           py::arg("method"),
           py::arg("name"),
           py::arg("label_varname"),
           py::arg("pred_varname"),
           py::arg("cmatch_rank_varname"),
           py::arg("mask_varname"),
           py::arg("metric_phase"),
           py::arg("cmatch_rank_group"),
           py::arg("ignore_rank"),
           py::arg("bucket_size") = 1000000,
           py::arg("mode_collect_in_gpu") = false,
           py::arg("max_batch_size") = 0,
           py::arg("sample_scale_varnam") = "",
           py::call_guard<py::gil_scoped_release>())
      .def("get_metric_msg",
           &framework::BoxWrapper::GetMetricMsg,
           py::call_guard<py::gil_scoped_release>())
      .def("get_continue_metric_msg",
           &framework::BoxWrapper::GetContinueMetricMsg,
           py::call_guard<py::gil_scoped_release>())
      .def("get_nan_inf_metric_msg",
           &framework::BoxWrapper::GetNanInfMetricMsg,
           py::call_guard<py::gil_scoped_release>())
      .def("get_metric_name_list",
           &framework::BoxWrapper::GetMetricNameList,
           py::call_guard<py::gil_scoped_release>())
      .def("flip_phase",
           &framework::BoxWrapper::FlipPhase,
           py::call_guard<py::gil_scoped_release>())
      .def("set_phase",
           &framework::BoxWrapper::SetPhase,
           py::call_guard<py::gil_scoped_release>())
      .def("init_afs_api",
           &framework::BoxWrapper::InitAfsAPI,
           py::call_guard<py::gil_scoped_release>())
      .def("finalize",
           &framework::BoxWrapper::Finalize,
           py::call_guard<py::gil_scoped_release>())
      .def("release_pool",
           &framework::BoxWrapper::ReleasePool,
           py::call_guard<py::gil_scoped_release>())
      .def("set_dataset_name",
           &framework::BoxWrapper::SetDatasetName,
           py::call_guard<py::gil_scoped_release>())
      .def("set_input_table_dim",
           &framework::BoxWrapper::SetInputTableDim,
           py::call_guard<py::gil_scoped_release>())
      .def("shrink_table",
           &framework::BoxWrapper::ShrinkTable,
           py::call_guard<py::gil_scoped_release>())
      .def("load_ssd2mem",
           &framework::BoxWrapper::LoadSSD2Mem,
           py::call_guard<py::gil_scoped_release>())
      .def("shrink_resource",
           &framework::BoxWrapper::ShrinkResource,
           py::call_guard<py::gil_scoped_release>())
      .def("merge_model",
           &framework::BoxWrapper::MergeModel,
           py::call_guard<py::gil_scoped_release>())
      .def("print_device_info",
           &framework::BoxWrapper::PrintDeviceInfo,
           py::call_guard<py::gil_scoped_release>())
      .def("merge_multi_models",
           &framework::BoxWrapper::MergeMultiModels,
           py::call_guard<py::gil_scoped_release>())
      .def("merge_multi_models", 
           &framework::BoxWrapper::MergeMultiModels,
           py::call_guard<py::gil_scoped_release>());
}  // end BoxWrapper
void BindBoxFileMgr(py::module* m) {
  py::class_<framework::BoxFileMgr, std::shared_ptr<framework::BoxFileMgr>>(
      *m, "BoxFileMgr")
      .def(py::init([]() { return std::make_shared<framework::BoxFileMgr>(); }))
      .def("init",
           &framework::BoxFileMgr::init,
           py::call_guard<py::gil_scoped_release>())
      .def("list_dir",
           &framework::BoxFileMgr::list_dir,
           py::call_guard<py::gil_scoped_release>())
      .def("makedir",
           &framework::BoxFileMgr::makedir,
           py::call_guard<py::gil_scoped_release>())
      .def("exists",
           &framework::BoxFileMgr::exists,
           py::call_guard<py::gil_scoped_release>())
      .def("download",
           &framework::BoxFileMgr::down,
           py::call_guard<py::gil_scoped_release>())
      .def("upload",
           &framework::BoxFileMgr::upload,
           py::call_guard<py::gil_scoped_release>())
      .def("remove",
           &framework::BoxFileMgr::remove,
           py::call_guard<py::gil_scoped_release>())
      .def("file_size",
           &framework::BoxFileMgr::file_size,
           py::call_guard<py::gil_scoped_release>())
      .def("dus",
           &framework::BoxFileMgr::dus,
           py::call_guard<py::gil_scoped_release>())
      .def("truncate",
           &framework::BoxFileMgr::truncate,
           py::call_guard<py::gil_scoped_release>())
      .def("touch",
           &framework::BoxFileMgr::touch,
           py::call_guard<py::gil_scoped_release>())
      .def("rename",
           &framework::BoxFileMgr::rename,
           py::call_guard<py::gil_scoped_release>())
      .def("list_info",
           &framework::BoxFileMgr::list_info,
           py::call_guard<py::gil_scoped_release>())
      .def("count",
           &framework::BoxFileMgr::count,
           py::call_guard<py::gil_scoped_release>())
      .def("finalize",
           &framework::BoxFileMgr::destory,
           py::call_guard<py::gil_scoped_release>());
}  // end BoxFileMgr
#endif

}  // end namespace pybind
}  // end namespace paddle
