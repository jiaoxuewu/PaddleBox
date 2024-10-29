if(WITH_PSLIB)
  if(WITH_PSLIB_BRPC)
    set(BRPC_DEPS pslib_brpc)
  else()
    if(NOT WITH_HETERPS)
      set(BRPC_DEPS brpc)
    endif()
  endif()
  cc_library(
    fleet_wrapper
    SRCS fleet_wrapper.cc
    DEPS framework_proto
         proto_desc
         op_registry
         variable_helper
         scope
         ${BRPC_DEPS}
         pslib)
else()
  cc_library(
    fleet_wrapper
    SRCS fleet_wrapper.cc
    DEPS framework_proto variable_helper scope)
endif()

if(WITH_HETERPS)
  if(WITH_NCCL AND WITH_GPU)
    if(WITH_PSCORE)
      nv_library(
        ps_gpu_wrapper
        SRCS ps_gpu_wrapper.cu ps_gpu_wrapper.cc
        DEPS heter_ps gloo_wrapper ps_framework_proto ${BRPC_DEPS})
    else()
      nv_library(
        ps_gpu_wrapper
        SRCS ps_gpu_wrapper.cu ps_gpu_wrapper.cc
        DEPS heter_ps gloo_wrapper ${BRPC_DEPS})
    endif()
    add_subdirectory(heter_ps)
  elseif(WITH_XPU_KP)
    set(SCALOPUS_PATH "${PROJECT_BINARY_DIR}/third_party/xpu/src/extern_xpu")
    include_directories("${SCALOPUS_PATH}/scalopus/include")
    include_directories("${SCALOPUS_PATH}/scalopus/include_consumer")
    include_directories("${SCALOPUS_PATH}/scalopus/thirdparty/nlohmann_json/single_include")
    if(TRACE_PROFILE)
      link_directories("${SCALOPUS_PATH}/scalopus/so/scalopus_catapult")
      link_directories("${SCALOPUS_PATH}/scalopus/so/scalopus_general")
      link_directories("${SCALOPUS_PATH}/scalopus/so/scalopus_interface/")
      link_directories("${SCALOPUS_PATH}/scalopus/so/scalopus_transport/")
      link_directories("${SCALOPUS_PATH}/scalopus/so/scalopus_tracing")
      link_directories("${SCALOPUS_PATH}/scalopus/so/thirdparty/seasocks/src/main/c/")
    endif()
    xpu_library(
      ps_gpu_wrapper
      SRCS ps_gpu_wrapper.kps ps_gpu_wrapper.cc
      DEPS heter_ps gloo_wrapper ${BRPC_DEPS})
    if(TRACE_PROFILE)
      target_link_libraries(ps_gpu_wrapper scalopus_catapult scalopus_tracing_native
                          scalopus_tracing_nop scalopus_general_consumer scalopus_general
                          scalopus_consumer scalopus_lttng_babeltrace scalopus_tracepoint_native
                          scalopus_tracepoint_nop scalopus_scope_tracing scalopus_transport
                          scalopus_interface seasocks scalopus_tracing_consumer)
    endif()
    add_subdirectory(heter_ps)
  elseif(WITH_RCCL)
    hip_library(
      ps_gpu_wrapper
      SRCS ps_gpu_wrapper.cu ps_gpu_wrapper.cc
      DEPS heter_ps gloo_wrapper ${BRPC_DEPS})
    add_subdirectory(heter_ps)
  endif()
else()
  cc_library(
    ps_gpu_wrapper
    SRCS ps_gpu_wrapper.cc
    DEPS gloo_wrapper)
endif()

if(WITH_NCCL OR WITH_RCCL)
  cc_library(
    nccl_wrapper
    SRCS nccl_wrapper.cc
    DEPS framework_proto variable_helper scope)
endif()
if(WITH_BOX_PS)
  if(WITH_GPU)
    nv_library(
      box_wrapper
      SRCS box_wrapper.cc box_wrapper.cu box_wrapper_impl.cc metrics.cc metrics.cu
      DEPS framework_proto lod_tensor box_ps)
  endif()
  if(WITH_ROCM)
    hip_library(
      box_wrapper
      SRCS box_wrapper.cc box_wrapper.cu box_wrapper_impl.cc
      DEPS framework_proto lod_tensor box_ps)
  endif()
  if(WITH_XPU)
    xpu_library(
   	   box_wrapper
      SRCS box_wrapper.cc box_wrapper_kernel.kps box_wrapper_impl.cc metrics.cc metrics.cu
      DEPS framework_proto lod_tensor box_ps)
  endif()
else()
  cc_library(
    box_wrapper
    SRCS box_wrapper.cc
    DEPS framework_proto lod_tensor)
endif()

if(WITH_GLOO)
  cc_library(
    gloo_wrapper
    SRCS gloo_wrapper.cc
    DEPS framework_proto variable_helper scope gloo)
  cc_library(
    metrics
    SRCS metrics.cc metrics.cu
    DEPS gloo_wrapper)
else()
  cc_library(
    gloo_wrapper
    SRCS gloo_wrapper.cc
    DEPS framework_proto variable_helper scope)
  cc_library(
    metrics
    SRCS metrics.cc metrics.cu
    DEPS gloo_wrapper)
endif()

if(WITH_PSLIB)
  set(DISTRIBUTE_COMPILE_FLAGS
      "-Wno-non-virtual-dtor -Wno-error=non-virtual-dtor -Wno-error=delete-non-virtual-dtor"
  )
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(DISTRIBUTE_COMPILE_FLAGS "${DISTRIBUTE_COMPILE_FLAGS} -faligned-new")
  endif()
  set_source_files_properties(
    heter_wrapper.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
endif()

cc_library(
  heter_wrapper
  SRCS heter_wrapper.cc
  DEPS framework_proto device_context heter_service_proto ${BRPC_DEPS})

cc_test(
  test_fleet_cc
  SRCS test_fleet.cc
  DEPS fleet_wrapper gloo_wrapper fs shell)

if(WITH_ASCEND OR WITH_ASCEND_CL)
  cc_library(
    ascend_wrapper
    SRCS ascend_wrapper.cc
    DEPS framework_proto lod_tensor ascend_ge ascend_graph)
endif()
