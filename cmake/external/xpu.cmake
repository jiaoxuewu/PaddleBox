if(NOT WITH_XPU)
  return()
endif()

include(ExternalProject)
set(XPU_PROJECT "extern_xpu")
set(XPU_API_LIB_NAME "libxpuapi.so")
set(XPU_API_PLUGIN_NAME  "libxpuplugin.so")
set(XPU_RT_LIB_NAME "libxpurt.so")
if(WITH_XPU_XRE5)
  set(XPU_RT_ALIAS_LIB_NAME "libxpurt.so.2")
else()
  set(XPU_RT_ALIAS_LIB_NAME "libxpurt.so.1")
endif()
set(XPU_ML_LIB_NAME "libxpuml.so")
set(XPU_ML_ALIAS_LIB_NAME "libxpuml.so.1")

if(NOT DEFINED XPU_BASE_URL)
  set(XPU_BASE_URL_WITHOUT_DATE
      "https://baidu-kunlun-product.cdn.bcebos.com/KL-SDK/klsdk-dev")
  set(XPU_BASE_URL "${XPU_BASE_URL_WITHOUT_DATE}/20221130")
else()
  set(XPU_BASE_URL "${XPU_BASE_URL}")
endif()

# ubuntu and centos: use output by XDNN API team
if(NOT DEFINED XPU_XDNN_BASE_URL)
  set(XPU_XDNN_BASE_URL_WITHOUT_DATE
      "https://klx-sdk-release-public.su.bcebos.com/xdnn/dev")
  set(XPU_XDNN_BASE_URL "${XPU_XDNN_BASE_URL_WITHOUT_DATE}/20230613")
else()
  set(XPU_XDNN_BASE_URL "${XPU_XDNN_BASE_URL}")
endif()

set(XPU_XCCL_BASE_URL
    "https://klx-sdk-release-public.su.bcebos.com/xccl/release/1.0.0")

set(XPU_XCTR_DIR_NAME "xctr_x86_64")

if(WITH_AARCH64)
  set(XPU_XRE_DIR_NAME "xre-kylin_aarch64")
  set(XPU_XDNN_DIR_NAME "xdnn-kylin_aarch64")
  set(XPU_XCCL_DIR_NAME "xccl-kylin_aarch64")
  set(XPU_XDNN_URL
      "${XPU_XDNN_BASE_URL}/${XPU_XDNN_DIR_NAME}.tar.gz"
      CACHE STRING "" FORCE)
elseif(WITH_SUNWAY)
  set(XPU_XRE_DIR_NAME "xre-deepin_sw6_64")
  set(XPU_XDNN_DIR_NAME "xdnn-deepin_sw6_64")
  set(XPU_XCCL_DIR_NAME "xccl-deepin_sw6_64")
  set(XPU_XDNN_URL
      "${XPU_BASE_URL}/${XPU_XDNN_DIR_NAME}.tar.gz"
      CACHE STRING "" FORCE)
elseif(WITH_BDCENTOS)
  set(XPU_XRE_DIR_NAME "xre-bdcentos_x86_64")
  set(XPU_XDNN_DIR_NAME "xdnn-bdcentos_x86_64")
  set(XPU_XCCL_DIR_NAME "xccl-bdcentos_x86_64")
  # ubuntu and centos: use output by XDNN API team
  set(XPU_XDNN_URL
      "${XPU_XDNN_BASE_URL}/${XPU_XDNN_DIR_NAME}.tar.gz"
      CACHE STRING "" FORCE)
elseif(WITH_UBUNTU)
  set(XPU_XRE_DIR_NAME "xre-ubuntu_x86_64")
  set(XPU_XDNN_DIR_NAME "xdnn-ubuntu_x86_64")
  set(XPU_XCCL_DIR_NAME "xccl-bdcentos_x86_64")
  # ubuntu and centos: use output by XDNN API team
  set(XPU_XDNN_URL
      "${XPU_XDNN_BASE_URL}/${XPU_XDNN_DIR_NAME}.tar.gz"
      CACHE STRING "" FORCE)
elseif(WITH_CENTOS)
  set(XPU_XRE_DIR_NAME "xre-centos7_x86_64")
  set(XPU_XDNN_DIR_NAME "xdnn-bdcentos_x86_64")
  set(XPU_XCCL_DIR_NAME "xccl-bdcentos_x86_64")
  # ubuntu and centos: use output by XDNN API team
  set(XPU_XDNN_URL
      "${XPU_XDNN_BASE_URL}/${XPU_XDNN_DIR_NAME}.tar.gz"
      CACHE STRING "" FORCE)
else()
  set(XPU_XRE_DIR_NAME "xre-ubuntu_x86_64")
  set(XPU_XDNN_DIR_NAME "xdnn-ubuntu_x86_64")
  set(XPU_XCCL_DIR_NAME "xccl-bdcentos_x86_64")
  # default: use output by XDNN API team
  set(XPU_XDNN_URL
      "${XPU_XDNN_BASE_URL}/${XPU_XDNN_DIR_NAME}.tar.gz"
      CACHE STRING "" FORCE)
endif()

set(XPU_XRE_URL
    "${XPU_BASE_URL}/${XPU_XRE_DIR_NAME}.tar.gz"
    CACHE STRING "" FORCE)
set(XPU_XCCL_URL
    "${XPU_XCCL_BASE_URL}/${XPU_XCCL_DIR_NAME}.tar.gz"
    CACHE STRING "" FORCE)
#https://klx-sdk-release-public.su.bcebos.com/xdnn_train/dev/20240312/xctr.tar.gz
#https://klx-sdk-release-public.su.bcebos.com/xdnn_train/dev/20221130/xctr.tar.gz
set(XPU_XCTR_URL
    https://klx-sdk-release-public.su.bcebos.com/xdnn_train/dev/20240701/xctr.tar.gz
    CACHE STRING "" FORCE)

#"data-im.baidu.com:/home/work/var/CI_DATA/im/static/pack_paddle_depence.sh/pack_paddle_depence.sh"
set(XPU_PACK_DEPENCE_URL
    "https://baidu-kunlun-public.su.bcebos.com/paddle_depence/pack_paddle_box_depence_v2.sh"
    CACHE STRING "" FORCE)

if(WITH_XPU_XRE5)
  add_definitions(-DPADDLE_WITH_XPU_XRE5)
  set(XPU_XRE_BASE_VERSION "5.0.9.1")
  set(XPU_XCCL_BASE_VERSION "1.2.8.2")
  set(XPU_XRE_BASE_URL
      "https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/${XPU_XRE_BASE_VERSION}"
  )
  set(XPU_XRE_DIR_NAME "xre-bdcentos-x86_64-${XPU_XRE_BASE_VERSION}")
else()
  set(XPU_XRE_BASE_VERSION "4.0.28.1")
  set(XPU_XCCL_BASE_VERSION "1.2.0.5")
  set(XPU_XRE_BASE_URL
      "https://klx-sdk-release-public.su.bcebos.com/xre/release/${XPU_XRE_BASE_VERSION}"
  )
  set(XPU_XRE_DIR_NAME "xre-bdcentos_x86_64")
endif()

set(XPU_XTDK_DIR_NAME "xtdk-llvm15-bdcentos7_x86_64")

# if(WITH_BOX_PS)
if (WITH_BOX_PS OR WITH_XPU_KP)
    set(XPU_XDNN_DIR_NAME "xdnn-bdcentos_x86_64")
    set(XPU_XCCL_DIR_NAME "xccl_rdma-bdcentos_x86_64")
    set(XPU_XRE_URL
        "${XPU_XRE_BASE_URL}/${XPU_XRE_DIR_NAME}.tar.gz"
        CACHE STRING "" FORCE)
    set(XPU_XCCL_URL
        "https://klx-sdk-release-public.su.bcebos.com/xccl/release/${XPU_XCCL_BASE_VERSION}/${XPU_XCCL_DIR_NAME}.tar.gz"
        CACHE STRING "" FORCE)
    #"https://klx-sdk-release-public.su.bcebos.com/xdnn/release/2.6.0.1/${XPU_XDNN_DIR_NAME}.tar.gz"
    set(XPU_XDNN_URL
        "https://klx-sdk-release-public.su.bcebos.com/xdnn_train/dev/paddlebox/xdnn_master/20240731/${XPU_XDNN_DIR_NAME}.tar.gz"
        CACHE STRING "" FORCE)
    set(SCALOPUS_URL
        "https://klx-sdk-release-public.su.bcebos.com/xdnn_train/dev/paddlebox/20230306/scalopus.tar.gz"
        CACHE STRING "" FORCE)
    set(XPU_XTDK_URL
        "https://klx-sdk-release-public.su.bcebos.com/xtdk_llvm15/release/3.0.0.1/${XPU_XTDK_DIR_NAME}.tar.gz"
        CACHE STRING "" FORCE)
endif()

set(SNAPPY_PREFIX_DIR "${THIRD_PARTY_PATH}/xpu")
set(XPU_DOWNLOAD_DIR "${SNAPPY_PREFIX_DIR}/src/${XPU_PROJECT}")
set(XPU_INSTALL_DIR "${THIRD_PARTY_PATH}/install/xpu")
set(XPU_INC_DIR "${THIRD_PARTY_PATH}/install/xpu/include")
set(XPU_LIB_DIR "${THIRD_PARTY_PATH}/install/xpu/lib")

set(XPU_API_LIB "${XPU_LIB_DIR}/${XPU_API_LIB_NAME}")
set(XPU_API_PLUGIN "${XPU_LIB_DIR}/${XPU_API_PLUGIN_NAME}")
set(XPU_RT_LIB "${XPU_LIB_DIR}/${XPU_RT_LIB_NAME}")
set(XPU_RT_ALIAS_LIB "${XPU_LIB_DIR}/${XPU_RT_ALIAS_LIB_NAME}")
set(XPU_ML_LIB "${THIRD_PARTY_PATH}/xpu/src/extern_xpu/${XPU_XRE_DIR_NAME}/so/${XPU_ML_LIB_NAME}")
set(XPU_ML_ALIAS_LIB "${THIRD_PARTY_PATH}/xpu/src/extern_xpu/${XPU_XRE_DIR_NAME}/so/${XPU_ML_ALIAS_LIB_NAME}")

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}" "${XPU_INSTALL_DIR}/lib")

file(
  WRITE ${XPU_DOWNLOAD_DIR}/CMakeLists.txt
  "PROJECT(XPU)\n" "cmake_minimum_required(VERSION 3.0)\n"
  "install(DIRECTORY xpu/include xpu/lib \n"
  "        DESTINATION ${XPU_INSTALL_DIR})\n")

ExternalProject_Add(
  ${XPU_PROJECT}
  ${EXTERNAL_PROJECT_LOG_ARGS}
  PREFIX ${SNAPPY_PREFIX_DIR}
  DOWNLOAD_DIR ${XPU_DOWNLOAD_DIR}
  DOWNLOAD_COMMAND
    wget --no-check-certificate ${XPU_PACK_DEPENCE_URL} -O pack_paddle_box_depence.sh && bash pack_paddle_box_depence.sh ${XPU_XRE_URL}
    ${XPU_XRE_DIR_NAME} ${XPU_XDNN_URL} ${XPU_XDNN_DIR_NAME} ${XPU_XCCL_URL}
    ${XPU_XCCL_DIR_NAME} ${XPU_XCTR_URL} ${XPU_XCTR_DIR_NAME} &&
    wget --no-check-certificate ${SCALOPUS_URL} && tar zxvf scalopus.tar.gz &&
    wget --no-check-certificate ${XPU_XTDK_URL} && tar zxvf ${XPU_XTDK_DIR_NAME}.tar.gz
  DOWNLOAD_NO_PROGRESS 1
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${XPU_INSTALL_ROOT}
  CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${XPU_INSTALL_ROOT}
  BUILD_BYPRODUCTS ${XPU_API_LIB}
  BUILD_BYPRODUCTS ${XPU_API_PLUGIN}
  BUILD_BYPRODUCTS ${XPU_RT_LIB})

include_directories(${XPU_INC_DIR})
add_library(shared_xpuapi SHARED IMPORTED GLOBAL)
set_property(TARGET shared_xpuapi PROPERTY IMPORTED_LOCATION "${XPU_API_LIB}")
set_property(TARGET shared_xpuapi PROPERTY IMPORTED_LOCATION "${XPU_API_PLUGIN}")

# generate a static dummy target to track xpulib dependencies
# for cc_library(xxx SRCS xxx.c DEPS xpulib)
generate_dummy_static_lib(LIB_NAME "xpulib" GENERATOR "xpu.cmake")

if(WITH_XPU_BKCL)
  message(STATUS "Compile with XPU BKCL!")
  add_definitions(-DPADDLE_WITH_XPU_BKCL)

  set(XPU_BKCL_LIB_NAME "libbkcl.so")
  set(XPU_BKCL_LIB "${XPU_LIB_DIR}/${XPU_BKCL_LIB_NAME}")
  set(XPU_BKCL_INC_DIR "${THIRD_PARTY_PATH}/install/xpu/include")
  include_directories(${XPU_BKCL_INC_DIR})
  target_link_libraries(xpulib -Wl,--push-state,--no-as-needed ${XPU_API_LIB} ${XPU_API_PLUGIN} ${XPU_RT_LIB} ${XPU_BKCL_LIB} ${XPU_ML_LIB} -Wl,--pop-state)
else()
  target_link_libraries(xpulib -Wl,--push-state,--no-as-needed ${XPU_API_LIB} ${XPU_API_PLUGIN} ${XPU_RT_LIB} ${XPU_ML_LIB} -Wl,--pop-state)
endif()

add_dependencies(xpulib ${XPU_PROJECT})

# Ensure that xpu/api.h can be included without dependency errors.
file(
  GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc
  CONTENT "")
add_library(xpu_headers_dummy STATIC
            ${CMAKE_CURRENT_BINARY_DIR}/.xpu_headers_dummy.cc)
add_dependencies(xpu_headers_dummy extern_xpu)
link_libraries(xpu_headers_dummy)
