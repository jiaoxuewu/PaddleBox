#!/bin/bash

set -e
set -x

XRE_URL=$1
XRE_DIR_NAME=$2

XDNN_URL=$3
XDNN_DIR_NAME=$4

XCCL_URL=$5
XCCL_DIR_NAME=$6

XCTR_URL=$7
XCTR_DIR_NAME=$8

XHPC_URL=$9
XHPC_DIR_NAME=${10}

wget --no-check-certificate ${XRE_URL} -c -q -O xre.tar.gz
tar xvf xre.tar.gz

wget --no-check-certificate ${XDNN_URL} -c -q -O xdnn.tar.gz
tar xvf xdnn.tar.gz

wget --no-check-certificate ${XCCL_URL} -c -q -O xccl.tar.gz
tar xvf xccl.tar.gz

wget --no-check-certificate ${XCTR_URL} -c -q -O xctr.tar.gz
tar xvf xctr.tar.gz

wget --no-check-certificate ${XHPC_URL} -c -q -O xhpc.tar.gz
tar xvf xhpc.tar.gz

mkdir -p xpu/include/xhpc/xblas
mkdir -p xpu/include/xhpc/xfa
mkdir -p xpu/include/xpu
mkdir -p xpu/lib

cp -r $XRE_DIR_NAME/include/xpu/* xpu/include/xpu/
cp -r $XRE_DIR_NAME/so/libxpurt* xpu/lib/
cp -r $XRE_DIR_NAME/so/libcuda* xpu/lib/

# cp -r $XDNN_DIR_NAME/include/xpu/* xpu/include/xpu/
# cp -r $XDNN_DIR_NAME/so/libxpuapi.so xpu/lib/
cp -r $XCCL_DIR_NAME/include/* xpu/include/xpu/
cp -r $XCCL_DIR_NAME/so/* xpu/lib/
cp -r $XCTR_DIR_NAME/include/* xpu/include/
cp -r $XCTR_DIR_NAME/so/*.so xpu/lib/

cp -r ${XHPC_DIR_NAME}/xblas/include/* xpu/include/xhpc/xblas
cp -r ${XHPC_DIR_NAME}/xblas/so/libxpu_blas.so xpu/lib/
cp -r ${XHPC_DIR_NAME}/xdnn/include/* xpu/include/
cp -r ${XHPC_DIR_NAME}/xdnn/so/libxpuapi.so xpu/lib
cp -r ${XHPC_DIR_NAME}/xfa/include/* xpu/include/xhpc/xfa
cp -r ${XHPC_DIR_NAME}/xfa/so/libxpu_flash_attention.so xpu/lib/