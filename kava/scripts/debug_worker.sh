#!/bin/bash

ROOT=$(cd $(dirname "$0") && pwd -P)/..

cd ${ROOT}
make clean; make
cd ${ROOT}/klib/shared_mem
echo "Installing kshm..."
sudo insmod kshm.ko shm_size=64
cd ${ROOT}/klib/cuda
echo "Installing klib..."
sudo insmod kcuda.ko
echo "Klib is installed"

cd ${ROOT}/worker/cuda
make clean; make G=1
echo "Starting worker..."
sudo -E gdb --args ./worker cuda cuda_file_poll

cd ${ROOT}/scripts
