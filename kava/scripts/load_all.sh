#!/bin/bash

ROOT=$(cd $(dirname "$0") && pwd -P)/..

# Trap Ctrl-C
trap ctrl_c INT
function ctrl_c() {
    $ROOT/scripts/unload_all.sh
}

cd ${ROOT}
make clean
make
cd ${ROOT}/klib/shared_mem
echo "Installing kshm..."
sudo insmod kshm.ko shm_size=64
cd ${ROOT}/klib/cuda
echo "Installing klib..."
#sudo insmod kcuda.ko chan_mode=file_poll
sudo insmod kcuda.ko chan_mode=netlink_socket
echo "Klib is installed"

cd ${ROOT}/worker/cuda
make clean; make G=1 R=1
echo "Starting worker..."
#sudo ./worker cuda cuda_file_poll
#sudo taskset --cpu-list 2 ./worker cuda cuda_nl_socket
sudo ./worker cuda cuda_nl_socket

cd ${ROOT}/scripts
