#!/bin/bash

ROOT=$(cd $(dirname "$0") && pwd -P)/..

# Trap Ctrl-C
trap ctrl_c INT
function ctrl_c() {
    $ROOT/scripts/unload_all.sh
}

cd ${ROOT}
make
cd ${ROOT}/klib/shared_mem
echo "Installing kshm..."
sudo insmod kshm.ko shm_size=64
cd ${ROOT}/klib/mvnc
echo "Installing klib..."
#sudo insmod kmvnc.ko chan_mode=file_poll
sudo insmod kmvnc.ko chan_mode=netlink_socket
echo "Klib is installed"

cd ${ROOT}/worker/mvnc
make clean; make G=1 R=1
echo "Starting worker..."
sudo ./worker mvnc mvnc_nl_socket
#sudo ./worker mvnc mvnc_file_poll

cd ${ROOT}/scripts
