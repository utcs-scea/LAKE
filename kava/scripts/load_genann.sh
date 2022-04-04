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
sudo insmod kshm.ko shm_size=128
cd ${ROOT}/klib/genann
echo "Installing genann..."
sudo insmod kgenann.ko chan_mode=netlink_socket
echo "Klib is installed"

cd ${ROOT}/worker/genann
make clean; make ${1} R=1
echo "Starting worker..."
sudo ./worker genann genann_nl_socket

cd ${ROOT}/scripts
