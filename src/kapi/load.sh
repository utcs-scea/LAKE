#!/bin/bash

ROOT=$(cd $(dirname "$0") && pwd -P)

# Trap Ctrl-C
trap ctrl_c INT
function ctrl_c() {
    $ROOT/unload.sh
}

cd ${ROOT}/kshm
sudo insmod lake_shm.ko shm_size=32

echo ${ROOT}/kernel
cd ${ROOT}/kernel
sudo insmod lake_kapi.ko

cd ${ROOT}/uspace
sudo ./lake_uspace

cd ${ROOT}
