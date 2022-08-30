#!/bin/bash

ROOT=$(cd $(dirname "$0") && pwd -P)

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi


# Trap Ctrl-C
trap ctrl_c INT
function ctrl_c() {
    $ROOT/unload.sh
}

cd ${ROOT}/kshm
sudo insmod lake_shm.ko shm_size=16

cd ${ROOT}/kernel
sudo insmod lake_kapi.ko

cd ${ROOT}/uspace
sudo ./lake_uspace

cd ${ROOT}
