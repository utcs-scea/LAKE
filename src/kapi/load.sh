#!/bin/bash

ROOT=$(cd $(dirname "$0") && pwd -P)

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

# Trap Ctrl-C
trap ctrl_c INT
trap ctrl_c SIGINT
function ctrl_c() {
  echo "** Trapped CTRL-C, cleaning up"
  sudo pkill -2 lake_uspace
  sudo rmmod lake_kapi
  sudo rmmod lake_shm
}

cd ${ROOT}/kshm
sudo insmod lake_shm.ko shm_size=120

cd ${ROOT}/kernel
sudo insmod lake_kapi.ko

cd ${ROOT}/uspace
sudo taskset 0x15 ./lake_uspace

cd ${ROOT}