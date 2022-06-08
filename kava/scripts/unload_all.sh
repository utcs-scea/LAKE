#!/bin/bash

ROOT=$(cd $(dirname "$0") && pwd -P)/..

echo "Unloading KAvA modules..."
sudo rmmod demo_drv
sudo pkill -2 worker
sudo pkill -2 gdb
sudo ${ROOT}/worker/kavactrl
sudo rmmod kcuda
sudo rmmod kgenann
sudo rmmod kmvnc
sudo rmmod kshm
sudo rmmod lstm_tf
sudo rmmod kshm

cd ${ROOT}/scripts
