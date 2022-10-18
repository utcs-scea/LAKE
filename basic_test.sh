#!/bin/bash
set -o pipefail

#check sudo
if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi

#Check if the kernel version is correct
kernel_version=$(uname -r)
echo $kernel_version
if [ "$kernel_version" != "6.0.0-lake" ]; then
  echo "Error : Required Kernel Version not found"
  exit
fi

#Check if the nvidia driver is installed
nvidia_check=$(nvidia-smi | grep 'Driver Version')
if [ "$nvidia_check" == "" ]; then
  echo "Error : Driver not found"
  exit
fi

#Compile kapi
make clean -C src/kapi/uspace/
make -C src/kapi/uspace/
exit_code=$?
if [ $exit_code != 0 ]; then
    echo "Error: Make failed exiting..."
    exit
fi

make clean -C src/kapi/kshm/
make -C src/kapi/kshm/
exit_code=$?
if [ $exit_code != 0 ]; then
    echo "Error: Make failed exiting..."
    exit
fi

make clean -C src/kapi/kernel/
make -C src/kapi/kernel/
exit_code=$?
if [ $exit_code != 0 ]; then
    echo "Error: Make failed exiting..."
    exit
fi

# #Compile hello_driver
cd src/hello_driver
make -f Makefile_cubin
if [ $exit_code != 0 ]; then
    echo "Error: Make failed exiting..."
    exit
fi

make
if [ $exit_code != 0 ]; then
    echo "Error: Make failed exiting..."
    exit
fi
cd ../../

# #load kapi
cd src/kapi
ROOT=$(cd $(dirname "$0") && pwd -P)

cd ${ROOT}/kshm
sudo insmod lake_shm.ko shm_size=80

cd ${ROOT}/kernel
sudo insmod lake_kapi.ko

cd ${ROOT}/uspace
sudo taskset 0x15 ./lake_uspace &

sleep 5
load_status=$(ps -ef | grep lake_uspace | wc -l)
if [ $load_status -lt 2 ]; then
    echo "Error: load failed..."
    exit
fi


cd ../../../

#Run Hello world
cd src/hello_driver
echo "Running hello world:"
./run.sh
echo "Success"

cd ../../

#Unload
cd src/kapi
ROOT=$(cd $(dirname "$0") && pwd -P)
sudo pkill -2 lake_uspace
sudo rmmod lake_kapi
sudo rmmod lake_shm