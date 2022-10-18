#!/bin/bash
set -o pipefail

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

#Compile hello_driver
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

#./load.sh