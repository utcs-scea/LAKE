#!/bin/bash
set -o pipefail

# #check sudo
# if [ "$EUID" -ne 0 ]
#   then echo "Please run as root"
#   exit
# fi

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
pushd src/hello_driver
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
popd #back to root

#Unload everything to make sure
echo " > Trying to unload previously set up API"
echo " > You might see errors. Ignore them"
pushd src/kapi
sudo pkill -2 lake_uspace
sudo rmmod lake_kapi
sudo rmmod lake_shm
echo " > All unloaded. Loading them now."
sleep 1

echo " > Loading shared memory module"
pushd kshm
sudo insmod lake_shm.ko shm_size=32
popd 
echo " > Done."

echo " > Loading kernel API remoting module"
pushd kernel
sudo insmod lake_kapi.ko
popd
echo " > Done."

echo " > unning user space daemon"
pushd uspace
sudo taskset 0x15 ./lake_uspace &
popd 
echo " > Done. Waiting.." 
sleep 2

echo " > Checking if the user space daemon is running..."
load_status=$(ps -ef | grep lake_uspace | wc -l)
if [ $load_status -lt 2 ]; then
    echo "Error: load failed..."
    exit
fi
echo " > Looks like it is."
echo " > If you see netlink errors, press ctrl+c and start again. This means the"
echo " > user space application cannot communicate with the shared memory module."
echo " > If you still can't run when trying this script again, run lsmod and check if the lake_* modules are loaded,"
echo " > if they are and are in use by one more modules, you need to restart your machine."
echo " > Make sure you added cma=128M@0-4G to your kernel parameters."

#back to root
popd 

#Run Hello world
pushd src/hello_driver
echo " > **************************************************"
echo " > **************************************************"
echo " > Running hello world kernel module that uses CUDA."
./run.sh
echo " > Success! run dmesg if you want to see the output"
echo " > **************************************************"
echo " > **************************************************"
#back to root
popd

#Unload
echo " > Unloading everything..."
sleep 2
sudo pkill -2 lake_uspace
sleep 2
sudo rmmod lake_kapi
sleep 2
sudo rmmod lake_shm
echo " > Finished!"