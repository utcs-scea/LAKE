#!/bin/bash

# to launch, go to the kleio/ dir and run
# docker run -v $(pwd):/hack -it --gpus 1 nvidia/cuda:11.4.0-devel-ubuntu20.04
# then run this script from the container

apt update
apt install build-essential git wget
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xf eigen-3.4.0.tar.gz
rm eigen-3.4.0.tar.gz
mv eigen-3.4.0 /usr/include/eigen3
make -C haste/ examples