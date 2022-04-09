#!/bin/bash

make
sudo insmod linnos_kern.ko cubin_path=$(readlink -f ./linnos.cubin)
sudo rmmod linnos_kern
