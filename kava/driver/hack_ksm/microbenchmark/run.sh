#!/bin/bash

make
sudo insmod ksm_kern.ko cubin_path=$(readlink -f ./xxhash.cubin)
sudo rmmod ksm_kern
