#!/bin/bash

make
sudo insmod hackcbc_kern.ko cubin_path=$(readlink -f ./ecb.cubin)
sudo rmmod hackcbc_kern
