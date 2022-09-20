#!/bin/bash

sudo insmod hello_kern.ko cubin_path=$(readlink -f ./hello.cubin)
sudo rmmod hello_kern
