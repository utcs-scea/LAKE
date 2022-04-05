#!/bin/bash

make
sudo insmod hello_kern.ko cubin_path=$(readlink -f ./hello.cubin) sequential=0 
sudo rmmod hello_kern
