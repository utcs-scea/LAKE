#!/bin/bash

make
sudo insmod router_kern.ko runtime=10 cubin_path=$(readlink -f ../user/firewall.cubin) batch=8192 \
    sequential=1 input_throughput=0 block_size=32 numrules=100
sudo rmmod router_kern
