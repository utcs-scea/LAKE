#!/bin/bash

make
sudo insmod ghost_buster.ko cubin_path=$(readlink -f ./knncuda.cubin)
sudo rmmod ghost_buster
