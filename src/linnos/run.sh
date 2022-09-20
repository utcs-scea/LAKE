#!/bin/bash

make
sudo insmod linnos.ko cubin_path=$(readlink -f ./linnos.cubin)
sudo rmmod linnos
