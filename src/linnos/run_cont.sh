#!/bin/bash

sudo insmod linnos_cont.ko cubin_path=$(readlink -f ./linnos.cubin) runtime_s=10
sudo rmmod linnos_cont