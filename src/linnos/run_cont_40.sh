#!/bin/bash

sudo insmod linnos_cont.ko cubin_path=$(readlink -f ./linnos.cubin) runtime_s=40
sudo rmmod linnos_cont
