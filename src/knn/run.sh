#!/bin/bash
./cuda/cpu > ae_plot/ae_cpu.csv
sudo insmod knn.ko cubin_path=$(readlink -f ./knncuda.cubin)
sudo rmmod knn
