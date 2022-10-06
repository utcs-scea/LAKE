#!/bin/bash

sudo insmod knn.ko cubin_path=$(readlink -f ./knncuda.cubin)
sudo rmmod knn
