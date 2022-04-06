#!/bin/bash

make -f Makefile_cu
sudo -E /usr/local/cuda/bin/nvprof --devices 0 --analysis-metrics -f -o ecb_profile.bin ./ecb_cuda 128
