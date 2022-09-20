#!/bin/bash

sudo insmod mllb_kern.ko cubin_path=$(readlink -f ./mllb.cubin)
sudo rmmod mllb_kern
