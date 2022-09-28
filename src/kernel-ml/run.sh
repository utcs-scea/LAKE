#!/bin/bash

make
sudo insmod kml_kern.ko cubin_path=$(readlink -f ./kml.cubin)
sudo rmmod kml_kern
