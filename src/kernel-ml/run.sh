#!/bin/bash

make
sudo insmod kml.ko cubin_path=$(readlink -f ./kml.cubin)
sudo rmmod kml
