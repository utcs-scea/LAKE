#!/bin/bash

make
sudo insmod hack_mllb.ko
sudo rmmod hack_mllb
