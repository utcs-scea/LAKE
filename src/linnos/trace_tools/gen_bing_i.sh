#!/bin/bash

#1x 416
#2x 208
#4x 104

mkdir -p bing_i4x
python3 gen.py bing_i4x/bing_i1.trace 0.23 10 55/512 30/1024 104
python3 gen.py bing_i4x/bing_i2.trace 0.23 10 55/512 30/1024 104
python3 gen.py bing_i4x/bing_i3.trace 0.23 10 55/512 30/1024 104