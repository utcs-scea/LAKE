#!/bin/bash

#1x 416
#2x 208

mkdir -p bing_i
python3 gen.py bing_i/bing_i1.trace 0.23 5 55/512 30/1024 208
python3 gen.py bing_i/bing_i2.trace 0.23 5 55/512 30/1024 208
python3 gen.py bing_i/bing_i3.trace 0.23 5 55/512 30/1024 208