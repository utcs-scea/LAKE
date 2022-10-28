#!/bin/bash
#original is 77us arrival
#2x 38
#3x is 25us
#4x is 19

mkdir -p azure4x
python3 gen.py azure4x/azure1.trace 0.25 10 25/64 17/64 19
python3 gen.py azure4x/azure2.trace 0.25 10 25/64 17/64 19
python3 gen.py azure4x/azure3.trace 0.25 10 25/64 17/64 19