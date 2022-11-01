#!/bin/bash
#original is 77us arrival
#2x 38
#3x is 25us
#4x is 19

mkdir -p azure3x
python3 gen.py azure3x/azure1.trace 0.25 10 25/64 17/64 25
python3 gen.py azure3x/azure2.trace 0.25 10 25/64 17/64 25
python3 gen.py azure3x/azure3.trace 0.25 10 25/64 17/64 25