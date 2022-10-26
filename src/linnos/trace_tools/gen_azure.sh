#!/bin/bash
#original is 77us arrival
#2x 38
#3x is 25us
#5x is 15

mkdir -p azure
python3 gen.py azure/azure1.trace 0.25 5 25/16/64/12 17/8/64/8 38
python3 gen.py azure/azure2.trace 0.25 5 25/16/64/12 17/8/64/8 38
python3 gen.py azure/azure3.trace 0.25 5 25/16/64/12 17/8/64/8 38