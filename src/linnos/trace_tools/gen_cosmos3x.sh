#!/bin/bash

#1x 400
#2x 200
#3x 133
#4x 100

mkdir -p cosmos3x
python3 gen.py cosmos3x/cosmos1.trace 0.23 10 430/7000 107/32000 134
python3 gen.py cosmos3x/cosmos2.trace 0.23 10 430/7000 107/32000 134
python3 gen.py cosmos3x/cosmos3.trace 0.23 10 430/7000 107/32000 134