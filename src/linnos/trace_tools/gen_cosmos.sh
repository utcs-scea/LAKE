#!/bin/bash

#1x 400
#2x 200
#4x 100

mkdir -p cosmos0.5
python3 gen.py cosmos0.5/cosmos1.trace 0.23 10 430/7000 107/32000 300
python3 gen.py cosmos0.5/cosmos2.trace 0.23 10 430/7000 107/32000 300
python3 gen.py cosmos0.5/cosmos3.trace 0.23 10 430/7000 107/32000 300