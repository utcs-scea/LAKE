#!/bin/bash

#1x 400
#2x 200
#4x 100

mkdir -p cosmos
python3 gen.py cosmos/cosmos1.trace 0.23 10 430/7000 107/32000 400
python3 gen.py cosmos/cosmos2.trace 0.23 10 430/7000 107/32000 400
python3 gen.py cosmos/cosmos3.trace 0.23 10 430/7000 107/32000 400