#!/bin/bash

#1x 400
#2x 200

mkdir -p cosmos
python3 gen.py cosmos/cosmos1.trace 0.23 5 430/200/7000/3000 107/64/32000/20000 200
python3 gen.py cosmos/cosmos2.trace 0.23 5 430/200/7000/3000 107/64/32000/20000 200
python3 gen.py cosmos/cosmos3.trace 0.23 5 430/200/7000/3000 107/64/32000/20000 200