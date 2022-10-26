#!/bin/bash

#1x 400
#2x 200

mkdir -p cosmos
python3 gen.py cosmos/cosmosxx1.trace 0.23 5 430/7000 107/32000 200
python3 gen.py cosmos/cosmosxx2.trace 0.23 5 430/7000 107/32000 300
python3 gen.py cosmos/cosmosxx3.trace 0.23 5 430/7000 107/32000 200