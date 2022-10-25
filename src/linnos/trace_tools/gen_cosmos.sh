#!/bin/bash

#python3 gen.py cosmos/co1.trace 0.22 7 430 107 138

mkdir -p cosmos
python3 gen.py cosmos/co1.trace 0.22 5 430 107 34
python3 gen.py cosmos/co2.trace 0.22 5 430 107 34
python3 gen.py cosmos/co3.trace 0.22 5 430 107 34