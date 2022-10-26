#!/bin/bash
#original is 77us arrival
#3x is 25us


mkdir -p azure
python3 gen.py azure/azure1.trace 0.25 4 25 17 25
python3 gen.py azure/azure2.trace 0.25 4 25 17 25
python3 gen.py azure/azure3.trace 0.25 4 25 17 25