#!/bin/bash
#76.923

mkdir -p azure
python3 gen.py azure/azure1.trace 0.25 7 256 17 15.3846
python3 gen_poisson.py azure/azure_poisson1.trace 0.25 7 256 17 15.3846

python3 gen.py azure/azure2.trace 0.25 7 256 17 15.3846
python3 gen_poisson.py azure/azure_poisson2.trace 0.25 7 256 17 15.3846

python3 gen.py azure/azure3.trace 0.25 7 256 17 15.3846
python3 gen_poisson.py azure/azure_poisson3.trace 0.25 7 256 17 15.3846