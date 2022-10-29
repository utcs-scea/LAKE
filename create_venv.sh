#!/bin/bash

python3 -m venv lakevenv

source lakevenv/bin/activate
pip3 install --upgrade pip
pip3 install numpy
pip3 install tensorflow
pip3 install keras
pip3 install pandas
pip3 install scikit-learn
pip3 install matplotlib
pip3 install psutil
