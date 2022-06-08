#!/bin/bash

make

sudo rmmod kleio_kern >/dev/null 2>&1
sudo insmod kleio_kern.ko \
    model_name="/home/hfingler/hf-HACK/kava/driver/kleio/original_code/coeus-sim-master/lstm_page_539"
sudo rmmod kleio_kern

