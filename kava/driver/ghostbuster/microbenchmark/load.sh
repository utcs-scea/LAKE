#!/bin/bash

make

sudo rmmod lstm_kern >/dev/null 2>&1
sudo insmod lstm_kern.ko \
    model_name="/home/hfingler/hf-HACK/kava/worker/lstm_tf/lstm_tf_wrapper/gb_model/"
sudo rmmod lstm_kern

