#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

$SCRIPTPATH/setup.sh

sudo rmmod lstm_kern >/dev/null 2>&1
sudo insmod lstm_kern.ko \
    model_name="/home/edwardhu/kava/worker/lstm_tf/lstm_tf_wrapper/"
sudo rmmod lstm_kern

