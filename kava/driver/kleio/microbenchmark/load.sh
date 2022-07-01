#!/bin/bash

make

sudo rmmod kleio_kern >/dev/null 2>&1
sudo insmod kleio_kern.ko model_name=$(readlink -f ./lstm_page_539)
sudo rmmod kleio_kern

