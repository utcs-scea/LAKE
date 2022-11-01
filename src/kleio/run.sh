#!/bin/bash

make
sudo insmod kleio.ko
sudo rmmod kleio
