#!/bin/bash

(cd ../crypto_ecb ; \
 make clean; make && make -f Makefile_cu && \
 sudo insmod kava_ecb.ko split_threshold=2048 aesni_fraction=75)
make && sudo insmod ecryptfs.ko
