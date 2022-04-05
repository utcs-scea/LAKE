#!/bin/bash

sudo modprobe -r ecryptfs || sudo rmmod ecryptfs
sudo rmmod kava_ecb || sudo modprobe -r kava_ecb
#sudo modprobe -r aesni_intel
