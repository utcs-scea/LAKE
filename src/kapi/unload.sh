#!/bin/bash

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi


sudo pkill -2 lake_uspace
sudo rmmod lake_kapi
sudo rmmod lake_shm
