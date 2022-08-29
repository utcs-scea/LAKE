#!/bin/bash

if [ "$EUID" -ne 0 ]
  then echo "Please run as root"
  exit
fi


sudo pkill -2 lake_uspace
sudo rmmod lake_kshm
sudo rmmod lake_kapi
