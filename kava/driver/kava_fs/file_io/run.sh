#!/bin/bash

DIR=$1
if [ ! -d "$1" ]; then
    echo "Directory \"$1\" does not exist. Switch to default: ~/crypt/secret."
    DIR="~/crypt/secret"
fi
sudo ./fs_bench $DIR 3 4M 128K
