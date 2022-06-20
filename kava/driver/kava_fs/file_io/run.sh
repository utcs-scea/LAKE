#!/bin/bash

DIR=$1
if [ ! -d "$1" ]; then
    echo "Directory \"$1\" does not exist. Switch to default: ~/crypt/secret."
    DIR="~/crypt/secret"
fi
sudo ./fs_bench $DIR 1 8K 4K
