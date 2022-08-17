#!/bin/bash

DIR=$1
if [ ! -d "$1" ]; then
    echo "Directory \"$1\" does not exist."
    exit 1
fi
sudo ./fs_bench $DIR 1 32k 4k
