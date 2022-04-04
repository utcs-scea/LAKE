#!/bin/bash

if [ ! -d "./iozone3_489" ]; then
    tar xf iozone3_489.tar
fi
cd iozone3_489/src/current
make linux

if [ "$1" != "" ]; then
    FILENAME="-f $1"
else
    FILENAME=""
fi

./iozone -Ra -n 128M -g 512M -y 4K -q 4M -e -j 4 -i 0 -i 1 -i 2 $FILENAME
