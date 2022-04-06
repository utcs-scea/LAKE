#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo $SCRIPTPATH

export KAVA_ROOT=$SCRIPTPATH/../../../../../

(cd $SCRIPTPATH; make -j6)
