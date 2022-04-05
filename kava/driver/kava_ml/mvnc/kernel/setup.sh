#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export KAVA_ROOT=$SCRIPTPATH/../../../../../

(cd $SCRIPTPATH; make)
