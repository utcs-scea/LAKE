#!/bin/bash

. `dirname $0`/../../scripts/environment

cd $KAVA_KBUILD_DIR
make -j8
