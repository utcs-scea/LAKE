#!/bin/bash

. `dirname $0`/../../scripts/environment

cd $KAVA_KBUILD_DIR
INSTALL_MOD_STRIP=1 sudo make modules_install -j8 && sudo make install -j8
