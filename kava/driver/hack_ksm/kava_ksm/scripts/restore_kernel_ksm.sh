#!/bin/bash

. `dirname $0`/../../scripts/environment

if [[ -d "$KAVA_KSM_ROOT/backup" ]]
then
    cp $KAVA_KSM_ROOT/backup/ksm.c $KAVA_KERNEL_DIR/mm
    cp $KAVA_KSM_ROOT/backup/Makefile $KAVA_KERNEL_DIR/mm
fi
