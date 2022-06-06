#!/bin/bash

. `dirname $0`/../../scripts/environment

if ! [[ -d $KAVA_KSM_ROOT/backup ]]; then
    mkdir -p $KAVA_KSM_ROOT/backup

    # Save old kernel files just in case we ruin everything
    cp $KAVA_KERNEL_DIR/mm/ksm.c $KAVA_KSM_ROOT/backup
    cp $KAVA_KERNEL_DIR/mm/Makefile $KAVA_KSM_ROOT/backup
fi

# Move new ksm files to kbuild dir
cp $KAVA_KSM_ROOT/mm/ksm.c $KAVA_KERNEL_DIR/mm
cp $KAVA_KSM_ROOT/mm/Makefile $KAVA_KERNEL_DIR/mm

# Copy xxhash files over for baseline
cp $KAVA_KSM_ROOT/xxhash_kernel/xxhash.c $KAVA_KERNEL_DIR/lib
cp $KAVA_KSM_ROOT/xxhash_kernel/xxhash.h $KAVA_KERNEL_DIR/include/linux

# Copy KAVA symbols over
# rm -f $KAVA_KBUILD_DIR/mm/Module.symvers
# ln -s $KAVA_ROOT/Module.symvers $KAVA_KBUILD_DIR/mm
