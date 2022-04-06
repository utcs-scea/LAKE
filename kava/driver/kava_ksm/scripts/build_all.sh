#!/bin/bash

. `dirname $0`/environment

echo "Building xxhash library..."
cd $SCRIPTPATH/../kava_ksm/cu/xxhash/cpu
make

echo "Building xxhash cubin..."
cd $SCRIPTPATH/../kava_ksm/cu/xxhash/gpu
make

echo "Building jhash cubin..."
cd $SCRIPTPATH/../kava_ksm/cu/jhash2
make

echo "Building samepage generator..."
cd $SCRIPTPATH/../samepage_generator
make

echo "Building ksm_start module... (Make sure KAVA_ROOT has been changed in Makefile)"
cd $SCRIPTPATH/../kava_ksm_start
make
