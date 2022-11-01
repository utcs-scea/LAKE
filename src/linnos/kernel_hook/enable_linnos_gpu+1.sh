#!/bin/bash

set -e 
set -o pipefail

# pushd ..
# make -B -f Makefile_cubin
# popd

#TODO: make sure cubin exists
CUBINPATH=../linnos.cubin

if [ ! -f "$CUBINPATH" ]; then
    echo "$CUBINPATH does not exists, go one dir up and run make -B -f Makefile_cubin"
    exit 1
fi


sudo insmod linnos_hook.ko predictor_str=gpu  cubin_path=$(readlink -e ../linnos.cubin) model_size=1
