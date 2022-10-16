#!/bin/bash

set -e 
set -o pipefail

echo "Forcing compile of cubin.."

pushd ..
make -B -f Makefile.cubin
popd


echo $(readlink -e ../linnos.cubin)

#sudo insmod linnos_hook.ko predictor_str=gpu  cubin_path=$(readlink -e ../linnos.cubin)
