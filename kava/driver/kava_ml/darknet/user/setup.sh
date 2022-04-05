#/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

cd $SCRIPTPATH
if [[ ! -d "darknet" ]]; then
    git clone https://github.com/pjreddie/darknet.git

    # Patch to enable GPU
    gpu_opt="GPU=1"
    sed -i '1s/GPU=0/$gpu_opt/' $SCRIPTPATH/darknet/Makefile
    sed -i 's/deprecated?/deprecated?\nARCH= -gencode arch=compute_60,code=[sm_60,sm_61]/' \
        $SCRIPTPATH/darknet/Makefile
fi

cd darknet
make
