#/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

# Clone the required repos
cd $SCRIPTPATH
if [[ ! -d "genann" ]]; then
    git clone https://github.com/yuhc/kava-genann.git genann
fi

if [[ -d "genann" ]]; then
    cd $SCRIPTPATH/genann/cpu
    make    # By default this will execute the test and examples

    mkdir -p $SCRIPTPATH/genann/gpu/build
    cd $SCRIPTPATH/genann/gpu/build
    cmake ..
    make
fi
