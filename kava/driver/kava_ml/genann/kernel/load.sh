#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

$SCRIPTPATH/setup.sh

# TODO: Select module to insmod with switch case.
case $1 in
ex1)
    sudo rmmod ex2_kern >/dev/null 2>&1
    sudo insmod ex1_kern.ko
    sudo rmmod ex1_kern
    ;;
ex2)
    sudo rmmod ex2_kern >/dev/null 2>&1
    sudo insmod ex2_kern.ko
    sudo rmmod ex2_kern
    ;;
ex3)
    sudo rmmod ex3_kern >/dev/null 2>&1
    sudo insmod ex3_kern.ko save_name="$SCRIPTPATH/../user/genann/cpu/example/xor.ann"
    sudo rmmod ex3_kern
    ;;
ex4)
    sudo rmmod ex4_kern >/dev/null 2>&1
    sudo insmod $SCRIPTPATH/ex4_kern.ko iris_data="$SCRIPTPATH/../user/genann/cpu/example/iris.data"
    sudo rmmod ex4_kern
    ;;
mnist)
    sudo rmmod mnist_kern >/dev/null 2>&1
    sudo insmod mnist_kern.ko \
        imageFileName="$SCRIPTPATH/../user/genann/cpu/mnist/train-images.idx3-ubyte" \
        labelFileName="$SCRIPTPATH/../user/genann/cpu/mnist/train-labels.idx1-ubyte"
    sudo rmmod mnist_kern
esac
