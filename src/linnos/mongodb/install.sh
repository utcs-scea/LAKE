#!/bin/bash

set -e 
set -o pipefail

mongo=$(ls -la | grep mongod)
if [ "$mongo" != "" ];then
    echo "Mongo binary is already here! Quitting.."
    exit 0
fi

ldssl=$(ldconfig -p | grep libssl.so.1.1)
if [ "$ldssl" == "" ];then
   echo "libssl-1.1 not found."
   read -p "Press enter to download, compile and install (you will need sudo)"
    wget https://www.openssl.org/source/openssl-1.1.1o.tar.gz
    tar -zxvf openssl-1.1.1o.tar.gz
    pushd openssl-1.1.1o
    ./config
    make
    make test
    sudo make install
    popd
    rm -r openssl-1.1.1o openssl-1.1.1o.tar.gz
    echo "Installed!"
fi

wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu2004-6.0.2.tgz
tar xf mongodb-linux-x86_64-ubuntu2004-6.0.2.tgz
cp mongodb-linux-x86_64-ubuntu2004-6.0.2/bin/* .
rm mongodb-linux-x86_64-ubuntu2004-6.0.2.tgz
rm -r mongodb-linux-x86_64-ubuntu2004-6.0.2


#install YCSB dependencies
#clone YCSB 
#TODO: https://github.com/mongodb-labs/YCSB/archive/refs/heads/master.zip
#put this thing in there