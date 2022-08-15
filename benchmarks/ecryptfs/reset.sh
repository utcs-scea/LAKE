#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENVF=${SCRIPT_DIR}/test.env
if [ ! -f $ENVF ]; then
    echo "Can't find test.env. Create it with path to mount dir"
    exit 1
fi

MNT_DIR=$(head -n 1 ${ENVF})
echo "Running at mount: " $MNT_DIR


# umount ~/crypt/cbc_plain
# umount ~/crypt/gcm_plain
# umount ~/crypt/lake_plain

# rm -rf ~/crypt