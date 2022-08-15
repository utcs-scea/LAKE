#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENVF=${SCRIPT_DIR}/test.env
if [ ! -f "$ENVF" ]; then
    echo "Can't find test.env. Create it with path to mount dir"
    exit 1
fi

MNT_DIR=$(head -n 1 ${ENVF})
echo "Running at mount: " $MNT_DIR


sudo mount -t ecryptfs -o ecryptfs_cipher_mode=gcm,no_sig_cache,verbose,ecryptfs_cipher=aes,ecryptfs_key_bytes=32,ecryptfs_passthrough=n,ecryptfs_enable_filename_crypto=n $MNT_DIR/gcm_enc $MNT_DIR/gcm_plain