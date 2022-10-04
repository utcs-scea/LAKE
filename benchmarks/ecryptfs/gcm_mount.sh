#!/bin/bash

sudo mount -t ecryptfs -o key=passphrase:passphrase_passwd=111,ecryptfs_cipher_mode=gcm,no_sig_cache,ecryptfs_cipher=aes,ecryptfs_key_bytes=32,ecryptfs_passthrough=n,ecryptfs_enable_filename_crypto=n /home/hfingler/crypt/lake_enc /home/hfingler/crypt/lake_plain