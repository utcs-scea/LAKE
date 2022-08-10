#!/bin/bash

sudo mount -t ecryptfs -o ecryptfs_cipher_mode=lake_gcm,no_sig_cache,verbose,ecryptfs_cipher=aes,ecryptfs_key_bytes=32,ecryptfs_passthrough=n,ecryptfs_enable_filename_crypto=n ~/crypt/lake_enc ~/crypt/lake_plain