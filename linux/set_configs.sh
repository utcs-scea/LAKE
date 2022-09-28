#!/bin/bash

scripts/config --enable CONFIG_DEBUG_INFO
scripts/config --enable CONFIG_DEBUG_INFO_BTF
scripts/config --enable CONFIG_CMA
scripts/config --enable CONFIG_DMA_CMA
scripts/config --disable SYSTEM_TRUSTED_KEYS
scripts/config --set-str LOCALVERSION "-hack"
scripts/config --enable CONFIG_BPF
scripts/config --enable CONFIG_BPF_SYSCALL
scripts/config --enable CONFIG_BPF_JIT
scripts/config --enable CONFIG_HAVE_EBPF_JIT
scripts/config --enable CONFIG_BPF_EVENTS
scripts/config --enable CONFIG_IKHEADERS
scripts/config --module CONFIG_ECRYPT_FS
scripts/config --enable CONFIG_ECRYPT_FS_MESSAGING
scripts/config --module CONFIG_CRYPTO_AES_NI_INTEL
scripts/config --disable SYSTEM_REVOCATION_KEYS
#scripts/config --enable CONFIG_NUMA_BALANCING
#scripts/config --disable CONFIG_HACK_MLLB
#scripts/config --enable CONFIG_USERFAULTFD

make olddefconfig