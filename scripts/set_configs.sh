#!/bin/bash

scripts/config --enable CONFIG_BPF
scripts/config --enable CONFIG_BPF_SYSCALL
scripts/config --enable CONFIG_HAVE_EBPF_JIT
scripts/config --enable CONFIG_BPF_JIT
scripts/config --enable CONFIG_BPF_JIT_ALWAYS_ON
scripts/config --enable CONFIG_DEBUG_INFO_BTF
scripts/config --enable CONFIG_DEBUG_INFO_BTF_MODULES

scripts/config --enable CONFIG_CGROUPS
scripts/config --enable CONFIG_CGROUP_BPF
scripts/config --enable CONFIG_CGROUP_NET_CLASSID
scripts/config --enable CONFIG_SOCK_CGROUP_DATA

scripts/config --enable CONFIG_BPF_EVENTS
scripts/config --enable CONFIG_KPROBE_EVENTS
scripts/config --enable CONFIG_UPROBE_EVENTS
scripts/config --enable CONFIG_TRACING
scripts/config --enable CONFIG_FTRACE_SYSCALLS
scripts/config --enable CONFIG_FUNCTION_ERROR_INJECTION
scripts/config --enable CONFIG_BPF_KPROBE_OVERRIDE

scripts/config --enable CONFIG_NET
scripts/config --enable CONFIG_XDP_SOCKETS
scripts/config --enable CONFIG_LWTUNNEL_BPF
scripts/config --enable CONFIG_NET_ACT_BPF
scripts/config --enable CONFIG_NET_CLS_BPF
scripts/config --enable CONFIG_NET_CLS_ACT
scripts/config --enable CONFIG_NET_SCH_INGRESS
scripts/config --enable CONFIG_XFRM
scripts/config --enable CONFIG_IP_ROUTE_CLASSID
scripts/config --enable CONFIG_IPV6_SEG6_BPF
scripts/config --enable CONFIG_BPF_LIRC_MODE2
scripts/config --enable CONFIG_BPF_STREAM_PARSER
scripts/config --enable CONFIG_NETFILTER_XT_MATCH_BPF
scripts/config --enable CONFIG_BPFILTER
scripts/config --enable CONFIG_BPFILTER_UMH

scripts/config --enable CONFIG_TEST_BPF
#scripts/config --enable CONFIG_HZ

#scripts/config --enable CONFIG_IKHEADERS
#scripts/config --enable CONFIG_KPROBES
#scripts/config --enable CONFIG_UPROBES
#scripts/config --enable CONFIG_DEBUG_FS
#scripts/config --enable CONFIG_FTRACE

scripts/config --enable CONFIG_CMA
scripts/config --enable CONFIG_DMA_CMA
scripts/config --set-str LOCALVERSION "-lake"

scripts/config --module CONFIG_ECRYPT_FS
scripts/config --enable CONFIG_ECRYPT_FS_MESSAGING
scripts/config --module CONFIG_CRYPTO_AES_NI_INTEL

scripts/config --disable SYSTEM_TRUSTED_KEYS
scripts/config --disable SYSTEM_REVOCATION_KEYS

make olddefconfig