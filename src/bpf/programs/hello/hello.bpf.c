// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/* Copyright (c) 2020 Facebook */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
//#include <bpf/bpf_core_read.h>
//#include <bpf/bpf_tracing.h>
//#include <linux/bpf.h>

//int my_pid = 0;

extern __u64 bpf_kfunc_call_test1(struct sock *sk, __u32 a, __u64 b,
				  __u32 c, __u64 d) __ksym;

// SEC("tp/raw_syscalls/sys_enter")
// int handle_tp(void *ctx)
// {
// 	__u64 a;
//    	a = bpf_kfunc_call_test1(0, 1, 2, 3, 4);
//    	bpf_printk("bpf_kfunc_call_test1:  %d.\n", a);

// 	return 0;
// }

SEC("classifier")
int kfunc_call_test1(struct __sk_buff *skb)
{
	//struct sock *sk = 0;
  	struct bpf_sock *sk = skb->sk;
	sk = bpf_sk_fullsock(sk);
	__u64 a;
  	a = bpf_kfunc_call_test1(sk, 1, 2, 3, 4);
  	bpf_printk("bpf_kfunc_call_test1:  %d.\n", a);
  	return a;
}


char _license[] SEC("license") = "GPL";