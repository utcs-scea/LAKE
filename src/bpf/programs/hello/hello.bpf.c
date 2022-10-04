// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/* Copyright (c) 2020 Facebook */
//#include <linux/bpf.h>
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

extern __u64 bpf_kfunc_call_test1(struct sock *sk, __u32 a, __u64 b,
				  __u32 c, __u64 d) __ksym;

SEC("tp/syscalls/sys_enter_write")
int handle_tp(void *ctx)
{
	__u64 a;

	bpf_printk("BPF triggered.\n");
	a = bpf_kfunc_call_test1(0, 1, 2, 3, 4);
	bpf_printk(" ~~~ bpf_kfunc_call_test:  %d.\n", a);

	return 0;
}
