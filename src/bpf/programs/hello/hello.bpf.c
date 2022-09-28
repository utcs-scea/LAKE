// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/* Copyright (c) 2020 Facebook */
#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include "bpf_helpers.h"

char LICENSE[] SEC("license") = "Dual BSD/GPL";

int my_pid = 0;

extern __u64 bpf_kfunc_call_test1(struct sock *sk, __u32 a, __u64 b,
				  __u32 c, __u64 d) __ksym;

SEC("tp/raw_syscalls/sys_enter")
int handle_tp(void *ctx)
{
	int pid = bpf_get_current_pid_tgid() >> 32;
	int r;

	if (pid != my_pid)
		return 0;

	bpf_printk("BPF triggered from PID %d.\n", pid);

	r = (__u32)bpf_kfunc_call_test1(0, 1, 2, 3, 4);
	bpf_printk("bpf_kfunc_call_test1:  %d.\n", r);

	return 0;
}
