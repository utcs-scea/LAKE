// SPDX-License-Identifier: GPL-2.0
/* Copyright (c) 2021 Facebook */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>

extern __u64 bpf_kfunc_call_test1(struct sock *sk, __u32 a, __u64 b,
				  __u32 c, __u64 d) __ksym;

extern __u64 bpf_kfunc_call_test4(__u32 a, __u64 b,
				  __u32 c, __u64 d) __ksym;

SEC("tc")
int kfunc_call_test1(struct __sk_buff *skb)
{
	struct bpf_sock *sk = skb->sk;
	__u64 a = 1ULL << 32;
	__u32 ret;

	if (!sk)
		return -1;

	sk = bpf_sk_fullsock(sk);
	if (!sk)
		return -1;

	a = bpf_kfunc_call_test1((struct sock *)sk, 1, a | 2, 3, a | 4);
	ret = a >> 32;   /* ret should be 2 */
	ret += (__u32)a; /* ret should be 12 */

	return ret;
}

SEC("tc")
int kfunc_call_test2(struct __sk_buff *skb)
{
    __u64 a = 1ULL << 32;
	return bpf_kfunc_call_test4(1, a | 2, 3, a | 4);
}

struct syscalls_enter_open_args {
	unsigned long long unused;
	long syscall_nr;
	long filename_ptr;
	long flags;
	long mode;
};

SEC("tracepoint/syscalls/sys_enter_open")
int trace_enter_open(struct syscalls_enter_open_args *ctx) {
	__u64 a;
	__u64 a1=0, a2=1, a3=2, a4=3;
   	a = bpf_kfunc_call_test4(a1, a2, a3, a4);
   	bpf_printk("bpf_kfunc_call_test4:  %d.\n", a);
	return 0;
}

SEC("kprobe/__x64_sys_write")
int bpf_prog3(struct pt_regs *ctx)
{
	__u64 a;
	__u64 a1=0, a2=1, a3=2, a4=3;
   	a = bpf_kfunc_call_test4(a1, a2, a3, a4);
   	bpf_printk("bpf_kfunc_call_test4:  %d.\n", a);
	return 0;
}


char _license[] SEC("license") = "GPL";
