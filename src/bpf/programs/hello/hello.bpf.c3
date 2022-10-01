/* Copyright (c) 2013-2015 PLUMgrid, http://plumgrid.com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of version 2 of the GNU General Public
 * License as published by the Free Software Foundation.
 */
//#include <linux/skbuff.h>
//#include <linux/netdevice.h>
//#include <linux/version.h>
//#include <uapi/linux/bpf.h>
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, long);
	__type(value, long);
	__uint(max_entries, 1024);
} my_map SEC(".maps");

/* kprobe is NOT a stable ABI. If kernel internals change this bpf+kprobe
 * example will no longer be meaningful
 */
// SEC("kprobe/kfree_skb")
// int bpf_prog2(struct pt_regs *ctx)
// {
// 	long loc = 0;
// 	long init_val = 1;
// 	long *value;

// 	/* read ip of kfree_skb caller.
// 	 * non-portable version of __builtin_return_address(0)
// 	 */
// 	BPF_KPROBE_READ_RET_IP(loc, ctx);

// 	value = bpf_map_lookup_elem(&my_map, &loc);
// 	if (value)
// 		*value += 1;
// 	else
// 		bpf_map_update_elem(&my_map, &loc, &init_val, BPF_ANY);
// 	return 0;
// }

// static unsigned int log2(unsigned int v)
// {
// 	unsigned int r;
// 	unsigned int shift;

// 	r = (v > 0xFFFF) << 4; v >>= r;
// 	shift = (v > 0xFF) << 3; v >>= shift; r |= shift;
// 	shift = (v > 0xF) << 2; v >>= shift; r |= shift;
// 	shift = (v > 0x3) << 1; v >>= shift; r |= shift;
// 	r |= (v >> 1);
// 	return r;
// }

// static unsigned int log2l(unsigned long v)
// {
// 	unsigned int hi = v >> 32;
// 	if (hi)
// 		return log2(hi) + 32;
// 	else
// 		return log2(v);
// }

// struct hist_key {
// 	char comm[16];
// 	u64 pid_tgid;
// 	u64 uid_gid;
// 	u64 index;
// };

// struct {
// 	__uint(type, BPF_MAP_TYPE_PERCPU_HASH);
// 	__uint(key_size, sizeof(struct hist_key));
// 	__uint(value_size, sizeof(long));
// 	__uint(max_entries, 1024);
// } my_hist_map SEC(".maps");

extern __u64 bpf_kfunc_call_test4(__u32 a, __u64 b,
				  __u32 c, __u64 d) __ksym;

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

