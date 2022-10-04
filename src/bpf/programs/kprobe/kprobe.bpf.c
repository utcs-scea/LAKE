// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause
/* Copyright (c) 2021 Sartura */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

// extern __u64 bpf_get_task_pid(void) __ksym;
// extern __u64 bpf_put_pid(void) __ksym;

extern __u64 bpf_kfunc_call_test0(void) __ksym;

SEC("kprobe/do_unlinkat")
int BPF_KPROBE(do_unlinkat, int dfd, struct filename *name)
{
	__u64 a = bpf_kfunc_call_test0();
	bpf_printk(" ~~~ kfunc:  %d.\n", a);
	return 0;
}

struct syscalls_enter_open_args {
	unsigned long long unused;
	long syscall_nr;
	long filename_ptr;
	long flags;
	long mode;
};

SEC("tracepoint/syscalls/sys_enter_open")
int trace_enter_open(struct syscalls_enter_open_args *ctx)
{
	__u64 a = bpf_kfunc_call_test0();
	bpf_printk(" >> trace_enter_open  %d\n", a);
	return 0;
}


	// pid_t pid;
	// const char *filename;
	// __u64 a;
	// pid = bpf_get_current_pid_tgid() >> 32;
	// filename = BPF_CORE_READ(name, name);
	// bpf_printk("KPROBE ENTRY pid = %d, filename = %s\n", pid, filename);
	// //a = bpf_kfunc_call_test1(0, 1, 2, 3, 4);

// SEC("kretprobe/do_unlinkat")
// int BPF_KRETPROBE(do_unlinkat_exit, long ret)
// {
// 	pid_t pid;
// 	__u64 a;

// 	pid = bpf_get_current_pid_tgid() >> 32;
// 	bpf_printk("KPROBE EXIT: pid = %d, ret = %ld\n", pid, ret);

// 	//a = bpf_kfunc_call_test1(0, 1, 2, 3, 4);
// 	//bpf_printk(" ~~~ bpf_kfunc_call_test:  %d.\n", a);

// 	return 0;
// }
