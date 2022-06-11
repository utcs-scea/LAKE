#!/usr/bin/python
# @lint-avoid-python-3-compatibility-imports
#
# fblktrace.py  Trace page cache misses
#
# USAGE: fblktrace.py
#
# Copyright 2018 Collabora Ltd.
# Licensed under the Apache License, Version 2.0 (the "License")
#
# Author: Gabriel Krisman Bertazi  -  2018-09-20

from bcc import BPF

bpf_text = """
#include <uapi/linux/ptrace.h>
#include <linux/fs.h>
#include <linux/kernel.h>

#define PAGE_SHIFT 12

#define fblktrace_container_of(ptr, type, member) ({			\
	const typeof(((type *)0)->member) * __mptr = (ptr);		\
	(type *)((char *)__mptr - offsetof(type, member)); })

int fblktrace_ext4_file_open(struct pt_regs *ctx, struct inode * inode,
				    struct file * filp)
{
	char lname[400];
	unsigned long ino = inode->i_ino;
	struct qstr qs = {};

	qs = filp->f_path.dentry->d_name;
	if (qs.len >= 400-1)
		qs.len = 399;

	bpf_probe_read(lname, 20, (void*)qs.name);

	bpf_trace_printk("=> Open inode %ld: fname = %s\\n", ino, lname);
	return 0;
}

int fblktrace_read_pages(struct pt_regs *ctx, struct address_space *mapping,
			 struct list_head *pages, struct page *page,
			 unsigned nr_pages, bool is_readahead)
{
	int i;
	u64 index;
	unsigned blkbits = mapping->host->i_blkbits;
	unsigned long ino = mapping->host->i_ino;;
 	u64 block_in_file;

	#pragma unroll
	for (i = 0; i < 32 && nr_pages--; i++) {
		if (pages) {
			pages = pages->prev;
			page = fblktrace_container_of(pages, struct page, lru);
		}
		index = page->index;
		block_in_file = (unsigned long) index << (12 - blkbits);
		if (!is_readahead)
			bpf_trace_printk("=> inode: %ld: FSBLK=%lu BSIZ=%lu\\n",
					 ino, index, 1<<blkbits);
		else
			bpf_trace_printk("=> inode: %ld: FSBLK=%lu BSIZ=%lu [RA]\\n",
					 ino, index, 1<<blkbits);
	}
	return 0;
}

"""

b = BPF(text=bpf_text)

b.attach_kprobe(event="ext4_mpage_readpages", fn_name="fblktrace_read_pages");
b.attach_kretprobe(event="ext4_file_open",  fn_name="fblktrace_ext4_file_open");
print ('printing...')
while True:
    b.trace_print();
