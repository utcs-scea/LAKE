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
#include <asm/tlbflush.h>

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
	unsigned long ino = mapping->host->i_ino;
 	u64 block_in_file;

	char comm[30];
    bpf_get_current_comm(&comm, 30);

	if (comm[0] != 't' || comm[1] != 'o') {
		return 0;
	}

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

int ext4_file_read(struct pt_regs *ctx, struct kiocb *iocb, struct iov_iter *to)
//int ext4_file_read(struct pt_regs *ctx)
{
	char comm[30];
    bpf_get_current_comm(&comm, 30);

	//if( __builtin_memcmp(iocb->ki_filp->f_path.dentry->d_iname, touchblk, 5) == 0 ) {
	//if( iocb->ki_filp->f_path.dentry->d_iname[0] == 'd' && iocb->ki_filp->f_path.dentry->d_iname[1] == 'u')
	
	if (comm[0] == 't' && comm[1] == 'o') 
	{
		bpf_trace_printk("=> name: %s  offset: %lu  \\n", iocb->ki_filp->f_path.dentry->d_iname, iocb->ki_pos);
	}
	return 0;
}

int ext4_readpage(struct pt_regs *ctx, struct file *file, struct page *page)
//int ext4_file_read(struct pt_regs *ctx)
{
	char comm[30];
    bpf_get_current_comm(&comm, 30);
	
	if (comm[0] == 't' && comm[1] == 'o') 
	{
		bpf_trace_printk("=> readpage:  name: %s  \\n", file->f_path.dentry->d_iname);
	}
	return 0;
}

int ext4_readpages(struct pt_regs *ctx, struct file *file, struct address_space *mapping,
		struct list_head *pages, unsigned nr_pages)
{
	char comm[30];
    bpf_get_current_comm(&comm, 30);
	
	if (comm[0] == 't' && comm[1] == 'o') 
	{
		bpf_trace_printk("=> readpageS:  name: %s  \\n", file->f_path.dentry->d_iname);
	}
	return 0;
}

int ext4_pf(struct pt_regs *ctx, struct vm_fault *vmf)
{
	char comm[30];
    bpf_get_current_comm(&comm, 30);
	
	if (comm[0] == 't' && comm[1] == 'o') 
	{
		bpf_trace_printk("=> ext4_pf: %s  VA: %lu   offset %lu \\n", vmf->vma->vm_file->f_path.dentry->d_iname, vmf->address, vmf->pgoff);
	}
	return 0;
}

"""

b = BPF(text=bpf_text)

b.attach_kprobe(event="ext4_mpage_readpages", fn_name="fblktrace_read_pages");
#b.attach_kretprobe(event="ext4_file_open",  fn_name="fblktrace_ext4_file_open");

#b.attach_kprobe(event="ext4_file_read_iter", fn_name="ext4_file_read")
#b.attach_kprobe(event="ext4_readpage", fn_name="ext4_readpage")
#b.attach_kprobe(event="ext4_readpages", fn_name="ext4_readpages")

b.attach_kprobe(event="ext4_filemap_fault", fn_name="ext4_pf")



print ('printing...')
while True:
    b.trace_print()
