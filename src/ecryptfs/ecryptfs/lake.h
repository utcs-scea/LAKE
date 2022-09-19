#ifndef LAKE_ECRYPTFS_KERNEL_H
#define LAKE_ECRYPTFS_KERNEL_H

#ifdef LAKE_ECRYPTFS

#include "ecryptfs_kernel.h"

//fns that were static
int ecryptfs_readpage(struct file *file, struct page *page);
int fill_zeros_to_end_of_page(struct page *page, unsigned int to);
int ecryptfs_write_begin(struct file *file,
			struct address_space *mapping,
			loff_t pos, unsigned len, unsigned flags,
			struct page **pagep, void **fsdata);


ssize_t lake_generic_file_write_iter(struct kiocb *iocb, struct iov_iter *from);
int lake_ecryptfs_encrypt_pages(struct page **pgs, unsigned int nr_pages);

ssize_t lake_ecryptfs_read_update_atime(struct kiocb *iocb, struct iov_iter *to);
ssize_t lake_ecryptfs_file_read_iter(struct kiocb *iocb, struct iov_iter *iter);
ssize_t lake_ecryptfs_file_buffered_read(struct kiocb *iocb, 
            struct iov_iter *iter, ssize_t written);
int lake_ecryptfs_decrypt_pages(struct page **pgs, unsigned int nr_pages);

//int lake_ecryptfs_mmap_writepages(struct address_space *mapping,
//			       struct writeback_control *wbc);

int lake_ecryptfs_mmap_readpages(struct file *filp, struct address_space *mapping,
			      struct list_head *pages, unsigned nr_pages);



#endif
#endif