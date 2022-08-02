#ifdef LAKE_ECRYPTFS

#include "lake.h"

ssize_t lake_ecryptfs_file_write(struct file *file, const char __user *data,
            size_t size, loff_t *poffset) 
{
    return 0;
}

int lake_ecryptfs_write(struct inode *inode, char *data, loff_t offset, size_t size)
{
    return 0;
}

int lake_ecryptfs_encrypt_pages(struct page **pgs, unsigned int nr_pages)
{
    return 0;
}

ssize_t lake_ecryptfs_read_update_atime(struct kiocb *iocb, struct iov_iter *to)
{
    return 0;
}

ssize_t lake_ecryptfs_file_read_iter(struct kiocb *iocb, struct iov_iter *iter)
{
    return 0;
}

ssize_t lake_ecryptfs_file_buffered_read(struct kiocb *iocb, 
            struct iov_iter *iter, ssize_t written)
{
    return 0;
}

int lake_ecryptfs_decrypt_pages(struct page **pgs, unsigned int nr_pages)
{
    return 0;
}

int lake_ecryptfs_writepages(struct address_space *mapping,
			       struct writeback_control *wbc)
{
    return 0;
}

int lake_ecryptfs_mmap_encrypt_pages(struct page **pgs, unsigned int nr_pages)
{
    return 0;
}

int lake_ecryptfs_mmap_readpages(struct file *filp, struct address_space *mapping,
			      struct list_head *pages, unsigned nr_pages)
{
    return 0;
}


#endif