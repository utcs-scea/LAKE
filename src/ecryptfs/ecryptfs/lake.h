/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#ifndef LAKE_ECRYPTFS_KERNEL_H
#define LAKE_ECRYPTFS_KERNEL_H

#ifdef LAKE_ECRYPTFS

#include "ecryptfs_kernel.h"

//fns that were static
int ecryptfs_read_folio(struct file *file, struct folio *folio);
int fill_zeros_to_end_of_page(struct page *page, unsigned int to);
int ecryptfs_write_begin(struct file *file,
			struct address_space *mapping,
			loff_t pos, unsigned len,
			struct page **pagep, void **fsdata);

ssize_t lake_generic_file_write_iter(struct kiocb *iocb, struct iov_iter *from);
int lake_ecryptfs_encrypt_pages(struct page **pgs, unsigned int nr_pages);

ssize_t lake_ecryptfs_read_update_atime(struct kiocb *iocb, struct iov_iter *to);
ssize_t lake_ecryptfs_file_read_iter(struct kiocb *iocb, struct iov_iter *iter);
ssize_t lake_ecryptfs_file_buffered_read(struct kiocb *iocb, 
            struct iov_iter *iter, ssize_t written);
int lake_ecryptfs_decrypt_pages(struct page **pgs, unsigned int nr_pages);

void lake_ecryptfs_readahead(struct readahead_control *ractl);





#endif
#endif