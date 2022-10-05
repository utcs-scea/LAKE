#ifdef LAKE_ECRYPTFS

#include "lake.h"
#include "ecryptfs_kernel.h"
#include <linux/sched/signal.h>
#include <linux/random.h>
#include <linux/scatterlist.h>
#include <linux/uio.h>
#include <linux/swap.h>
#include <linux/pagevec.h>
#include <linux/delay.h>
#include <linux/fs.h>

#define DECRYPT		0
#define ENCRYPT		1

struct write_end_args {
	loff_t pos;
	unsigned bytes; 
	unsigned copied;
};

static int lake_write_end(struct file *file,
			struct address_space *mapping,
			loff_t pos, unsigned len, unsigned copied,
			struct page *page, void *fsdata)
{
	struct inode *ecryptfs_inode = mapping->host;

	int rc;
	//ecryptfs_printk(KERN_ERR, "ecryptfs_write_end\n");

	if (pos + copied > i_size_read(ecryptfs_inode)) {
		i_size_write(ecryptfs_inode, pos + copied);
		ecryptfs_printk(KERN_DEBUG, "Expanded file size to "
			"[0x%.16llx]\n",
			(unsigned long long)i_size_read(ecryptfs_inode));
	}
	rc = ecryptfs_write_inode_size_to_metadata(ecryptfs_inode);
	if (rc)
		printk(KERN_ERR "Error writing inode size to metadata; "
		       "rc = [%d]\n", rc);
	else
		rc = copied;

	unlock_page(page);
	put_page(page);
	return rc;
}

static int lake_write_middle(struct file *file,
			struct address_space *mapping,
			loff_t pos, unsigned len, unsigned copied,
			struct page *page, void *fsdata)
{
	pgoff_t index = pos >> PAGE_SHIFT;
	unsigned from = pos & (PAGE_SIZE - 1);
	unsigned to = from + copied;
	struct inode *ecryptfs_inode = mapping->host;
	struct ecryptfs_crypt_stat *crypt_stat =
		&ecryptfs_inode_to_private(ecryptfs_inode)->crypt_stat;
	int rc;
	//ecryptfs_printk(KERN_ERR, "ecryptfs_write_middle\n");

	////ecryptfs_printk(KERN_DEBUG, "Calling fill_zeros_to_end_of_page"
	//		"(page w/ index = [0x%.16lx], to = [%d])\n", index, to);
	if (!(crypt_stat->flags & ECRYPTFS_ENCRYPTED)) {
		rc = ecryptfs_write_lower_page_segment(ecryptfs_inode, page, 0,
						       to);
		if (!rc) {
			rc = copied;
			fsstack_copy_inode_size(ecryptfs_inode,
				ecryptfs_inode_to_lower(ecryptfs_inode));
		}
		goto out;
	}
	if (!PageUptodate(page)) {
		if (copied < PAGE_SIZE) {
			rc = 0;
			goto out;
		}
		SetPageUptodate(page);
	}
	/* Fills in zeros if 'to' goes beyond inode size */
	rc = fill_zeros_to_end_of_page(page, to);
	if (rc) {
		ecryptfs_printk(KERN_WARNING, "Error attempting to fill "
			"zeros in page with index = [0x%.16lx]\n", index);
	}

	// in the original code, we would ecryptfs_encrypt_page now
	// XXX: we need to lie and say that we expanded the file size,
	// otherwise we get a bunch of truncates later
	if (pos + copied > i_size_read(ecryptfs_inode)) {
		i_size_write(ecryptfs_inode, pos + copied);
		ecryptfs_printk(KERN_DEBUG, "Expanded file size to "
			"[0x%.16llx]\n",
			(unsigned long long)i_size_read(ecryptfs_inode));
	}
	rc = ecryptfs_write_inode_size_to_metadata(ecryptfs_inode);
	if (rc)
		printk(KERN_ERR "Error writing inode size to metadata; "
		       "rc = [%d]\n", rc);
	else
		rc = copied;

out:
	return rc;
}

static ssize_t lake_generic_perform_write(struct file *file,
				struct iov_iter *i, loff_t pos)
{
	struct address_space *mapping = file->f_mapping;
	//const struct address_space_operations *a_ops = mapping->a_ops;
	long status = 0;
	ssize_t written = 0;

	//let's try to guess how many IOs we will do
	unsigned int n_ios = DIV_ROUND_UP(iov_iter_count(i), PAGE_SIZE) + 1;
	struct write_end_args *io_args;
	int count = 0, j;
	struct page **pages;

	io_args = kmalloc(n_ios * sizeof(struct write_end_args), GFP_KERNEL);
	pages = kmalloc(n_ios * sizeof(struct page*), GFP_KERNEL);

	do {
		struct page *page;
		unsigned long offset;	/* Offset into pagecache page */
		unsigned long bytes;	/* Bytes to write to page */
		size_t copied;		/* Bytes copied from user */
		void *fsdata;

		offset = (pos & (PAGE_SIZE - 1));
		bytes = min_t(unsigned long, PAGE_SIZE - offset,
						iov_iter_count(i));

again:
		/*
		 * Bring in the user page that we will copy from _first_.
		 * Otherwise there's a nasty deadlock on copying from the
		 * same page as we're writing to, without it being marked
		 * up-to-date.
		 *
		 * Not only is this an optimisation, but it is also required
		 * to check that the address is actually valid, when atomic
		 * usercopies are used, below.
		 */
		//if (unlikely(iov_iter_fault_in_readable(i, bytes))) {
		if (unlikely(fault_in_iov_iter_readable(i, bytes))) {
			status = -EFAULT;
			break;
		}

		if (fatal_signal_pending(current)) {
			status = -EINTR;
			break;
		}
	
		//status = a_ops->write_begin(file, mapping, pos, bytes, flags,
		//ecryptfs_printk(KERN_ERR, "lake calling ecryptfs_write_begin\n");
		status = ecryptfs_write_begin(file, mapping, pos, bytes,
						&page, &fsdata);
		if (unlikely(status < 0))
			break;

		if (mapping_writably_mapped(mapping))
			flush_dcache_page(page);

		//copied = iov_iter_copy_from_user_atomic(page, i, offset, bytes);
		copied = copy_page_from_iter_atomic(page, offset, bytes, i);
		flush_dcache_page(page);

		//status = a_ops->write_end(file, mapping, pos, bytes, copied,
		//				page, fsdata);
		//if (unlikely(status < 0))
		//	break;

		//call middle, then save arguments
		status = lake_write_middle(file, mapping, pos, bytes, copied,
						page, fsdata);
		io_args[count].pos = pos;
		io_args[count].bytes = bytes;
		io_args[count].copied = copied;
		pages[count] = page;
		count++;
		//XXX: this is bad for concurrency and needs to eventually be fixed
		//originally it's begin (lock), encrypt, end (unlock), but since
		//we are batching the encrypt part, we need to unlock here
		//bc the next begin might want to lock the previous, causing
		//a deadlock
		unlock_page(page);

		copied = status;

		cond_resched();

		iov_iter_advance(i, copied);
		if (unlikely(copied == 0)) {
			/*
			 * If we were unable to copy any data at all, we must
			 * fall back to a single segment length write.
			 *
			 * If we didn't fallback here, we could livelock
			 * because not all segments in the iov can be copied at
			 * once without a pagefault.
			 */
			bytes = min_t(unsigned long, PAGE_SIZE - offset,
						iov_iter_single_seg_count(i));
			goto again;
		}
		pos += copied;
		written += copied;

		//balance_dirty_pages_ratelimited(mapping);
	} while (iov_iter_count(i));

	//at this point we still need to encrypt and end IOs
	//this will lock every page again
	lake_ecryptfs_encrypt_pages(pages, count);

	for (j = 0 ; j < count ; j++) {
		//unlocks pages sequentially
		lake_write_end(file, mapping, io_args[j].pos,
				io_args[j].bytes, io_args[j].copied,
				pages[j], 0);
		balance_dirty_pages_ratelimited(mapping);
	}

	kfree(io_args);
	kfree(pages);

	return written ? written : status;
}

// no changes from linux
static ssize_t __lake_generic_file_write_iter(struct kiocb *iocb, struct iov_iter *from)
{
	struct file *file = iocb->ki_filp;
	struct address_space * mapping = file->f_mapping;
	struct inode 	*inode = mapping->host;
	ssize_t		written = 0;
	ssize_t		err;
	//ssize_t		status;

	//ecryptfs_printk(KERN_ERR, "__lake_generic_file_write_iter\n");

	/* We can write back this queue in page reclaim */
	current->backing_dev_info = inode_to_bdi(inode);
	err = file_remove_privs(file);
	if (err)
		goto out;

	err = file_update_time(file);
	if (err)
		goto out;

	if (iocb->ki_flags & IOCB_DIRECT) {
		//ecryptfs_printk(KERN_ERR, "NYI IOCB_DIRECT\n");
		err = -1;
		goto out;
		// loff_t pos, endbyte;

		// written = generic_file_direct_write(iocb, from);
		// /*
		//  * If the write stopped short of completing, fall back to
		//  * buffered writes.  Some filesystems do this for writes to
		//  * holes, for example.  For DAX files, a buffered write will
		//  * not succeed (even if it did, DAX does not handle dirty
		//  * page-cache pages correctly).
		//  */
		// if (written < 0 || !iov_iter_count(from) || IS_DAX(inode))
		// 	goto out;

		// status = generic_perform_write(file, from, pos = iocb->ki_pos);
		// /*
		//  * If generic_perform_write() returned a synchronous error
		//  * then we want to return the number of bytes which were
		//  * direct-written, or the error code if that was zero.  Note
		//  * that this differs from normal direct-io semantics, which
		//  * will return -EFOO even if some bytes were written.
		//  */
		// if (unlikely(status < 0)) {
		// 	err = status;
		// 	goto out;
		// }
		// /*
		//  * We need to ensure that the page cache pages are written to
		//  * disk and invalidated to preserve the expected O_DIRECT
		//  * semantics.
		//  */
		// endbyte = pos + status - 1;
		// err = filemap_write_and_wait_range(mapping, pos, endbyte);
		// if (err == 0) {
		// 	iocb->ki_pos = endbyte + 1;
		// 	written += status;
		// 	invalidate_mapping_pages(mapping,
		// 				 pos >> PAGE_SHIFT,
		// 				 endbyte >> PAGE_SHIFT);
		// } else {
		// 	/*
		// 	 * We don't know how much we wrote, so just return
		// 	 * the number of bytes which were direct-written
		// 	 */
		// }
	} else {
		written = lake_generic_perform_write(file, from, iocb->ki_pos);
		if (likely(written > 0))
			iocb->ki_pos += written;
	}
out:
	current->backing_dev_info = NULL;
	return written ? written : err;
}

// no changes from linux
ssize_t lake_generic_file_write_iter(struct kiocb *iocb, struct iov_iter *from)
{
	struct file *file = iocb->ki_filp;
	struct inode *inode = file->f_mapping->host;
	ssize_t ret;

	//ecryptfs_printk(KERN_ERR, "lake_generic_file_write_iter\n");

	inode_lock(inode);
	ret = generic_write_checks(iocb, from);
	if (ret > 0)
		ret = __lake_generic_file_write_iter(iocb, from);
	inode_unlock(inode);

	if (ret > 0)
		ret = generic_write_sync(iocb, ret);
	return ret;
}

// our own function, sort of related to ecryptfs_encrypt_page and crypt_extent_aead
int lake_ecryptfs_encrypt_pages(struct page **pgs, unsigned int nr_pages)
{
 	struct inode *ecryptfs_inode;
	struct ecryptfs_crypt_stat *crypt_stat;
 	struct ecryptfs_extent_metadata extent_metadata;
 	char *enc_extent_virt;
	struct page *enc_extent_page = NULL;
	loff_t extent_offset = 0;
	loff_t lower_offset;
 	int rc = 0;
	int data_extent_num;
	int num_extents = 1;
	int meta_extent_num;
	int metadata_per_extent;
	u8 *tag_data = NULL;
	u8 *iv_data = NULL;

 	struct scatterlist *src_sg = NULL, *dst_sg = NULL;
 	unsigned int i = 0;
 	u32 sz = 0;
	int cipher_mode_code;

	//ecryptfs_printk(KERN_ERR, "attemping to lock pages\n");
	for (i = 0; i < nr_pages; i++) {
		lock_page(pgs[i]);
	}
	//ecryptfs_printk(KERN_ERR, "locked %d pages\n", nr_pages);

	ecryptfs_inode = pgs[0]->mapping->host;
	crypt_stat =
		&(ecryptfs_inode_to_private(ecryptfs_inode)->crypt_stat);
	BUG_ON(!(crypt_stat->flags & ECRYPTFS_ENCRYPTED));
	
	cipher_mode_code = ecryptfs_code_for_cipher_mode_string(
		crypt_stat->cipher_mode);

	// if using linux GCM, use the original function
	if (cipher_mode_code == ECRYPTFS_CIPHER_MODE_GCM) {
		for (i = 0 ; i < nr_pages ; i++) {
			ecryptfs_encrypt_page(pgs[i]);
		}
		return 0;
	}

	lake_print(KERN_ERR, "[lake] lake_ecryptfs_encrypt_pages %d pages\n", nr_pages);

 	if (!nr_pages || !pgs || !pgs[0]) {
 		goto out;
 	}

	ecryptfs_inode = pgs[0]->mapping->host;
	crypt_stat =
		&(ecryptfs_inode_to_private(ecryptfs_inode)->crypt_stat);
	metadata_per_extent = crypt_stat->extent_size / sizeof(extent_metadata);

	// source sgs
 	src_sg = (struct scatterlist *)kmalloc(
 		nr_pages * sizeof(struct scatterlist), GFP_KERNEL);
 	if (!src_sg) {
 		rc = -ENOMEM;
 		ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for source scatter list\n");
 		goto higher_out;
	}
 
	// dst sgs
    dst_sg = (struct scatterlist *)kmalloc(
 		2 * nr_pages * sizeof(struct scatterlist), GFP_KERNEL);
 	if (!dst_sg) {
 		ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for destination scatter list\n");
 		rc = -ENOMEM;
 		goto higher_out;
    }
 
	//ivs
	iv_data = (u8 *)kmalloc(nr_pages * ECRYPTFS_MAX_IV_BYTES, GFP_KERNEL);
 	if (!iv_data) {
 		ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for ivs\n");
 		rc = -ENOMEM;
 		goto higher_out;
    }

	//tags
	tag_data = (u8 *)kmalloc(nr_pages * ECRYPTFS_GCM_TAG_SIZE, GFP_KERNEL);
 	if (!tag_data) {
 		ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for ivs\n");
 		rc = -ENOMEM;
 		goto higher_out;
    }

	//generate ivs
	for (i = 0; i < nr_pages; i++) {
		get_random_bytes(iv_data+(i*ECRYPTFS_MAX_IV_BYTES), ECRYPTFS_MAX_IV_BYTES);
	}

    sg_init_table(src_sg, nr_pages);
    sg_init_table(dst_sg, nr_pages*2);
 
 	for (i = 0; i < nr_pages; i++) {
 		enc_extent_page = alloc_page(GFP_USER);
 		if (!enc_extent_page) {
 			rc = -ENOMEM;
            ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for encrypted extent\n");
 			for (sz = 0; sz < i; sz++) {
 				__free_page(sg_page(dst_sg + sz));
 			}
 			goto higher_out;
 		}
 
 		sg_set_page(src_sg + i, pgs[i], PAGE_SIZE, 0);
 		sg_set_page(dst_sg + (i*2), enc_extent_page, PAGE_SIZE, 0);
		sg_set_buf(dst_sg + (i*2) + 1, tag_data + (i*ECRYPTFS_GCM_TAG_SIZE), 
				ECRYPTFS_GCM_TAG_SIZE);
 	}
 
 	rc = crypt_scatterlist(crypt_stat, dst_sg, src_sg, PAGE_SIZE * nr_pages,
             iv_data, ENCRYPT);
     if (rc) {
         printk(KERN_ERR "%s: Error encrypting extents in scatter list; "
                 "rc = [%d]\n", __func__, rc);
 	    for (i = 0; i < nr_pages; i++) {
             __free_page(sg_page(dst_sg + i));
         }
         goto higher_out;
	}
 
 	for (i = 0; i < nr_pages; i++) {
		/*
		* Lower offset must take into account the number of
		* data extents, auth tag extents, and header size.
		*/
		lower_offset = ecryptfs_lower_header_size(crypt_stat);
		data_extent_num = (pgs[i]->index * num_extents) + 1;
		data_extent_num += extent_offset;
		lower_offset += (data_extent_num - 1)
			* crypt_stat->extent_size;
		meta_extent_num = (data_extent_num
			+ (metadata_per_extent - 1))
			/ metadata_per_extent;
		lower_offset += meta_extent_num
			* crypt_stat->extent_size;

		// get the page we created previously
		enc_extent_page = sg_page(dst_sg + (i*2));

		enc_extent_virt = kmap(enc_extent_page);
		rc = ecryptfs_write_lower(ecryptfs_inode,
				enc_extent_virt + (extent_offset
					* crypt_stat->extent_size),
				lower_offset,
				crypt_stat->extent_size);
		kunmap(enc_extent_page);
		__free_page(enc_extent_page);
		if (rc < 0) {
			printk(KERN_ERR "Error attempting to write lower"
					"page; rc = [%d]\n", rc);
			goto out;
		}

		memcpy(extent_metadata.iv_bytes, iv_data+(i*ECRYPTFS_MAX_IV_BYTES),
			ECRYPTFS_MAX_IV_BYTES);
		memcpy(extent_metadata.auth_tag_bytes, tag_data+(i*ECRYPTFS_GCM_TAG_SIZE),
			ECRYPTFS_GCM_TAG_SIZE);

		lower_offset = ecryptfs_lower_header_size(crypt_stat);
		lower_offset += (meta_extent_num - 1) *
			(metadata_per_extent + 1) *
			crypt_stat->extent_size;

		rc = ecryptfs_write_lower(ecryptfs_inode,
				(void *) &extent_metadata,
				lower_offset,
				sizeof(extent_metadata));
		if (rc < 0) {
			printk(KERN_ERR "Error attempting to write lower"
					"page; rc = [%d]\n", rc);
			goto out;
		}
 	}
 
	rc = 0;
	// for (i = 0; i < nr_pages; i++) {
	// 	enc_extent_page = sg_page(dst_sg + (i*2));
	// 	if (enc_extent_page) {
	// 		__free_page(enc_extent_page);
	// 	}
	// }

higher_out:
 	kfree(src_sg);
 	kfree(dst_sg);
	kfree(iv_data);
	kfree(tag_data);
out:
	//ecryptfs_printk(KERN_ERR, "[lake] lake_ecryptfs_encrypt_pages done\n");

 	return rc;
}

/**********************************************
 * 
 *   END OF WRITE_ITER
 * 
 *********************************************/

/**********************************************
 * 
 *   START OF READ_ITER
 * 
 *********************************************/

ssize_t lake_ecryptfs_read_update_atime(struct kiocb *iocb, struct iov_iter *to)
{
	ssize_t rc;
	struct path *path;
	struct file *file = iocb->ki_filp;

	lake_print(KERN_ERR, "[lake] start of lake_ecryptfs_read_update_atime\n");
	rc = lake_ecryptfs_file_read_iter(iocb, to);
	if (rc >= 0) {
		path = ecryptfs_dentry_to_lower_path(file->f_path.dentry);
		touch_atime(path);
	}
	return rc;
}

ssize_t lake_ecryptfs_file_read_iter(struct kiocb *iocb, struct iov_iter *iter)
{
	size_t count = iov_iter_count(iter);
	ssize_t retval = 0;

	lake_print(KERN_ERR, "[lake] start of lake_ecryptfs_file_read_iter\n");

    if (!count) {
        goto out; /* skip atime */
    }

	if (iocb->ki_flags & IOCB_DIRECT) {
        pr_err("IOCB not supported\n");
        goto out;
	}

    retval = lake_ecryptfs_file_buffered_read(iocb, iter, retval);
out:
	return retval;
}

static void shrink_readahead_size_eio(struct file *filp,
					struct file_ra_state *ra)
{
	ra->ra_pages /= 4;
}

//related to   static ssize_t generic_file_buffered_read(struct kiocb *iocb,
//			struct iov_iter *iter, ssize_t written)
ssize_t lake_ecryptfs_file_buffered_read(struct kiocb *iocb, 
            struct iov_iter *iter, ssize_t written)
{
	struct file *filp = iocb->ki_filp;
	struct address_space *mapping = filp->f_mapping;
	struct inode *inode = mapping->host;
	struct file_ra_state *ra = &filp->f_ra;
	loff_t *ppos = &iocb->ki_pos;
	pgoff_t index;
	pgoff_t last_index;
	pgoff_t prev_index;
	unsigned long offset;      /* offset into pagecache page */
	unsigned int prev_offset;
	int error = 0;

	struct page **pgs_cached, **pgs_no_cached;
	int nr_pgs;
	int i = 0, pg_idx = 0, nr_pgs_no_cached = 0, nr_pgs_cached = 0;
	//loff_t isize = i_size_read(inode);
    size_t real_count = iter->count;

	lake_print(KERN_ERR, "[lake] start of lake_ecryptfs_file_buffered_read\n");

    if (unlikely(*ppos >= i_size_read(inode))) {
        return 0;
	}
	if (unlikely(*ppos >= inode->i_sb->s_maxbytes)) {
		return 0;
	}
	iov_iter_truncate(iter, inode->i_sb->s_maxbytes);
    if (i_size_read(inode) - *ppos < real_count)
        real_count = i_size_read(inode) - *ppos;

    nr_pgs = DIV_ROUND_UP(real_count, PAGE_SIZE);
	pgs_cached = kzalloc(nr_pgs * sizeof(struct page *), GFP_KERNEL);
	pgs_no_cached = kzalloc(nr_pgs * sizeof(struct page *), GFP_KERNEL);
	if (!pgs_cached || !pgs_no_cached) {
	    error = -ENOMEM;
	    printk(KERN_ERR "[kava] Error allocating pages\n");
	    goto out;
	}

	lake_print(KERN_ERR, "[lake] reading %d pages\n", nr_pgs);

	index = *ppos >> PAGE_SHIFT;
	last_index = (*ppos + iter->count + PAGE_SIZE-1) >> PAGE_SHIFT;

	for (; pg_idx < nr_pgs;) {
		struct page *page;

		cond_resched();
find_page:
		if (fatal_signal_pending(current)) {
			error = -EINTR;
			goto out;
		}

		page = find_get_page(mapping, index);
		if (!page) {
			lake_print(KERN_ERR, "[lake] page not found in cache..\n");
			if (iocb->ki_flags & IOCB_NOWAIT)
				goto would_block;
			page_cache_sync_readahead(mapping,
					ra, filp,
					index, last_index - index);
			page = find_get_page(mapping, index);
			if (unlikely(page == NULL))
				goto no_cached_page;
		}
		if (PageReadahead(page)) {
			lake_print(KERN_ERR, "[lake] page FOUND in cache, doing async readahead\n");
			page_cache_async_readahead(mapping,
					ra, filp, page_folio(page),
					index, last_index - index);
		}

        pgs_cached[pg_idx++] = page;
        index++;
        nr_pgs_cached++;
		continue;

no_cached_page:
		/*
		 * Ok, it wasn't cached, so we need to create a new
		 * page..
		 */
		lake_print(KERN_ERR, "[lake] no_cached_page\n");
		page = page_cache_alloc(mapping);
		if (!page) {
			error = -ENOMEM;
			goto out;
		}
		error = add_to_page_cache_lru(page, mapping, index,
				mapping_gfp_constraint(mapping, GFP_KERNEL));
		if (error) {
			put_page(page);
			if (error == -EEXIST) {
				error = 0;
				goto find_page;
			}
			goto out;
		}

		//original code would do:
		//goto readpage;

        /*
         * A previous I/O error may have been due to temporary
         * failures, eg. multipath errors.
         * PG_error will be set again if readpage fails.
         */
        ClearPageError(page);

        pgs_no_cached[nr_pgs_no_cached++] = page;
        pg_idx++;
        index++;
	}

    /* Start the actual read. The read will unlock the page. */
    lake_print(KERN_ERR, "nr_pgs_no_cached = %d, nr_pgs_cached = %d\n", nr_pgs_no_cached, nr_pgs_cached);
    //error = mapping->a_ops->readpages(filp, mapping, pgs_no_cached, nr_pgs_no_cached);
	// this would call ecryptfs_readpage in mmap.c
    if (nr_pgs_no_cached) {
		//ecryptfs_printk(KERN_ERR, "[lake] calling to decrypt %d pages\n", nr_pgs_no_cached);
        error = lake_ecryptfs_decrypt_pages(pgs_no_cached, nr_pgs_no_cached);
		if (unlikely(error)) {
			//ecryptfs_printk(KERN_ERR, "!!!!! [lake] error decrypting pages\n");
			goto readpage_error;
		}
	}

//page_ok:
	index = *ppos >> PAGE_SHIFT;
	prev_index = ra->prev_pos >> PAGE_SHIFT;
	prev_offset = ra->prev_pos & (PAGE_SIZE-1);
	offset = *ppos & ~PAGE_MASK;

    pg_idx = 0;
    i = 0;

    for (; pg_idx < nr_pgs;) {
        struct page *page;
		pgoff_t end_index;
		loff_t isize;
		unsigned long nr, ret;

        page = pgs_cached[pg_idx];
        pgs_cached[pg_idx++] = NULL;
        if (likely(!page)) {
            page = pgs_no_cached[i];
            pgs_no_cached[i++] = NULL;

            if (unlikely(!page)) {
                error = -EEXIST;
                goto out;
            }
//readpage:
			ClearPageError(page);
			///* Start the actual read. The read will unlock the page. */
			//error = mapping->a_ops->readpage(filp, page);

            if (!PageUptodate(page)) {
                error = lock_page_killable(page);
                if (unlikely(error))
                    goto readpage_error;
                if (!PageUptodate(page)) {
                    if (page->mapping == NULL) {
                        /*
                         * invalidate_mapping_pages got it
                         */
                        unlock_page(page);
                        put_page(page);
                        error = -EIO;
                        goto readpage_error;
                    }
                    unlock_page(page);
                    shrink_readahead_size_eio(filp, ra);
                    error = -EIO;
                    goto readpage_error;
                }
                unlock_page(page);
            }
        }

//page_ok:
		/*
		 * i_size must be checked after we know the page is Uptodate.
		 *
		 * Checking i_size after the check allows us to calculate
		 * the correct value for "nr", which means the zero-filled
		 * part of the page is not copied back to userspace (unless
		 * another truncate extends the file - this is desired though).
		 */

		isize = i_size_read(inode);
		end_index = (isize - 1) >> PAGE_SHIFT;
		if (unlikely(!isize || index > end_index)) {
			put_page(page);
			goto out;
		}

		/* nr is the maximum number of bytes to copy from this page */
		nr = PAGE_SIZE;
		if (index == end_index) {
			nr = ((isize - 1) & ~PAGE_MASK) + 1;
			if (nr <= offset) {
				put_page(page);
				goto out;
			}
		}
		nr = nr - offset;

		/* If users can be writing to this page using arbitrary
		 * virtual addresses, take care about potential aliasing
		 * before reading the page on the kernel side.
		 */
		if (mapping_writably_mapped(mapping))
			flush_dcache_page(page);

		/*
		 * When a sequential read accesses a page several times,
		 * only mark it as accessed the first time.
		 */
		if (prev_index != index || offset != prev_offset)
			mark_page_accessed(page);
		prev_index = index;

		/*
		 * Ok, we have the page, and it's up-to-date, so
		 * now we can copy it to user space...
		 */

		ret = copy_page_to_iter(page, offset, nr, iter);
		offset += ret;
		index += offset >> PAGE_SHIFT;
		offset &= ~PAGE_MASK;
		prev_offset = offset;

		put_page(page);
		written += ret;
		if (!iov_iter_count(iter))
			goto out;
		if (ret < nr) {
			error = -EFAULT;
			goto out;
		}
		//continue
    }

readpage_error:
	/* UHHUH! A synchronous read error occurred. Report it */
	//ecryptfs_printk(KERN_ERR, "[lake] readpage_error\n");
	goto out;

would_block:
	error = -EAGAIN;
out:
    for (i = 0; i < nr_pgs; i++) {
        if (pgs_cached[i])
            put_page(pgs_cached[i]);
        if (pgs_no_cached[i])
            put_page(pgs_no_cached[i]);
    }

    if (nr_pgs > 0) {
        kfree(pgs_cached);
        kfree(pgs_no_cached);
    }

	ra->prev_pos = prev_index;
	ra->prev_pos <<= PAGE_SHIFT;
	ra->prev_pos |= prev_offset;

	*ppos = ((loff_t)index << PAGE_SHIFT) + offset;
	file_accessed(filp);
	return written ? written : error;
}

//related to ecryptfs_decrypt_page(struct page *page)
// and crypt_extent_aead
int lake_ecryptfs_decrypt_pages(struct page **pgs, unsigned int nr_pages)
{
	//ecryptfs_decrypt_page
	struct inode *ecryptfs_inode;
    struct ecryptfs_crypt_stat *crypt_stat;
	struct ecryptfs_extent_metadata extent_metadata;
	char *page_virt;
	unsigned long extent_offset = 0;
	loff_t lower_offset;
	int rc = 0;
	int num_extents = 1;
	int data_extent_num;
	int meta_extent_num;
	int metadata_per_extent;
	u8 *tag_data = NULL;
	u8 *iv_data = NULL;

	struct scatterlist *src_sg = NULL;
	struct scatterlist *dst_sg = NULL;
    unsigned int i = 0;

	lake_print(KERN_ERR, "[lake] lake_ecryptfs_decrypt_pages %d pages\n", nr_pages);

    if (!nr_pages || !pgs || !pgs[0]) {
        goto out;
    }

	ecryptfs_inode = pgs[0]->mapping->host;
    crypt_stat =
        &(ecryptfs_inode_to_private(ecryptfs_inode)->crypt_stat);
    BUG_ON(!(crypt_stat->flags & ECRYPTFS_ENCRYPTED));

	metadata_per_extent = crypt_stat->extent_size / sizeof(extent_metadata);
	//extent_size = crypt_stat->extent_size;

    src_sg = (struct scatterlist *)kmalloc(nr_pages * sizeof(struct scatterlist) * 2,
            GFP_KERNEL);
    if (!src_sg) {
        rc = -EFAULT;
        ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for "
                "source scatter list\n");
        goto out;
    }

	dst_sg = (struct scatterlist *)kmalloc(nr_pages * sizeof(struct scatterlist),
            GFP_KERNEL);
    if (!dst_sg) {
        rc = -EFAULT;
        ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for "
                "dest scatter list\n");
        goto out;
    }

    sg_init_table(src_sg, nr_pages*2);
	sg_init_table(dst_sg, nr_pages);

	tag_data = kmalloc(ECRYPTFS_GCM_TAG_SIZE * nr_pages, GFP_KERNEL);
	if (!tag_data) {
		rc = -ENOMEM;
		ecryptfs_printk(KERN_ERR, "Error allocating memory for "
				"auth_tag\n");
		goto out;
	}

	iv_data = kmalloc(ECRYPTFS_MAX_IV_BYTES * nr_pages, GFP_KERNEL);
	if (!iv_data) {
		rc = -ENOMEM;
		ecryptfs_printk(KERN_ERR, "Error allocating memory for "
				"iv_data\n");
		goto out;
	}

    for (i = 0; i < nr_pages; i++) {
		/*
		* Lower offset must take into account the number of
		* data extents, auth tag extents, and header size.
		*/
		lower_offset = ecryptfs_lower_header_size(crypt_stat);
		data_extent_num = (pgs[i]->index * num_extents) + 1;
		data_extent_num += extent_offset;
		lower_offset += (data_extent_num - 1)
			* crypt_stat->extent_size;
		meta_extent_num = (data_extent_num
			+ (metadata_per_extent - 1))
			/ metadata_per_extent;
		lower_offset += meta_extent_num
			* crypt_stat->extent_size;

		page_virt = kmap(pgs[i]);
		rc = ecryptfs_read_lower(page_virt +
			(extent_offset * crypt_stat->extent_size),
			lower_offset,
			crypt_stat->extent_size,
			ecryptfs_inode);
		kunmap(pgs[i]);


		if (rc < 0) {
			printk(KERN_ERR "Error attempting to read lower"
					"page; rc = [%d]\n", rc);
			goto out;
		}

		lower_offset = ecryptfs_lower_header_size(crypt_stat);
		lower_offset += (meta_extent_num - 1) *
			(metadata_per_extent + 1) *
			crypt_stat->extent_size;

		rc = ecryptfs_read_lower((void *)&extent_metadata,
				lower_offset,
				sizeof(extent_metadata),
				ecryptfs_inode);

		memcpy(tag_data + ECRYPTFS_GCM_TAG_SIZE * i,
			&(extent_metadata.auth_tag_bytes),
			ECRYPTFS_GCM_TAG_SIZE);

		memcpy(iv_data + ECRYPTFS_MAX_IV_BYTES * i,
			&(extent_metadata.iv_bytes),
			ECRYPTFS_MAX_IV_BYTES);

		if (rc < 0) {
			printk(KERN_ERR "Error attempting to read lower"
					"page; rc = [%d]\n", rc);
		}

		// original code would call crypt_extent_aead now
		//rc = crypt_extent_aead(crypt_stat, page, page,
		//		  tag_data, iv_data, extent_offset, DECRYPT);

		sg_set_page(&src_sg[i*2], pgs[i], PAGE_SIZE, 0);
		sg_set_buf(&src_sg[(i*2)+1], tag_data + (i*ECRYPTFS_GCM_TAG_SIZE), ECRYPTFS_GCM_TAG_SIZE);
		sg_set_page(&dst_sg[i], pgs[i], PAGE_SIZE, 0);
    }

    rc = crypt_scatterlist(crypt_stat, dst_sg, src_sg, nr_pages * PAGE_SIZE,
            iv_data, DECRYPT);

	if (rc == -74) {
		//ecryptfs_printk(KERN_ERR, "Decryption auth failed, ignoring for now..\n");
		rc = 0;
	}

	if (rc < 0) {
		printk(KERN_ERR "Error attempting to crypt pages "
		       "rc = [%d]\n", rc);
		goto out;
	}

    for (i = 0; i < nr_pages; i++) {
        SetPageUptodate(pgs[i]);
        if (PageLocked(pgs[i]))
            unlock_page(pgs[i]);
	}

out:
    kfree(src_sg);
	kfree(dst_sg);
	kfree(tag_data);
	kfree(iv_data);

    return (rc >= 0 ? 0 : rc);
}

/**********************************************
 * 
 *   END OF READ_ITER
 * 
 *********************************************/

/**********************************************
 * 
 *   START OF MMAP OPS
 * 
 *********************************************/

void lake_ecryptfs_readahead(struct readahead_control *ractl)
{
	struct file *filp = ractl->file;
	struct address_space *mapping = ractl->mapping;
	struct ecryptfs_crypt_stat *crypt_stat =
	    		&ecryptfs_inode_to_private(mapping->host)->crypt_stat;
	unsigned int pgcount = readahead_count(ractl);
	struct page **pgs = NULL;
	int dont_decrypt = 0;	//no decryption needed flag
	unsigned int page_idx = 0;
	int rc = 0;
	
	lake_print(KERN_ERR, "[lake] ++++++  lake_ecryptfs_readahead %u pages\n", pgcount);

	if (!crypt_stat
	    || !(crypt_stat->flags & ECRYPTFS_ENCRYPTED)
	    || (crypt_stat->flags & ECRYPTFS_VIEW_AS_ENCRYPTED)) {
	    dont_decrypt = 1;
	}

	if (!dont_decrypt) {
	    pgs = (struct page **)kmalloc(pgcount * sizeof(struct page *), GFP_KERNEL);
	    if (!pgs) {
			pr_err("Cant allocate page strucs!\n");
			return;
	    }
	}

	for (page_idx = 0; page_idx < pgcount; page_idx++) {
		//fetch next page we need to read
		struct page *page = readahead_page(ractl);

        if (dont_decrypt) {
			lake_print(KERN_ERR, "[lake] ++++++  dont_decrypt is set\n");
			rc |= ecryptfs_read_folio(filp, page_folio(page));
            put_page(page);
		}
        else
            pgs[page_idx] = page;
    }

    if (!dont_decrypt) {
        rc = lake_ecryptfs_decrypt_pages(pgs, pgcount);
        for (page_idx = 0; page_idx < pgcount; page_idx++) {
            if (rc)
                ClearPageUptodate(pgs[page_idx]);
            else
                SetPageUptodate(pgs[page_idx]);
            unlock_page(pgs[page_idx]);
            put_page(pgs[page_idx]);
        }
        kfree(pgs);
    }
}


// int lake_ecryptfs_mmap_readpages(struct file *filp, struct address_space *mapping,
// 			      struct list_head *pages, unsigned nr_pages)
// {
// 	struct ecryptfs_crypt_stat *crypt_stat =
// 	    		&ecryptfs_inode_to_private(mapping->host)->crypt_stat;
// 	struct page **pgs = NULL;
// 	unsigned int page_idx = 0;
// 	int rc = 0;
// 	int nodec = 0;	//no decryption needed flag

// 	lake_print(KERN_ERR, "[lake] ++++++  ecryptfs_readpages\n");

// 	if (!crypt_stat
// 	    || !(crypt_stat->flags & ECRYPTFS_ENCRYPTED)
// 	    || (crypt_stat->flags & ECRYPTFS_VIEW_AS_ENCRYPTED)) {
// 	    nodec = 1;
// 	}

// 	if (!nodec) {
// 	    pgs = (struct page **)kmalloc(nr_pages * sizeof(struct page *), GFP_KERNEL);
// 	    if (!pgs) {
//             return -EFAULT;
// 	    }
// 	}

// 	for (page_idx = 0; page_idx < nr_pages; page_idx++) {
// 	    struct page *page = list_entry(pages->prev, struct page, lru);
// 	    list_del(&page->lru);
// 	    if (add_to_page_cache_lru(page, mapping, page->index, GFP_KERNEL)) {
// 			printk(KERN_INFO "[kava] INFO: cannot add page %lu to cache lru\n",
// 			       (unsigned long)(page->index));
// 	    }
//         else {
//             if (nodec)
//                 rc |= ecryptfs_readpage(filp, page);
//         }

//         if (nodec)
//             put_page(page);
//         else
//             pgs[page_idx] = page;
//     }

//     if (!nodec) {
//         rc = lake_ecryptfs_decrypt_pages(pgs, nr_pages);
//         for (page_idx = 0; page_idx < nr_pages; page_idx++) {
//             if (rc)
//                 ClearPageUptodate(pgs[page_idx]);
//             else
//                 SetPageUptodate(pgs[page_idx]);
//             unlock_page(pgs[page_idx]);
//             put_page(pgs[page_idx]);
//         }
//         kfree(pgs);
//     }
// 	return 0;
// }



#endif