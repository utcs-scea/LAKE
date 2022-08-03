#ifdef LAKE_ECRYPTFS

#include "lake.h"
#include "ecryptfs_kernel.h"
#include <linux/sched/signal.h>
#include <linux/random.h>
#include <linux/scatterlist.h>

#define DECRYPT		0
#define ENCRYPT		1

ssize_t lake_ecryptfs_file_write(struct file *file, const char __user *data,
            size_t size, loff_t *poffset) 
{
    lake_ecryptfs_write(file_inode(file), (char *)data, *poffset, size);
    *poffset += size;
    return size;
}

//was on read_write.c
// related to int ecryptfs_write(struct inode *ecryptfs_inode, char *data, loff_t offset, size_t size)
int lake_ecryptfs_write(struct inode *ecryptfs_inode, char *data, loff_t offset, size_t size)
{	
    struct page *ecryptfs_page;
 	struct ecryptfs_crypt_stat *crypt_stat;
 	char *ecryptfs_page_virt;
 	loff_t ecryptfs_file_size = i_size_read(ecryptfs_inode);
 	loff_t data_offset = offset;
 	loff_t pos;
 	int rc = 0;

    //lake
    struct address_space *mapping = ecryptfs_inode->i_mapping;
    struct page **pgs;
 	int nr_pgs = DIV_ROUND_UP(size, PAGE_SIZE);
 	int i = 0;
 	pgs = kmalloc(nr_pgs * sizeof(struct page *), GFP_KERNEL);
 	if (!pgs) {
 	    rc = -ENOMEM;
 	    printk(KERN_ERR "[lake] Error allocating pages\n");
 	    goto out;
 	}
 
 	crypt_stat = &ecryptfs_inode_to_private(ecryptfs_inode)->crypt_stat;
 	/*
 	 * if we are writing beyond current size, then start pos
 	 * at the current size - we'll fill in zeros from there.
 	 */
 	if (offset > ecryptfs_file_size)
 		pos = ecryptfs_file_size;
 	else
 		pos = offset;
 
 	while (pos < (offset + size)) {
 		pgoff_t ecryptfs_page_idx = (pos >> PAGE_SHIFT);
 		size_t start_offset_in_page = (pos & ~PAGE_MASK);
 		size_t num_bytes = (PAGE_SIZE - start_offset_in_page);
 		loff_t total_remaining_bytes = ((offset + size) - pos);
 
         if (fatal_signal_pending(current)) {
             rc = -EINTR;
             break;
         }
 
 		if (num_bytes > total_remaining_bytes)
 			num_bytes = total_remaining_bytes;
 		if (pos < offset) {
 			/* remaining zeros to write, up to destination offset */
 			loff_t total_remaining_zeros = (offset - pos);
 
 			if (num_bytes > total_remaining_zeros)
 				num_bytes = total_remaining_zeros;
 		}

        /* lake: the following change is only correct when overwriting the whole page.
        * TODO: use ecryptfs_get_locked_page when only modify part of the page.
        */
 		//ecryptfs_page = ecryptfs_get_locked_page(ecryptfs_inode,
 		//					 ecryptfs_page_idx);
 		ecryptfs_page = page_cache_alloc(mapping);
 		if (IS_ERR(ecryptfs_page)) {
 			rc = PTR_ERR(ecryptfs_page);
 			printk(KERN_ERR "%s: Error getting page at "
 			       "index [%ld] from eCryptfs inode "
 			       "mapping; rc = [%d]\n", __func__,
 			       ecryptfs_page_idx, rc);
 			goto out;
 		}
 		rc = add_to_page_cache_lru(ecryptfs_page, mapping, ecryptfs_page_idx,
 				mapping_gfp_constraint(mapping, GFP_KERNEL));
 		if (rc) {
 			put_page(ecryptfs_page);
 			printk(KERN_ERR "%s: Error adding page to cache lru at "
 			       "index [%ld] from eCryptfs inode "
 			       "mapping; rc = [%d]\n", __func__,
 			       ecryptfs_page_idx, rc);
 			goto out;
 		}
         ClearPageError(ecryptfs_page);
 
 		ecryptfs_page_virt = kmap(ecryptfs_page);
        //ecryptfs_page_virt = kmap_atomic(ecryptfs_page);

 		/*
 		 * pos: where we're now writing, offset: where the request was
 		 * If current pos is before request, we are filling zeros
 		 * If we are at or beyond request, we are writing the *data*
 		 * If we're in a fresh page beyond eof, zero it in either case
 		 */
 		if (pos < offset || !start_offset_in_page) {
 			/* We are extending past the previous end of the file.
 			 * Fill in zero values to the end of the page */
 			memset(((char *)ecryptfs_page_virt
 				+ start_offset_in_page), 0,
 				PAGE_SIZE - start_offset_in_page);
 		}
 
 		/* pos >= offset, we are now writing the data request */
 		if (pos >= offset) {
 			copy_from_user(((char *)ecryptfs_page_virt
                 + start_offset_in_page),
 			       (data + data_offset), num_bytes);
 			data_offset += num_bytes;
 		}
 		kunmap(ecryptfs_page);
 		flush_dcache_page(ecryptfs_page);
 		SetPageUptodate(ecryptfs_page);
 		unlock_page(ecryptfs_page);

 		if (crypt_stat->flags & ECRYPTFS_ENCRYPTED) {
 		    pgs[i++] = ecryptfs_page;
            //rc = ecryptfs_encrypt_page(ecryptfs_page);
 		}
 		else {
 		    rc = ecryptfs_write_lower_page_segment(ecryptfs_inode,
 						ecryptfs_page,
 						start_offset_in_page,
 						data_offset);
 		    put_page(ecryptfs_page);
 		    if (rc) {
                 printk(KERN_ERR "%s: Error encrypting "
                        "page; rc = [%d]\n", __func__, rc);
                 goto out;
 		    }
 		}
        //put_page(ecryptfs_page);
        //if (rc) {
        //    printk(KERN_ERR "%s: Error encrypting "
        //            "page; rc = [%d]\n", __func__, rc);
        //    goto out;
 		//}

 		pos += num_bytes;
 	}
 
    //lake
 	if (crypt_stat->flags & ECRYPTFS_ENCRYPTED) {
 	    rc = lake_ecryptfs_encrypt_pages(crypt_stat, pgs, nr_pgs);
 	    for (i = 0; i < nr_pgs; i++)
 		    put_page(pgs[i]);
 	    kfree(pgs);
 	}
 
 	if (pos > ecryptfs_file_size) {
 		i_size_write(ecryptfs_inode, (offset + size));
 		if (crypt_stat->flags & ECRYPTFS_ENCRYPTED) {
             int rc2;
 
 			rc2 = ecryptfs_write_inode_size_to_metadata(ecryptfs_inode);
 			if (rc2) {
 				printk(KERN_ERR	"Problem with "
 				       "ecryptfs_write_inode_size_to_metadata; "
 				       "rc = [%d]\n", rc);
                 if (!rc)
                     rc = rc2;
 				goto out;
 			}
 		}
 	}
 out:
 	return rc;
}

// our own function, sort of related to crypt_extent_aead
int lake_ecryptfs_encrypt_pages(struct ecryptfs_crypt_stat *crypt_stat, struct page **pgs, unsigned int nr_pages)
{
 	struct inode *ecryptfs_inode;
 	loff_t lower_offset;
 	struct page *enc_extent_page  = NULL;
 	char *enc_extent_virt;
 	int rc = 0;
 
	int meta_extent_num;
	int data_extent_num;
	u8 *extent_iv = NULL;
	u8 *tag_data_dst = NULL;
	struct ecryptfs_extent_metadata extent_metadata;

 	struct scatterlist *src_sg = NULL, *dst_sg = NULL;
 	unsigned int i = 0;
 	u32 sz = 0;
	int metadata_per_extent = crypt_stat->extent_size / sizeof(extent_metadata);

 	if (!nr_pages || !pgs || !pgs[0]) {
 		goto out;
 	}

	ecryptfs_printk(KERN_ERR, "[lake] lake_ecryptfs_encrypt_pages %d "
                 "pages\n", nr_pages);

	// source sgs
 	src_sg = (struct scatterlist *)kmalloc(
 		nr_pages * sizeof(struct scatterlist), GFP_KERNEL);
 	if (!src_sg) {
 		rc = -ENOMEM;
 		ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for "
                 "source scatter list\n");
 		goto higher_out;
     }
 
	// dst sgs
    dst_sg = (struct scatterlist *)kmalloc(
 		nr_pages * sizeof(struct scatterlist), GFP_KERNEL);
 	if (!dst_sg) {
 		ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for "
                 "destination scatter list\n");
 		rc = -ENOMEM;
 		goto higher_out;
    }
 
	//ivs
	extent_iv = (u8 *)kmalloc(nr_pages * ECRYPTFS_MAX_IV_BYTES, GFP_KERNEL);
 	if (!extent_iv) {
 		ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for "
                 "ivs\n");
 		rc = -ENOMEM;
 		goto higher_out;
    }

	//tags
	tag_data_dst = (u8 *)kmalloc(nr_pages * ECRYPTFS_GCM_TAG_SIZE, GFP_KERNEL);
 	if (!tag_data_dst) {
 		ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for "
                 "ivs\n");
 		rc = -ENOMEM;
 		goto higher_out;
    }

	//generate ivs
	for (i = 0; i < nr_pages; i++) {
		get_random_bytes(extent_iv+(i*ECRYPTFS_MAX_IV_BYTES), ECRYPTFS_MAX_IV_BYTES);
	}

    sg_init_table(src_sg, nr_pages);
    sg_init_table(dst_sg, nr_pages*2);
    ecryptfs_inode = pgs[0]->mapping->host;
 
 	for (i = 0; i < nr_pages; i++) {
 		enc_extent_page = alloc_page(GFP_USER);
 		if (!enc_extent_page) {
 			rc = -ENOMEM;
            ecryptfs_printk(KERN_ERR, "[lake] Error allocating memory for "
                     "encrypted extent\n");
 			for (sz = 0; sz < i; sz++) {
 				__free_page(sg_page(dst_sg + sz));
 			}
 			goto higher_out;
 		}
 
 		sg_set_page(src_sg + i, pgs[i], PAGE_SIZE, 0);
 		sg_set_page(dst_sg + i*2, enc_extent_page, PAGE_SIZE, 0);
		sg_set_buf(dst_sg + (i*2) + 1, tag_data_dst + (i*ECRYPTFS_GCM_TAG_SIZE), 
				ECRYPTFS_GCM_TAG_SIZE);
 	}
 
 	rc = crypt_scatterlist(crypt_stat, dst_sg, src_sg, PAGE_SIZE * nr_pages,
             extent_iv, ENCRYPT);
     if (rc) {
         printk(KERN_ERR "%s: Error encrypting extents in scatter list; "
                 "rc = [%d]\n", __func__, rc);
 	    for (i = 0; i < nr_pages; i++) {
             __free_page(sg_page(dst_sg + i));
         }
         goto higher_out;
     }
 
 	for (i = 0; i < nr_pages; i++) {
        // lower_offset = lower_offset_for_page(crypt_stat, pgs[i]);
 		// ret = ecryptfs_write_lower(ecryptfs_inode, enc_extent_virt, lower_offset,
        //          PAGE_SIZE);
 		// kunmap(enc_extent_page);
 		// __free_page(enc_extent_page);
 
 		// if (ret < 0) {
 		// 	ecryptfs_printk(KERN_ERR, "Error attempting "
 		// 			"to write lower page; rc = [%d]\n", ret);
 		// 	rc = ret;
 		// }

		/*
		* Lower offset must take into account the number of
		* data extents, auth tag extents, and header size.
		*/
		lower_offset = ecryptfs_lower_header_size(crypt_stat);
		//data_extent_num = page->index + 1;
		data_extent_num = pgs[i]->index + 1;
		lower_offset += (data_extent_num - 1)
			* crypt_stat->extent_size;
		meta_extent_num = (data_extent_num
			+ (metadata_per_extent - 1))
			/ metadata_per_extent;
		lower_offset += meta_extent_num
			* crypt_stat->extent_size;

		enc_extent_page = sg_page(dst_sg + (i*2));
		enc_extent_virt = kmap(enc_extent_page);
		rc = ecryptfs_write_lower(ecryptfs_inode,
				enc_extent_virt,
				lower_offset,
				crypt_stat->extent_size);
		kunmap(enc_extent_page);
		if (rc < 0) {
			printk(KERN_ERR "Error attempting to write lower"
					"page; rc = [%d]\n", rc);
			goto out;
		}

		memcpy(extent_metadata.iv_bytes, extent_iv+(i*ECRYPTFS_MAX_IV_BYTES),
			ECRYPTFS_MAX_IV_BYTES);
		memcpy(extent_metadata.auth_tag_bytes, tag_data_dst+(i*ECRYPTFS_GCM_TAG_SIZE),
			ECRYPTFS_GCM_TAG_SIZE);

		lower_offset = ecryptfs_lower_header_size(crypt_stat);
		lower_offset += (meta_extent_num - 1) *
			(metadata_per_extent + 1) *
			crypt_stat->extent_size;

		rc = ecryptfs_write_lower(ecryptfs_inode,
				(void *) &extent_metadata,
				lower_offset,
				sizeof(extent_metadata));
				//ECRYPTFS_GCM_TAG_SIZE);
		if (rc < 0) {
			printk(KERN_ERR "Error attempting to write lower"
					"page; rc = [%d]\n", rc);
			goto out;
		}
 	}
 
higher_out:
 	kfree(src_sg);
 	kfree(dst_sg);
	kfree(extent_iv);
	kfree(tag_data_dst);
out:
 	return rc;
}

ssize_t lake_ecryptfs_read_update_atime(struct kiocb *iocb, struct iov_iter *to)
{
	printk(KERN_ERR "NIY lake_ecryptfs_read_update_atime\n");
    return 0;
}

ssize_t lake_ecryptfs_file_read_iter(struct kiocb *iocb, struct iov_iter *iter)
{
	printk(KERN_ERR "NIY lake_ecryptfs_file_read_iter\n");
    return 0;
}

ssize_t lake_ecryptfs_file_buffered_read(struct kiocb *iocb, 
            struct iov_iter *iter, ssize_t written)
{
	printk(KERN_ERR "NIY lake_ecryptfs_file_buffered_read\n");
    return 0;
}

int lake_ecryptfs_decrypt_pages(struct page **pgs, unsigned int nr_pages)
{
	printk(KERN_ERR "NIY lake_ecryptfs_decrypt_pages\n");
    return 0;
}

int lake_ecryptfs_mmap_writepages(struct address_space *mapping,
			       struct writeback_control *wbc)
{
	printk(KERN_ERR "NIY lake_ecryptfs_mmap_writepages\n");
    return 0;
}

int lake_ecryptfs_mmap_encrypt_pages(struct page **pgs, unsigned int nr_pages)
{
	printk(KERN_ERR "NIY lake_ecryptfs_mmap_encrypt_pages\n");
    return 0;
}

int lake_ecryptfs_mmap_readpages(struct file *filp, struct address_space *mapping,
			      struct list_head *pages, unsigned nr_pages)
{
	printk(KERN_ERR "NIY lake_ecryptfs_mmap_readpages\n");
    return 0;
}


#endif