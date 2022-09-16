#ifdef __KERNEL__
#include <linux/scatterlist.h>
#include <crypto/aead.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/crypto.h>
#include <linux/completion.h>
#include "lake_shm.h"
#else
#include <stdio.h>
#include <stdlib.h>
#endif

struct extent_crypt_result {
	struct completion completion;
	int rc;
};

static void extent_crypt_complete(struct crypto_async_request *req, int rc)
{
	struct extent_crypt_result *ecr = req->data;
    printk(KERN_ERR "completing.. \n");
	if (rc == -EINPROGRESS)
		return;

	ecr->rc = rc;
	complete(&ecr->completion);
}

static int ecryptfs_test_init(void)
{
	int rc;
    struct crypto_aead *aead_tfm;
    struct aead_request *aead_req = NULL;
    struct extent_crypt_result ecr;
    int n = 1;
    char** src_bufs=0;
    char** dst_bufs=0;
    char** dec_bufs=0;
    char* iv;
    u8 key[32];
    int i, j;
    struct scatterlist *src_sg=0, *dst_sg=0, *dec_sg=0;

    printk(KERN_ERR "starting\n");

    aead_tfm = crypto_alloc_aead("generic-gcm-aesni", 0, 0);
    if (IS_ERR(aead_tfm)) {
        printk(KERN_ERR "err crypto_alloc_aead %ld\n", PTR_ERR(aead_tfm));
        return -2;
    }

    crypto_aead_set_flags(aead_tfm, CRYPTO_TFM_REQ_FORBID_WEAK_KEYS);

	aead_req = aead_request_alloc(aead_tfm, GFP_NOFS);
    if (!aead_req) {
        printk(KERN_ERR "err aead_request_alloc\n");
        return -1;
    }

    aead_request_set_callback(aead_req,
			CRYPTO_TFM_REQ_MAY_BACKLOG | CRYPTO_TFM_REQ_MAY_SLEEP,
			extent_crypt_complete, &ecr);

    for (i = 0 ; i < 32 ; i++)
        key[i] = i;

    rc = crypto_aead_setkey(aead_tfm, key, 32);
    if (rc) {
        printk(KERN_ERR "err setkey\n");
        //return -1;
    }

    rc = crypto_aead_setauthsize(aead_tfm, 16);
    if (rc) {
        printk(KERN_ERR "err crypto_aead_setauthsize\n");
        return -1;
    }

    src_bufs = vmalloc(n * sizeof(char*));
    dst_bufs = vmalloc(2 * n * sizeof(char*));
    dec_bufs = vmalloc(n * sizeof(char*));
    iv = vmalloc(12);
    for (i=0 ; i < 12 ; i++) {
        iv[i] = i;
    }

    for (i=0 ; i < n ; i++) {
        src_bufs[i] = vmalloc(PAGE_SIZE);
        dst_bufs[i*2] = vmalloc(PAGE_SIZE);
        dst_bufs[(i*2)+1] = vmalloc(16);
        memset(dst_bufs[(i*2)+1], 0, 16);
        dec_bufs[i] = vmalloc(PAGE_SIZE);
    }

    for (i=0 ; i < n ; i++) {
        for (j=0 ; j < PAGE_SIZE ; j++) {
            src_bufs[i][j] = j%255;
        }
    }

    src_sg = (struct scatterlist *)kmalloc(
 		n * sizeof(struct scatterlist), GFP_KERNEL);
    dst_sg = (struct scatterlist *)kmalloc(
 		2 * n * sizeof(struct scatterlist), GFP_KERNEL);
    dec_sg = (struct scatterlist *)kmalloc(
 		n * sizeof(struct scatterlist), GFP_KERNEL);

    sg_init_table(src_sg, n);
    sg_init_table(dst_sg, n*2);
    sg_init_table(dec_sg, n);

    for (i=0 ; i < n ; i++) {
        sg_set_buf(&src_sg[i], src_bufs[i], PAGE_SIZE);
        sg_set_buf(&dst_sg[i*2], dst_bufs[i*2], PAGE_SIZE);
        sg_set_buf(&dst_sg[(i*2)+1], dst_bufs[(i*2)+1], PAGE_SIZE);
        sg_set_buf(&dec_sg[i], dec_bufs[i], PAGE_SIZE);
    }

    aead_request_set_crypt(aead_req, src_sg, dst_sg, n*PAGE_SIZE, iv);
    aead_request_set_ad(aead_req, 0);

    rc = crypto_aead_encrypt(aead_req);
    if (rc == -EINPROGRESS || rc == -EBUSY) {
        printk(KERN_DEBUG "waiting..\n");
		struct extent_crypt_result *ecr;
		ecr = aead_req->base.data;
		wait_for_completion(&ecr->completion);
		rc = ecr->rc;
		reinit_completion(&ecr->completion);
	}
	printk(KERN_DEBUG "Encryption done. %d\n", rc);

    // for (i=0 ; i < n ; i++) {
    //     printk(KERN_DEBUG "MAC %d ", i);
    //     for (j=0 ; j < 16 ; j++) {
    //         printk(KERN_DEBUG "%x", dst_bufs[(i*2)+1][j]);
    //     }
    //     printk(KERN_DEBUG "\n");
    // }

    //start decryption
    aead_request_free(aead_req);
    aead_req = aead_request_alloc(aead_tfm, GFP_NOFS);
    if (!aead_req) {
        printk(KERN_ERR "err aead_request_alloc\n");
        return -1;
    }
    aead_request_set_crypt(aead_req, dst_sg, dec_sg, n*(PAGE_SIZE+16), iv);
    aead_request_set_ad(aead_req, 0);

    rc = crypto_aead_decrypt(aead_req);
    if (rc == -EINPROGRESS || rc == -EBUSY) {
        printk(KERN_DEBUG "waiting..\n");
		struct extent_crypt_result *ecr;
		ecr = aead_req->base.data;
		wait_for_completion(&ecr->completion);
		rc = ecr->rc;
		reinit_completion(&ecr->completion);
	}
    printk(KERN_DEBUG "Decryption done.\n");
    aead_request_free(aead_req);

    for (i=0 ; i < n ; i++) {
        for (j=0 ; j < PAGE_SIZE ; j++) {
            if(src_bufs[i][j] != dec_bufs[i][j]) {
                printk(KERN_DEBUG "Error in src vs. dec: %d != %d at idx %d/%d\n", 
                    src_bufs[i][j], dec_bufs[i][j], i, j);
                return -EFAULT;
            }
        }
    }

    printk(KERN_DEBUG "Test passed!\n");
out:
    vfree(src_bufs);
    vfree(dst_bufs);
    vfree(dec_bufs);
    kfree(src_sg);
    kfree(dst_sg);
    kfree(dec_sg);
    crypto_free_aead(aead_tfm);
    //return an error so the module is unloaded
    return -EAGAIN;
}

static void ecryptfs_test_exit(void)
{
}

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("crypto aesni test");

MODULE_LICENSE("GPL");
module_init(ecryptfs_test_init)
module_exit(ecryptfs_test_exit)