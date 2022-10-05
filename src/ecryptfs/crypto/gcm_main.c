/*
 * GCM: Galois/Counter Mode.
 *
 * Copyright (c) 2007 Nokia Siemens Networks - Mikko Herranen <mh1@iki.fi>
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published
 * by the Free Software Foundation.
 */

#include <crypto/gf128mul.h>
#include <crypto/internal/aead.h>
#include <crypto/internal/skcipher.h>
#include <crypto/internal/hash.h>
#include <crypto/null.h>
#include <linux/scatterlist.h>
#include <crypto/gcm.h>
#include <crypto/hash.h>
//#include "internal.h"
#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/fs.h> 
#include <asm/uaccess.h>
#include "lake_shm.h"
#include "gcm_cuda.h"

static char *cubin_path = "gcm_kernels.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to gcm_kernels.cubin");

static int aesni_fraction = 0;
module_param(aesni_fraction, int, 0444);
MODULE_PARM_DESC(aesni_fraction, "Fraction of the file to be encrypted using AES-NI (out of 100), default 0");

//tfm ctx
struct crypto_gcm_ctx {
	struct AES_GCM_engine_ctx cuda_ctx;
	struct crypto_aead *aesni_tfm;
};

struct extent_crypt_result {
	struct completion completion;
	int rc;
};

static int get_aesni_fraction(int n) {
	return n*aesni_fraction/100;
}

static void extent_crypt_complete(struct crypto_async_request *req, int rc)
{
	struct extent_crypt_result *ecr = req->data;
	if (rc == -EINPROGRESS)
		return;
	ecr->rc = rc;
	DBG_PRINT("completing.. \n");
	complete(&ecr->completion);
}

static int crypto_gcm_setkey(struct crypto_aead *aead, const u8 *key,
			     unsigned int keylen)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(aead);
	int err = 0;

	if (keylen != AESGCM_KEYLEN) {
		printk(KERN_ERR "Wrong key size.. need %u, got %u\n", AESGCM_KEYLEN, keylen);
		goto out;
	}

	lake_AES_GCM_setkey(&ctx->cuda_ctx, key);
	err = crypto_aead_setkey(ctx->aesni_tfm, key, 32);
	if (err) {
		printk(KERN_ERR "err setkey\n");
		return err;
	}
out:
	return err;
}

static int crypto_gcm_setauthsize(struct crypto_aead *tfm,
				  unsigned int authsize)
{
	switch (authsize) {
	case 16:
		break;
	case 4:
	case 8:
	case 12:
	case 13:
	case 14:
	case 15:
	default:
		return -EINVAL;
	}

	return 0;
}

static int crypto_gcm_encrypt(struct aead_request *req)
{
	// when we get here, all fields in req were set by aead_request_set_crypt
	// ->src, ->dst, ->cryptlen, ->iv
	struct crypto_aead *tfm = crypto_aead_reqtfm(req);
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	struct aead_request **aead_req = NULL;
	int count_dst = 0, lake_count = 0;
	void* buf;
	struct scatterlist *src_sg = req->src;
	struct scatterlist *dst_sg = req->dst;
	CUdeviceptr d_src = ctx->cuda_ctx.d_src;
	CUdeviceptr d_dst = ctx->cuda_ctx.d_dst;
	char *pages_buf, *bad_iv;
	int npages, i;
	int *rcs;
	int aesni_n, lake_n;
	struct extent_crypt_result *ecrs;

	npages = sg_nents(src_sg);
	if (sg_nents(dst_sg) != 2*npages) {
		printk(KERN_ERR "encrypt: error, wrong number of ents on sgs. src: %d, dst: %d\n", npages, sg_nents(dst_sg));
		return -1;
	}

	aesni_n = get_aesni_fraction(npages);
	lake_n = npages - aesni_n;
	DBG_PRINT("encrypt: processing %d pages. %d on aesni, %d on gpu\n", npages, aesni_n, lake_n);

	if (lake_n > 0) {
		pages_buf = (char *)kava_alloc(lake_n*PAGE_SIZE);

		for(i = aesni_n ; i < npages ; i++) {
			buf = sg_virt(&src_sg[i]);
			memcpy(pages_buf+(lake_count*PAGE_SIZE), buf, PAGE_SIZE);
			lake_count++;	
		}
		//TODO: copy IVs, set enc to use it. it's currently constant and set at setkey
		lake_AES_GCM_copy_to_device(d_src, pages_buf, lake_count*PAGE_SIZE);
		lake_AES_GCM_encrypt(&ctx->cuda_ctx, d_dst, d_src, lake_count*PAGE_SIZE);
	}

	if (aesni_n > 0) {
		// GPU is doing work, lets do AESNI now
		// GCM can't do multiple blocks in one request..
		aead_req = vmalloc(aesni_n * sizeof(struct aead_request*));
		//ignore iv for now
		bad_iv = vmalloc(12);
		ecrs = vmalloc(aesni_n * sizeof(struct extent_crypt_result));
		rcs = vmalloc(aesni_n * sizeof(int));

		for(i = 0 ; i < 12 ; i++)
			bad_iv[i] = i;

		for(i = 0 ; i < aesni_n ;i++) {
			aead_req[i] = aead_request_alloc(ctx->aesni_tfm, GFP_NOFS);
			if (!aead_req[i]) {
				printk(KERN_ERR "err aead_request_alloc\n");
				return -1;
			}
			init_completion(&ecrs[i].completion);
			aead_request_set_callback(aead_req[i],
					CRYPTO_TFM_REQ_MAY_BACKLOG | CRYPTO_TFM_REQ_MAY_SLEEP,
					extent_crypt_complete, &ecrs[i]);
			//TODO: use req->iv
			aead_request_set_crypt(aead_req[i], &src_sg[i], &dst_sg[i*2], PAGE_SIZE, bad_iv);
			aead_request_set_ad(aead_req[i], 0);
			rcs[i] = crypto_aead_encrypt(aead_req[i]);
		}
	}

	if (lake_n > 0) {
		//copy cipher back
		lake_AES_GCM_copy_from_device(pages_buf, d_dst, lake_count*PAGE_SIZE);
		//TODO: copy back MACs
		cuCtxSynchronize();

		for(i = aesni_n ; i < npages ; i++) {
			// cipher sg
			buf = sg_virt(&dst_sg[i*2]);
			memcpy(buf, pages_buf+(count_dst * PAGE_SIZE), PAGE_SIZE);
			//TODO: copy MAC
			//memcpy(buf, pages_buf+((count_dst*PAGE_SIZE) + PAGE_SIZE), crypto_aead_aes256gcm_ABYTES);
			count_dst++;
		}
		kava_free(pages_buf);
	}

	if (aesni_n > 0) {
		for(i = 0 ; i < aesni_n ; i++) {
			if (rcs[i] == -EINPROGRESS || rcs[i] == -EBUSY) {
				printk(KERN_ERR "waiting for enc req %d\n", i);
				wait_for_completion(&ecrs[i].completion);
			} 
			else if (rcs[i] == 0 || rcs[i] == -EBADMSG) {
				//ignore
			} 
			else {
				printk(KERN_ERR "decrypt error: %d\n", rcs[i]);
				return -1;
			} 
			aead_request_free(aead_req[i]);
		}
		vfree(rcs);
		vfree(aead_req);
		vfree(bad_iv);
		vfree(ecrs);
	}
	return 0;
}

static int crypto_gcm_decrypt(struct aead_request *req)
{
	struct crypto_aead *tfm = crypto_aead_reqtfm(req);
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	struct aead_request **aead_req = NULL;
	int count_dst = 0, lake_count = 0;
	void* buf;
	struct scatterlist *src_sg = req->src; 
	struct scatterlist *dst_sg = req->dst; 
	CUdeviceptr d_src = ctx->cuda_ctx.d_src;
	CUdeviceptr d_dst = ctx->cuda_ctx.d_dst;
	char *pages_buf, *bad_iv;
	int npages, i;
	int *rcs;
	int aesni_n, lake_n;
	struct extent_crypt_result *ecrs;

	npages = sg_nents(src_sg);
	if (2*sg_nents(dst_sg) != npages) {
		printk(KERN_ERR "decrypt: error, wrong number of ents on sgs. src: %d, dst: %d\n", npages, sg_nents(dst_sg));
		return -1;
	}
	npages = npages/2;

	aesni_n = get_aesni_fraction(npages);
	lake_n = npages - aesni_n;
	DBG_PRINT("decrypt: processing %d pages. %d on aesni" 
		"%d on gpu\n", npages, aesni_n, lake_n);

	if (lake_n > 0) {
		pages_buf = (char *)kava_alloc(lake_n*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
		if(!pages_buf) {
			printk(KERN_ERR "decrypt: error allocating %ld bytes\n", 
				lake_n*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
			return -1;
		}

		for(i = aesni_n ; i < npages ; i++) {
			buf = sg_virt(&src_sg[i*2]);	
			memcpy(pages_buf+(lake_count*PAGE_SIZE), buf, PAGE_SIZE);
			//TODO: copy MACs sg_virt(&src_sg[i*2+1]);	
			lake_count++; 
		}

		lake_AES_GCM_copy_to_device(d_src, pages_buf, lake_n*PAGE_SIZE);
		//TODO: copy MACs too
		lake_AES_GCM_decrypt(&ctx->cuda_ctx, d_dst, d_src, lake_n*PAGE_SIZE);
	}

	if (aesni_n > 0) {
		// GPU is doing work, lets do AESNI now
		// GCM can't do multiple blocks in one request..
		aead_req = vmalloc(aesni_n * sizeof(struct aead_request*));
		//ignore iv for now
		bad_iv = vmalloc(12);
		ecrs = vmalloc(aesni_n * sizeof(struct extent_crypt_result));
		rcs = vmalloc(aesni_n * sizeof(int));

		for(i = 0 ; i < 12 ; i++)
			bad_iv[i] = i;

		for(i = 0 ; i < aesni_n ;i++) {
			aead_req[i] = aead_request_alloc(ctx->aesni_tfm, GFP_NOFS);
			if (!aead_req[i]) {
				printk(KERN_ERR "err aead_request_alloc\n");
				return -1;
			}
			init_completion(&ecrs[i].completion);
			aead_request_set_callback(aead_req[i],
					CRYPTO_TFM_REQ_MAY_BACKLOG | CRYPTO_TFM_REQ_MAY_SLEEP,
					extent_crypt_complete, &ecrs[i]);
			//TODO: use req->iv
			aead_request_set_crypt(aead_req[i], &src_sg[i*2], &dst_sg[i], PAGE_SIZE+crypto_aead_aes256gcm_ABYTES, bad_iv);
			aead_request_set_ad(aead_req[i], 0);
			rcs[i] = crypto_aead_decrypt(aead_req[i]);
		}
	}

	if (lake_n > 0) {
		//copy cipher back
		lake_AES_GCM_copy_from_device(pages_buf, d_dst, lake_n*PAGE_SIZE);
		cuCtxSynchronize();

		for(i = aesni_n ; i < npages ; i++) {
			// plain sg
			buf = sg_virt(&dst_sg[i]);
			memcpy(buf, pages_buf+(count_dst * PAGE_SIZE), PAGE_SIZE);
			count_dst++;
		}
		kava_free(pages_buf);
	}
	
	if (aesni_n > 0) {
		for(i = 0 ; i < aesni_n ; i++) {
			if (rcs[i] == -EINPROGRESS || rcs[i] == -EBUSY) {
				printk(KERN_ERR "waiting for enc req %d\n", i);
				wait_for_completion(&ecrs[i].completion);
			} 
			else if (rcs[i] == 0 || rcs[i] == -EBADMSG) {
				//ignore
			} 
			else {
				printk(KERN_ERR "decrypt error: %d\n", rcs[i]);
				return -1;
			} 
			aead_request_free(aead_req[i]);
		}
		vfree(rcs);
		vfree(aead_req);
		vfree(bad_iv);
		vfree(ecrs);
	}
	return 0;
}

static int crypto_gcm_init_tfm(struct crypto_aead *tfm)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	unsigned long align;

	align = crypto_aead_alignmask(tfm);
	align &= ~(crypto_tfm_ctx_alignment() - 1);
	crypto_aead_set_reqsize(tfm, align);

	lake_AES_GCM_init_fns(&ctx->cuda_ctx, cubin_path);
	lake_AES_GCM_init(&ctx->cuda_ctx);

	if (aesni_fraction > 100)
		aesni_fraction = 100;
	if (aesni_fraction < 0)
		aesni_fraction = 0;

	ctx->aesni_tfm = crypto_alloc_aead("generic-gcm-aesni", 0, 0);
	if (IS_ERR(ctx->aesni_tfm)) {
		printk(KERN_ERR "Error allocating generic-gcm-aesni %ld\n", PTR_ERR(ctx->aesni_tfm));
		return -ENOENT;
	}
	
	return 0;
}

static void crypto_gcm_exit_tfm(struct crypto_aead *tfm)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	lake_AES_GCM_destroy(&ctx->cuda_ctx);


	crypto_free_aead(ctx->aesni_tfm);
}

static void crypto_gcm_free(struct aead_instance *inst)
{
	kfree(inst);
}

static int crypto_gcm_create_common(struct crypto_template *tmpl,
				    struct rtattr **tb)
{
	struct aead_instance *inst;
	int err;

	err = -ENOMEM;
	inst = kzalloc(sizeof(*inst), GFP_KERNEL);
	if (!inst)
		goto out_err;

	snprintf(inst->alg.base.cra_name, CRYPTO_MAX_ALG_NAME, "lake_gcm(aes)");
	snprintf(inst->alg.base.cra_driver_name, CRYPTO_MAX_ALG_NAME, "lake(gcm_cuda,aes)");

	inst->alg.base.cra_flags = CRYPTO_ALG_ASYNC;
	//inst->alg.base.cra_priority = (ghash->base.cra_priority +
	//			       ctr->base.cra_priority) / 2;
	inst->alg.base.cra_priority = 100;
	inst->alg.base.cra_blocksize = 1;
	//XXX
	//inst->alg.base.cra_alignmask = ghash->base.cra_alignmask |
	//			       ctr->base.cra_alignmask;
	inst->alg.base.cra_alignmask = 0;
	inst->alg.base.cra_ctxsize = sizeof(struct crypto_gcm_ctx);
	inst->alg.ivsize = GCM_AES_IV_SIZE;

	//XXX
	//inst->alg.chunksize = crypto_skcipher_alg_chunksize(ctr);
	inst->alg.chunksize = 1;

	inst->alg.maxauthsize = 16;
	inst->alg.init = crypto_gcm_init_tfm;
	inst->alg.exit = crypto_gcm_exit_tfm;
	inst->alg.setkey = crypto_gcm_setkey;
	inst->alg.setauthsize = crypto_gcm_setauthsize;
	inst->alg.encrypt = crypto_gcm_encrypt;
	inst->alg.decrypt = crypto_gcm_decrypt;

	inst->free = crypto_gcm_free;

	err = aead_register_instance(tmpl, inst);
	if (err) {
		printk(KERN_ERR "error aead_register_instance %d\n", err);
		goto out_err;
	}
	return err;
out_err:
	printk(KERN_ERR "error in crypto_gcm_create_common %d\n", err);
	kfree(inst);
	return err;
}

static int crypto_gcm_create(struct crypto_template *tmpl, struct rtattr **tb)
{
	return crypto_gcm_create_common(tmpl, tb);
}

static struct crypto_template crypto_gcm_tmpl = {
	.name = "lake_gcm",
	.create = crypto_gcm_create,
	.module = THIS_MODULE,
};

static int __init crypto_gcm_module_init(void)
{
	int err;
	struct file *f;

	f = filp_open(cubin_path, O_RDONLY, 0600);
	if (IS_ERR(f) || !f) {
		printk(KERN_ERR "cant open cubin file at %s\n", cubin_path);
		return -2;
	}
	printk(KERN_ERR "cubin found at %s\n", cubin_path);
	filp_close(f, 0);

	err = crypto_register_template(&crypto_gcm_tmpl);
	if (err)
		goto out_undo_gcm;
	printk(KERN_ERR "lake_gcm crypto template registered.\n");
	return 0;

out_undo_gcm:
	printk(KERN_ERR "error registering template\n");
	crypto_unregister_template(&crypto_gcm_tmpl);
	return err;
}

static void __exit crypto_gcm_module_exit(void)
{
	crypto_unregister_template(&crypto_gcm_tmpl);
}

module_init(crypto_gcm_module_init);
module_exit(crypto_gcm_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Galois/Counter Mode using CUDA");
MODULE_AUTHOR("Henrique Fingler");
MODULE_ALIAS_CRYPTO("lake_gcm");
