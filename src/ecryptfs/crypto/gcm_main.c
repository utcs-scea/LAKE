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
#include "internal.h"
#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/fs.h> 
#include <asm/uaccess.h>

#include "gcm_cuda.h"

static char *cubin_path = "gcm_kernels.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to gcm_kernels.cubin");

//tfm ctx
struct crypto_gcm_ctx {
	struct AES_GCM_engine_ctx cuda_ctx;
};

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
	int count = 0, count_dst = 0;
	void* buf;
	unsigned int len;
	struct scatterlist *src_sg = req->src; 
	struct scatterlist *dst_sg = req->dst; 
	CUdeviceptr d_src, d_dst;
	char *pages_buf;
	int npages;

	npages = sg_nents(src_sg);
	if (sg_nents(dst_sg) != 2*npages) {
		printk(KERN_ERR "encrypt: error, wrong number of ents on sgs. src: %d, dst: %d\n", npages, sg_nents(dst_sg));
		return -1;
	}

	printk(KERN_ERR "encrypt: processing %d pages\n", npages);
	lake_AES_GCM_alloc_pages(&d_src, npages*PAGE_SIZE);
	lake_AES_GCM_alloc_pages(&d_dst, npages*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
	//TODO: switch between these
	//pages_buf = (char *)kava_alloc(nbytes);
	pages_buf = vmalloc(npages*PAGE_SIZE);

	while(src_sg) {
		buf = sg_virt(src_sg);
		len = src_sg->length;
		//printk(KERN_ERR "encrypt: processing sg input #%d w/ size %u\n", count, len);
		printk(KERN_ERR "memcpy..\n");
		memcpy(pages_buf+(count*PAGE_SIZE), buf, PAGE_SIZE);
		printk(KERN_ERR "ok\n");
		src_sg = sg_next(src_sg);
		count++;
	}
	printk(KERN_ERR "src sg done\n");
	//TODO: copy IVs, set enc to use it. it's currently constant and set at setkey
	lake_AES_GCM_copy_to_device(d_src, pages_buf, npages*PAGE_SIZE);
	lake_AES_GCM_encrypt(&ctx->cuda_ctx, d_dst, d_src, npages*PAGE_SIZE);
	//copy cipher back
	lake_AES_GCM_copy_from_device(pages_buf, d_dst, npages*PAGE_SIZE);
	//TODO: copy MAC
	cuCtxSynchronize();

	while(dst_sg) {
		// cipher sg
		buf = sg_virt(dst_sg);
		printk(KERN_ERR "memcpy dst\n");
		//memcpy(buf, pages_buf+(count_dst * (PAGE_SIZE+crypto_aead_aes256gcm_ABYTES)), PAGE_SIZE);
		memcpy(buf, pages_buf+(count_dst * PAGE_SIZE), PAGE_SIZE);
		printk(KERN_ERR "ok\n");
		// MAC sg
		dst_sg = sg_next(dst_sg);
		//TODO copy MAC
		//buf = sg_virt(dst_sg);
		//memcpy(buf, pages_buf+((count_dst*PAGE_SIZE) + PAGE_SIZE), crypto_aead_aes256gcm_ABYTES);
		dst_sg = sg_next(dst_sg);
	}

	lake_AES_GCM_free(d_src);
	lake_AES_GCM_free(d_dst);
	vfree(pages_buf);
	return 0;
}

static int crypto_gcm_decrypt(struct aead_request *req)
{
	struct crypto_aead *tfm = crypto_aead_reqtfm(req);
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	int count = 0, count_dst = 0;
	void* buf;
	unsigned int len;
	struct scatterlist *src_sg = req->src; 
	struct scatterlist *dst_sg = req->dst; 
	CUdeviceptr d_src, d_dst;
	char *pages_buf;
	int npages;

	npages = sg_nents(src_sg);
	if (2*sg_nents(dst_sg) != npages) {
		printk(KERN_ERR "encrypt: error, wrong number of ents on sgs. src: %d, dst: %d\n", npages, sg_nents(dst_sg));
		return -1;
	}

	printk(KERN_ERR "decrypt: processing %d pages\n", npages);
	lake_AES_GCM_alloc_pages(&d_src, npages*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
	lake_AES_GCM_alloc_pages(&d_dst, npages*PAGE_SIZE);
	//TODO: switch between these
	//pages_buf = (char *)kava_alloc(nbytes);
	pages_buf = vmalloc(npages*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
	if(!pages_buf) {
		printk(KERN_ERR "decrypt: error allocating %d bytes\n", 
			npages*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
		return -1;
	}

	while(src_sg) {
		buf = sg_virt(src_sg);	
		len = src_sg->length;
		//printk(KERN_ERR "decrypt: processing sg input #%d w/ size %u\n", count, len);
		memcpy(pages_buf+(count*PAGE_SIZE), buf, PAGE_SIZE);
		src_sg = sg_next(src_sg);
		//TODO: copy MACs
		src_sg = sg_next(src_sg);
		count++;
	}
	lake_AES_GCM_copy_to_device(d_src, pages_buf, npages*PAGE_SIZE);
	//TODO: copy MACs too
	lake_AES_GCM_decrypt(&ctx->cuda_ctx, d_dst, d_src, npages*PAGE_SIZE);
	//copy cipher back
	lake_AES_GCM_copy_from_device(pages_buf, d_dst, npages*PAGE_SIZE);
	cuCtxSynchronize();
	while(dst_sg) {
		// plain sg
		buf = sg_virt(dst_sg);
		memcpy(buf, pages_buf+(count_dst*PAGE_SIZE), PAGE_SIZE);
		dst_sg = sg_next(dst_sg);
		count_dst++;
	}
	vfree(pages_buf);
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

	return 0;
}

static void crypto_gcm_exit_tfm(struct crypto_aead *tfm)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	lake_AES_GCM_destroy(&ctx->cuda_ctx);
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
	usleep_range(50, 100);
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
out:
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
