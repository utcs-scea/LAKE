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
#include <crypto/scatterlist.h>
#include <crypto/gcm.h>
#include <crypto/hash.h>
#include "internal.h"
#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>

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
	case 4:
	case 8:
	case 12:
	case 13:
	case 14:
	case 15:
	case 16:
		break;
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
	int count = 0;	
	void* buf;
	unsigned int len;
	struct scatterlist *src = req->src; 
	CUdeviceptr src, dst;

	//TODO: figure out total # of pages
	u32 npages;
	lake_AES_GCM_alloc_pages(&ctx->cuda_ctx, &src, npages*PAGE_SIZE);
	lake_AES_GCM_alloc_pages(&ctx->cuda_ctx, &dst, npages*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
	while(src) {
		buf = sg_virt(src);	
		len = src->length;
		printk(KERN_ERR "encrypt: processing sg input #%d w/ size %u\n", count, len);
		src = sg_next(src);
		//copy to device
		lake_AES_GCM_copy_to_device(&ctx->cuda_ctx, src, count, PAGE_SIZE);
	}

	void lake_AES_GCM_encrypt(&ctx->cuda_ctx, dst, src, npages*PAGE_SIZE);

	cuCtxSynchronize();
	return 0;
}

static int crypto_gcm_decrypt(struct aead_request *req)
{
	struct crypto_aead *tfm = crypto_aead_reqtfm(req);

	unsigned int authsize = crypto_aead_authsize(tfm);
	unsigned int cryptlen = req->cryptlen;
	
	// when we get here, all fields in req were set by aead_request_set_crypt
	// ->src, ->dst, ->cryptlen, ->iv

	// gctx->src = sg_next(pctx->src);

	return 0;
}

static int crypto_gcm_init_tfm(struct crypto_aead *tfm)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	unsigned long align;

	align = crypto_aead_alignmask(tfm);
	align &= ~(crypto_tfm_ctx_alignment() - 1);
	crypto_aead_set_reqsize(tfm, align);

	lake_AES_GCM_init(&ctx->cuda_ctx);
	lake_init_fns(&ctx->cuda_ctx, cubin_path);

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
	//struct gcm_instance_ctx *ctx;
	struct aead_instance *inst;
	int err;

	err = -ENOMEM;
	inst = kzalloc(sizeof(*inst), GFP_KERNEL);
	if (!inst)
		goto out_put_ghash;

	//ctx = aead_instance_ctx(inst);
	
	//inst->alg.base.cra_flags = (ghash->base.cra_flags |
	//			    ctr->base.cra_flags) & CRYPTO_ALG_ASYNC;
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
	if (err)
		goto err_free_inst;

out_put_ghash:
	//crypto_mod_put(ghash_alg);
	return err;

//out_put_ctr:
	//crypto_drop_skcipher(&ctx->ctr);
//err_drop_ghash:
	//crypto_drop_ahash(&ctx->ghash);
err_free_inst:
	kfree(inst);
	goto out_put_ghash;
}

static int crypto_gcm_create(struct crypto_template *tmpl, struct rtattr **tb)
{
	return crypto_gcm_create_common(tmpl, tb);
}

static struct crypto_template crypto_gcm_tmpl = {
	.name = "gcm_cuda",
	.create = crypto_gcm_create,
	.module = THIS_MODULE,
};

static int __init crypto_gcm_module_init(void)
{
	int err;
	err = crypto_register_template(&crypto_gcm_tmpl);
	if (err)
		goto out_undo_gcm;

	return 0;

out_undo_gcm:
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
MODULE_DESCRIPTION("Galois/Counter Mode");
MODULE_AUTHOR("Mikko Herranen <mh1@iki.fi>");
MODULE_ALIAS_CRYPTO("lake-gcm");
