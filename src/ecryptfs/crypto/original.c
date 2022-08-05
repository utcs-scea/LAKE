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
#include <crypto/scatterwalk.h>
#include <crypto/gcm.h>
#include <crypto/hash.h>
#include "internal.h"
#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>

/*
struct aead_request {
	struct crypto_async_request base;

	unsigned int assoclen;
	unsigned int cryptlen;

	u8 *iv;

	struct scatterlist *src;
	struct scatterlist *dst;

	void *__ctx[] CRYPTO_MINALIGN_ATTR;
};
*/

struct gcm_instance_ctx {
	struct crypto_skcipher_spawn ctr;
	struct crypto_ahash_spawn ghash;
};

struct crypto_gcm_ctx {
	struct crypto_skcipher *ctr;
	struct crypto_ahash *ghash;
};

struct crypto_gcm_ghash_ctx {
	unsigned int cryptlen;
	struct scatterlist *src;
	int (*complete)(struct aead_request *req, u32 flags);
};

struct crypto_gcm_req_priv_ctx {
	u8 iv[16];
	u8 auth_tag[16];
	u8 iauth_tag[16];
	struct scatterlist src[3];
	struct scatterlist dst[3];
	struct scatterlist sg;
	struct crypto_gcm_ghash_ctx ghash_ctx;
	union {
		struct ahash_request ahreq;
		struct skcipher_request skreq;
	} u;
};

static struct {
	u8 buf[16];
	struct scatterlist sg;
} *gcm_zeroes;

static inline struct crypto_gcm_req_priv_ctx *crypto_gcm_reqctx(
	struct aead_request *req)
{
	unsigned long align = crypto_aead_alignmask(crypto_aead_reqtfm(req));

	return (void *)PTR_ALIGN((u8 *)aead_request_ctx(req), align + 1);
}

static int crypto_gcm_setkey(struct crypto_aead *aead, const u8 *key,
			     unsigned int keylen)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(aead);
	struct crypto_ahash *ghash = ctx->ghash;
	struct crypto_skcipher *ctr = ctx->ctr;
	struct {
		be128 hash;
		u8 iv[16];

		struct crypto_wait wait;

		struct scatterlist sg[1];
		struct skcipher_request req;
	} *data;
	int err;

	crypto_skcipher_clear_flags(ctr, CRYPTO_TFM_REQ_MASK);
	crypto_skcipher_set_flags(ctr, crypto_aead_get_flags(aead) &
				       CRYPTO_TFM_REQ_MASK);
	err = crypto_skcipher_setkey(ctr, key, keylen);
	crypto_aead_set_flags(aead, crypto_skcipher_get_flags(ctr) &
				    CRYPTO_TFM_RES_MASK);
	if (err)
		return err;

	data = kzalloc(sizeof(*data) + crypto_skcipher_reqsize(ctr),
		       GFP_KERNEL);
	if (!data)
		return -ENOMEM;

	crypto_init_wait(&data->wait);
	sg_init_one(data->sg, &data->hash, sizeof(data->hash));
	skcipher_request_set_tfm(&data->req, ctr);
	skcipher_request_set_callback(&data->req, CRYPTO_TFM_REQ_MAY_SLEEP |
						  CRYPTO_TFM_REQ_MAY_BACKLOG,
				      crypto_req_done,
				      &data->wait);
	skcipher_request_set_crypt(&data->req, data->sg, data->sg,
				   sizeof(data->hash), data->iv);

	err = crypto_wait_req(crypto_skcipher_encrypt(&data->req),
							&data->wait);

	if (err)
		goto out;

	crypto_ahash_clear_flags(ghash, CRYPTO_TFM_REQ_MASK);
	crypto_ahash_set_flags(ghash, crypto_aead_get_flags(aead) &
			       CRYPTO_TFM_REQ_MASK);
	err = crypto_ahash_setkey(ghash, (u8 *)&data->hash, sizeof(be128));
	crypto_aead_set_flags(aead, crypto_ahash_get_flags(ghash) &
			      CRYPTO_TFM_RES_MASK);

out:
	kzfree(data);
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

static void crypto_gcm_init_common(struct aead_request *req)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	__be32 counter = cpu_to_be32(1);
	struct scatterlist *sg;

	memset(pctx->auth_tag, 0, sizeof(pctx->auth_tag));
	memcpy(pctx->iv, req->iv, GCM_AES_IV_SIZE);
	memcpy(pctx->iv + GCM_AES_IV_SIZE, &counter, 4);

	sg_init_table(pctx->src, 3);
	sg_set_buf(pctx->src, pctx->auth_tag, sizeof(pctx->auth_tag));
	sg = scatterwalk_ffwd(pctx->src + 1, req->src, req->assoclen);
	if (sg != pctx->src + 1)
		sg_chain(pctx->src, 2, sg);

	if (req->src != req->dst) {
		sg_init_table(pctx->dst, 3);
		sg_set_buf(pctx->dst, pctx->auth_tag, sizeof(pctx->auth_tag));
		sg = scatterwalk_ffwd(pctx->dst + 1, req->dst, req->assoclen);
		if (sg != pctx->dst + 1)
			sg_chain(pctx->dst, 2, sg);
	}
}

static void crypto_gcm_init_crypt(struct aead_request *req,
				  unsigned int cryptlen)
{
	struct crypto_aead *aead = crypto_aead_reqtfm(req);
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(aead);
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct skcipher_request *skreq = &pctx->u.skreq;
	struct scatterlist *dst;

	dst = req->src == req->dst ? pctx->src : pctx->dst;

	skcipher_request_set_tfm(skreq, ctx->ctr);
	skcipher_request_set_crypt(skreq, pctx->src, dst,
				     cryptlen + sizeof(pctx->auth_tag),
				     pctx->iv);
}

static inline unsigned int gcm_remain(unsigned int len)
{
	len &= 0xfU;
	return len ? 16 - len : 0;
}

static void gcm_hash_len_done(struct crypto_async_request *areq, int err);

static int gcm_hash_update(struct aead_request *req,
			   crypto_completion_t compl,
			   struct scatterlist *src,
			   unsigned int len, u32 flags)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct ahash_request *ahreq = &pctx->u.ahreq;

	ahash_request_set_callback(ahreq, flags, compl, req);
	ahash_request_set_crypt(ahreq, src, NULL, len);

	return crypto_ahash_update(ahreq);
}

static int gcm_hash_remain(struct aead_request *req,
			   unsigned int remain,
			   crypto_completion_t compl, u32 flags)
{
	return gcm_hash_update(req, compl, &gcm_zeroes->sg, remain, flags);
}

static int gcm_hash_len(struct aead_request *req, u32 flags)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct ahash_request *ahreq = &pctx->u.ahreq;
	struct crypto_gcm_ghash_ctx *gctx = &pctx->ghash_ctx;
	u128 lengths;

	lengths.a = cpu_to_be64(req->assoclen * 8);
	lengths.b = cpu_to_be64(gctx->cryptlen * 8);
	memcpy(pctx->iauth_tag, &lengths, 16);
	sg_init_one(&pctx->sg, pctx->iauth_tag, 16);
	ahash_request_set_callback(ahreq, flags, gcm_hash_len_done, req);
	ahash_request_set_crypt(ahreq, &pctx->sg,
				pctx->iauth_tag, sizeof(lengths));

	return crypto_ahash_finup(ahreq);
}

static int gcm_hash_len_continue(struct aead_request *req, u32 flags)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct crypto_gcm_ghash_ctx *gctx = &pctx->ghash_ctx;

	return gctx->complete(req, flags);
}

static void gcm_hash_len_done(struct crypto_async_request *areq, int err)
{
	struct aead_request *req = areq->data;

	if (err)
		goto out;

	err = gcm_hash_len_continue(req, 0);
	if (err == -EINPROGRESS)
		return;

out:
	aead_request_complete(req, err);
}

static int gcm_hash_crypt_remain_continue(struct aead_request *req, u32 flags)
{
	return gcm_hash_len(req, flags) ?:
	       gcm_hash_len_continue(req, flags);
}

static void gcm_hash_crypt_remain_done(struct crypto_async_request *areq,
				       int err)
{
	struct aead_request *req = areq->data;

	if (err)
		goto out;

	err = gcm_hash_crypt_remain_continue(req, 0);
	if (err == -EINPROGRESS)
		return;

out:
	aead_request_complete(req, err);
}

static int gcm_hash_crypt_continue(struct aead_request *req, u32 flags)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct crypto_gcm_ghash_ctx *gctx = &pctx->ghash_ctx;
	unsigned int remain;

	remain = gcm_remain(gctx->cryptlen);
	if (remain)
		return gcm_hash_remain(req, remain,
				       gcm_hash_crypt_remain_done, flags) ?:
		       gcm_hash_crypt_remain_continue(req, flags);

	return gcm_hash_crypt_remain_continue(req, flags);
}

static void gcm_hash_crypt_done(struct crypto_async_request *areq, int err)
{
	struct aead_request *req = areq->data;

	if (err)
		goto out;

	err = gcm_hash_crypt_continue(req, 0);
	if (err == -EINPROGRESS)
		return;

out:
	aead_request_complete(req, err);
}

static int gcm_hash_assoc_remain_continue(struct aead_request *req, u32 flags)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct crypto_gcm_ghash_ctx *gctx = &pctx->ghash_ctx;

	if (gctx->cryptlen)
		return gcm_hash_update(req, gcm_hash_crypt_done,
				       gctx->src, gctx->cryptlen, flags) ?:
		       gcm_hash_crypt_continue(req, flags);

	return gcm_hash_crypt_remain_continue(req, flags);
}

static void gcm_hash_assoc_remain_done(struct crypto_async_request *areq,
				       int err)
{
	struct aead_request *req = areq->data;

	if (err)
		goto out;

	err = gcm_hash_assoc_remain_continue(req, 0);
	if (err == -EINPROGRESS)
		return;

out:
	aead_request_complete(req, err);
}

static int gcm_hash_assoc_continue(struct aead_request *req, u32 flags)
{
	unsigned int remain;

	remain = gcm_remain(req->assoclen);
	if (remain)
		return gcm_hash_remain(req, remain,
				       gcm_hash_assoc_remain_done, flags) ?:
		       gcm_hash_assoc_remain_continue(req, flags);

	return gcm_hash_assoc_remain_continue(req, flags);
}

static void gcm_hash_assoc_done(struct crypto_async_request *areq, int err)
{
	struct aead_request *req = areq->data;

	if (err)
		goto out;

	err = gcm_hash_assoc_continue(req, 0);
	if (err == -EINPROGRESS)
		return;

out:
	aead_request_complete(req, err);
}

static int gcm_hash_init_continue(struct aead_request *req, u32 flags)
{
	if (req->assoclen)
		return gcm_hash_update(req, gcm_hash_assoc_done,
				       req->src, req->assoclen, flags) ?:
		       gcm_hash_assoc_continue(req, flags);

	return gcm_hash_assoc_remain_continue(req, flags);
}

static void gcm_hash_init_done(struct crypto_async_request *areq, int err)
{
	struct aead_request *req = areq->data;

	if (err)
		goto out;

	err = gcm_hash_init_continue(req, 0);
	if (err == -EINPROGRESS)
		return;

out:
	aead_request_complete(req, err);
}

static int gcm_hash(struct aead_request *req, u32 flags)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct ahash_request *ahreq = &pctx->u.ahreq;
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(crypto_aead_reqtfm(req));

	ahash_request_set_tfm(ahreq, ctx->ghash);

	ahash_request_set_callback(ahreq, flags, gcm_hash_init_done, req);
	return crypto_ahash_init(ahreq) ?:
	       gcm_hash_init_continue(req, flags);
}

static int gcm_enc_copy_hash(struct aead_request *req, u32 flags)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct crypto_aead *aead = crypto_aead_reqtfm(req);
	u8 *auth_tag = pctx->auth_tag;

	crypto_xor(auth_tag, pctx->iauth_tag, 16);
	scatterwalk_map_and_copy(auth_tag, req->dst,
				 req->assoclen + req->cryptlen,
				 crypto_aead_authsize(aead), 1);
	return 0;
}

static int gcm_encrypt_continue(struct aead_request *req, u32 flags)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct crypto_gcm_ghash_ctx *gctx = &pctx->ghash_ctx;

	gctx->src = sg_next(req->src == req->dst ? pctx->src : pctx->dst);
	gctx->cryptlen = req->cryptlen;
	gctx->complete = gcm_enc_copy_hash;

	return gcm_hash(req, flags);
}

static void gcm_encrypt_done(struct crypto_async_request *areq, int err)
{
	struct aead_request *req = areq->data;

	if (err)
		goto out;

	err = gcm_encrypt_continue(req, 0);
	if (err == -EINPROGRESS)
		return;

out:
	aead_request_complete(req, err);
}

static int crypto_gcm_encrypt(struct aead_request *req)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct skcipher_request *skreq = &pctx->u.skreq;
	u32 flags = aead_request_flags(req);

	crypto_gcm_init_common(req);
	crypto_gcm_init_crypt(req, req->cryptlen);
	skcipher_request_set_callback(skreq, flags, gcm_encrypt_done, req);

	return crypto_skcipher_encrypt(skreq) ?:
	       gcm_encrypt_continue(req, flags);
}

static int crypto_gcm_verify(struct aead_request *req)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct crypto_aead *aead = crypto_aead_reqtfm(req);
	u8 *auth_tag = pctx->auth_tag;
	u8 *iauth_tag = pctx->iauth_tag;
	unsigned int authsize = crypto_aead_authsize(aead);
	unsigned int cryptlen = req->cryptlen - authsize;

	crypto_xor(auth_tag, iauth_tag, 16);
	scatterwalk_map_and_copy(iauth_tag, req->src,
				 req->assoclen + cryptlen, authsize, 0);
	return crypto_memneq(iauth_tag, auth_tag, authsize) ? -EBADMSG : 0;
}

static void gcm_decrypt_done(struct crypto_async_request *areq, int err)
{
	struct aead_request *req = areq->data;

	if (!err)
		err = crypto_gcm_verify(req);

	aead_request_complete(req, err);
}

static int gcm_dec_hash_continue(struct aead_request *req, u32 flags)
{
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct skcipher_request *skreq = &pctx->u.skreq;
	struct crypto_gcm_ghash_ctx *gctx = &pctx->ghash_ctx;

	crypto_gcm_init_crypt(req, gctx->cryptlen);
	skcipher_request_set_callback(skreq, flags, gcm_decrypt_done, req);
	return crypto_skcipher_decrypt(skreq) ?: crypto_gcm_verify(req);
}

static int crypto_gcm_decrypt(struct aead_request *req)
{
	struct crypto_aead *aead = crypto_aead_reqtfm(req);
	struct crypto_gcm_req_priv_ctx *pctx = crypto_gcm_reqctx(req);
	struct crypto_gcm_ghash_ctx *gctx = &pctx->ghash_ctx;
	unsigned int authsize = crypto_aead_authsize(aead);
	unsigned int cryptlen = req->cryptlen;
	u32 flags = aead_request_flags(req);

	cryptlen -= authsize;

	crypto_gcm_init_common(req);

	gctx->src = sg_next(pctx->src);
	gctx->cryptlen = cryptlen;
	gctx->complete = gcm_dec_hash_continue;

	return gcm_hash(req, flags);
}

static int crypto_gcm_init_tfm(struct crypto_aead *tfm)
{
	struct aead_instance *inst = aead_alg_instance(tfm);
	struct gcm_instance_ctx *ictx = aead_instance_ctx(inst);
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);
	struct crypto_skcipher *ctr;
	struct crypto_ahash *ghash;
	unsigned long align;
	int err;

	ghash = crypto_spawn_ahash(&ictx->ghash);
	if (IS_ERR(ghash))
		return PTR_ERR(ghash);

	ctr = crypto_spawn_skcipher(&ictx->ctr);
	err = PTR_ERR(ctr);
	if (IS_ERR(ctr))
		goto err_free_hash;

	ctx->ctr = ctr;
	ctx->ghash = ghash;

	align = crypto_aead_alignmask(tfm);
	align &= ~(crypto_tfm_ctx_alignment() - 1);
	crypto_aead_set_reqsize(tfm,
		align + offsetof(struct crypto_gcm_req_priv_ctx, u) +
		max(sizeof(struct skcipher_request) +
		    crypto_skcipher_reqsize(ctr),
		    sizeof(struct ahash_request) +
		    crypto_ahash_reqsize(ghash)));

	return 0;

err_free_hash:
	crypto_free_ahash(ghash);
	return err;
}

static void crypto_gcm_exit_tfm(struct crypto_aead *tfm)
{
	struct crypto_gcm_ctx *ctx = crypto_aead_ctx(tfm);

	crypto_free_ahash(ctx->ghash);
	crypto_free_skcipher(ctx->ctr);
}

static void crypto_gcm_free(struct aead_instance *inst)
{
	struct gcm_instance_ctx *ctx = aead_instance_ctx(inst);

	crypto_drop_skcipher(&ctx->ctr);
	crypto_drop_ahash(&ctx->ghash);
	kfree(inst);
}

static int crypto_gcm_create_common(struct crypto_template *tmpl,
				    struct rtattr **tb,
				    const char *ctr_name,
				    const char *ghash_name)
{
	struct crypto_attr_type *algt;
	struct aead_instance *inst;
	struct skcipher_alg *ctr;
	struct crypto_alg *ghash_alg;
	struct hash_alg_common *ghash;
	struct gcm_instance_ctx *ctx;
	int err;

	algt = crypto_get_attr_type(tb);
	if (IS_ERR(algt))
		return PTR_ERR(algt);

	if ((algt->type ^ CRYPTO_ALG_TYPE_AEAD) & algt->mask)
		return -EINVAL;

	ghash_alg = crypto_find_alg(ghash_name, &crypto_ahash_type,
				    CRYPTO_ALG_TYPE_HASH,
				    CRYPTO_ALG_TYPE_AHASH_MASK |
				    crypto_requires_sync(algt->type,
							 algt->mask));
	if (IS_ERR(ghash_alg))
		return PTR_ERR(ghash_alg);

	ghash = __crypto_hash_alg_common(ghash_alg);

	err = -ENOMEM;
	inst = kzalloc(sizeof(*inst) + sizeof(*ctx), GFP_KERNEL);
	if (!inst)
		goto out_put_ghash;

	ctx = aead_instance_ctx(inst);
	err = crypto_init_ahash_spawn(&ctx->ghash, ghash,
				      aead_crypto_instance(inst));
	if (err)
		goto err_free_inst;

	err = -EINVAL;
	if (strcmp(ghash->base.cra_name, "ghash") != 0 ||
	    ghash->digestsize != 16)
		goto err_drop_ghash;

	crypto_set_skcipher_spawn(&ctx->ctr, aead_crypto_instance(inst));
	err = crypto_grab_skcipher(&ctx->ctr, ctr_name, 0,
				   crypto_requires_sync(algt->type,
							algt->mask));
	if (err)
		goto err_drop_ghash;

	ctr = crypto_spawn_skcipher_alg(&ctx->ctr);

	/* The skcipher algorithm must be CTR mode, using 16-byte blocks. */
	err = -EINVAL;
	if (strncmp(ctr->base.cra_name, "ctr(", 4) != 0 ||
	    crypto_skcipher_alg_ivsize(ctr) != 16 ||
	    ctr->base.cra_blocksize != 1)
		goto out_put_ctr;

	err = -ENAMETOOLONG;
	if (snprintf(inst->alg.base.cra_name, CRYPTO_MAX_ALG_NAME,
		     "gcm(%s", ctr->base.cra_name + 4) >= CRYPTO_MAX_ALG_NAME)
		goto out_put_ctr;

	if (snprintf(inst->alg.base.cra_driver_name, CRYPTO_MAX_ALG_NAME,
		     "gcm_base(%s,%s)", ctr->base.cra_driver_name,
		     ghash_alg->cra_driver_name) >=
	    CRYPTO_MAX_ALG_NAME)
		goto out_put_ctr;

	inst->alg.base.cra_flags = (ghash->base.cra_flags |
				    ctr->base.cra_flags) & CRYPTO_ALG_ASYNC;
	inst->alg.base.cra_priority = (ghash->base.cra_priority +
				       ctr->base.cra_priority) / 2;
	inst->alg.base.cra_blocksize = 1;
	inst->alg.base.cra_alignmask = ghash->base.cra_alignmask |
				       ctr->base.cra_alignmask;
	inst->alg.base.cra_ctxsize = sizeof(struct crypto_gcm_ctx);
	inst->alg.ivsize = GCM_AES_IV_SIZE;
	inst->alg.chunksize = crypto_skcipher_alg_chunksize(ctr);
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
		goto out_put_ctr;

out_put_ghash:
	crypto_mod_put(ghash_alg);
	return err;

out_put_ctr:
	crypto_drop_skcipher(&ctx->ctr);
err_drop_ghash:
	crypto_drop_ahash(&ctx->ghash);
err_free_inst:
	kfree(inst);
	goto out_put_ghash;
}

static int crypto_gcm_create(struct crypto_template *tmpl, struct rtattr **tb)
{
	const char *cipher_name;
	char ctr_name[CRYPTO_MAX_ALG_NAME];

	cipher_name = crypto_attr_alg_name(tb[1]);
	if (IS_ERR(cipher_name))
		return PTR_ERR(cipher_name);

	if (snprintf(ctr_name, CRYPTO_MAX_ALG_NAME, "ctr(%s)", cipher_name) >=
	    CRYPTO_MAX_ALG_NAME)
		return -ENAMETOOLONG;

	return crypto_gcm_create_common(tmpl, tb, ctr_name, "ghash");
}

static struct crypto_template crypto_gcm_tmpl = {
	.name = "gcm",
	.create = crypto_gcm_create,
	.module = THIS_MODULE,
};

static int __init crypto_gcm_module_init(void)
{
	int err;

	gcm_zeroes = kzalloc(sizeof(*gcm_zeroes), GFP_KERNEL);
	if (!gcm_zeroes)
		return -ENOMEM;

	sg_init_one(&gcm_zeroes->sg, gcm_zeroes->buf, sizeof(gcm_zeroes->buf));

	err = crypto_register_template(&crypto_gcm_tmpl);
	if (err)
		goto out_undo_gcm;

	return 0;

out_undo_gcm:
	crypto_unregister_template(&crypto_gcm_tmpl);
out:
	kfree(gcm_zeroes);
	return err;
}

static void __exit crypto_gcm_module_exit(void)
{
	kfree(gcm_zeroes);
	crypto_unregister_template(&crypto_gcm_tmpl);
}

module_init(crypto_gcm_module_init);
module_exit(crypto_gcm_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Galois/Counter Mode");
MODULE_AUTHOR("Mikko Herranen <mh1@iki.fi>");
MODULE_ALIAS_CRYPTO("lake-gcm");
