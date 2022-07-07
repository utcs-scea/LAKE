#include <crypto/algapi.h>
#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <linux/completion.h>

#include "kava_gcm.h"
#include "aesni/aesni-intel_glue.h"

#include <cuda.h>
#include <debug.h>
#include <shared_memory.h>

struct AES_GCM_engine_device {
    uint8_t *sbox;
    uint8_t *rsbox;
    uint8_t *Rcon;
    uint8_t *key;
    uint8_t *aes_roundkey;
    uint8_t *gcm_h;

    uint64_t *HL;
    uint64_t *HH;
    uint64_t *HL_long;
    uint64_t *HH_long;
    uint64_t *HL_sqr_long;
    uint64_t *HH_sqr_long;
    uint64_t *gf_last4;

    uint8_t *buffer1;
    uint8_t *buffer2;

    uint8_t key[AESGCM_KEYLEN];
    uint8_t nonce_host[crypto_aead_aes256gcm_NPUBBYTES];
    uint8_t* nonce_device;
};

struct crypto_kava_ecb_inst {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction xcrypt_fn;
    CUfunction mac_fn;
    CUfunction key_expansion;
    CUfunction setup_table;
    CUfunction encrypt_oneblock;
    CUstream *g_stream;

    struct AES_GCM_engine_device d_engine;

    //aes-ni, fix later
    struct task_struct *aesni_kth;
    struct kthread_ecb_arg aesni_kargs;
    struct completion aesni_completion;
};
static struct crypto_kava_ecb_inst kava_ecb_inst;

struct crypto_kava_ecb_ctx {
	struct crypto_cipher *child;

    struct crypto_aes_ctx aes_ctx;
	u8 aesni_aes_ctx[sizeof(struct crypto_aes_ctx)] AESNI_ALIGN_ATTR;
    struct crypto_kava_ecb_inst *g_inst;
    //CUdeviceptr g_aes_ctx;
    //u32 key_round;
    //CUdeviceptr g_key_enc;
    //CUdeviceptr g_key_dec;
    CUdeviceptr g_buf;
};

static int split_threshold = 128;
module_param(split_threshold, int, 0444);
MODULE_PARM_DESC(split_threshold, "Threshold for data split, default 128 pages");

static int split_max = 8;
module_param(split_max, int, 0444);
MODULE_PARM_DESC(split_max, "Maximum number of split sub-requests, default 8");

static int aesni_fraction = 0;
module_param(aesni_fraction, int, 0444);
MODULE_PARM_DESC(aesni_fraction, "Fraction of the file to be encrypted using AES-NI (out of 100), default 0");

static volatile char dev_initialized = 0;
static volatile char aesni_crypt_called = 0;
static wait_queue_head_t aesni_crypt_wq_head;

static int crypto_kava_ecb_setkey(struct crypto_tfm *parent, const u8 *key,
			     unsigned int keylen)
{
	struct crypto_kava_ecb_ctx *ctx = crypto_tfm_ctx(parent);
	struct crypto_cipher *child = ctx->child;
	int err;

	crypto_cipher_clear_flags(child, CRYPTO_TFM_REQ_MASK);
	crypto_cipher_set_flags(child, crypto_tfm_get_flags(parent) &
				       CRYPTO_TFM_REQ_MASK);
	err = crypto_cipher_setkey(child, key, keylen);
    if (aesni_fraction > 0) {
        err = aes_set_key_common(parent, ctx->aesni_aes_ctx, key, keylen);
    }

#if KAVA_ENABLE
    // Expand key
    //err = crypto_aes_expand_key(&ctx->aes_ctx, key, keylen);
    //cvt_cpu_to_be32(ctx->aes_ctx.key_enc, AES_MAX_KEYLENGTH_U32);
    //cvt_cpu_to_be32(ctx->aes_ctx.key_dec, AES_MAX_KEYLENGTH_U32);
    //ctx->key_round = ctx->aes_ctx.key_length / 4 + 6;
    //cuMemcpyHtoD(ctx->g_aes_ctx, &(ctx->aes_ctx), sizeof(struct crypto_aes_ctx));

    //AES_key_expansion_kernel<<<numBlocks, dimBlocks>>>(d_engine.sbox, d_engine.Rcon, d_engine.key, d_engine.aes_roundkey);
    void *args[] = { &ctx->g_inst->d_engine.sbox, &ctx->g_inst->d_engine.Rcon, &ctx->g_inst->d_engine.key, &ctx->g_inst->d_engine.aes_roundkey};
    cuLaunchKernel(ctx->g_inst->key_expansion, 1, 1, 1, 1, 1, 1, 0, 0, args, NULL);

    //AES_encrypt_one_block_kernel<<<numBlocks, dimBlocks>>>(d_engine.sbox, d_engine.aes_roundkey, d_engine.gcm_h);
    void *args2[] = { &ctx->g_inst->d_engine.sbox, &ctx->g_inst->d_engine.aes_roundkey, &ctx->g_inst->d_engine.gcm_h};
    cuLaunchKernel(ctx->g_inst->encrypt_oneblock, 1, 1, 1, 1, 1, 1, 0, 0, args2, NULL);

    //cudaMemcpy(d_engine.gf_last4, gf_last4_host, AESGCM_BLOCK_SIZE, cudaMemcpyHostToDevice);
    cuMemcpyHtoD(ctx->g_inst->d_engine.gf_last4, gf_last4_host, BLOCK_SIZE);

    //AES_GCM_setup_gf_mult_table_kernel<<<numBlocks, dimBlocks>>>(d_engine.gf_last4, d_engine.gcm_h,
    //    d_engine.HL, d_engine.HH, d_engine.HL_long, d_engine.HH_long, d_engine.HL_sqr_long, d_engine.HH_sqr_long);
    void *args3[] = { &ctx->g_inst->d_engine.gf_last4, &ctx->g_inst->d_engine.gcm_h,
        &ctx->g_inst->d_engine.HL, &ctx->g_inst->d_engine.HH, &ctx->g_inst->d_engine.HL_long,
        &ctx->g_inst->d_engine.HH_long, &ctx->g_inst->d_engine.HL_sqr_long, &ctx->g_inst->d_engine.HH_sqr_long};
    cuLaunchKernel(ctx->g_inst->setup_table, 1, 1, 1, 1, 1, 1, 0, 0, args3, NULL);

    cuCtxSynchronize();
#endif

	crypto_tfm_set_flags(parent, crypto_cipher_get_flags(child) &
				     CRYPTO_TFM_RES_MASK);

	return err;
}

#define AES_BLOCK_MASK	(~(AES_BLOCK_SIZE - 1))

/* AES-NI version */
static int crypto_ecb_aesni_crypt(struct blkcipher_desc *desc,
			    struct blkcipher_walk *walk,
                struct crypto_blkcipher *tfm,
                struct crypto_kava_ecb_ctx *ctx,
                unsigned int offset,
			    enum kava_crypt_type crypt_type)
{
	unsigned int nbytes;
    unsigned int cur_offset = 0;
	int err;

	err = blkcipher_walk_virt(desc, walk);

	kernel_fpu_begin();
	while ((nbytes = walk->nbytes)) {
		u8 *wsrc = walk->src.virt.addr;
		u8 *wdst = walk->dst.virt.addr;

        /* Skip offset */
        if (cur_offset < offset) {
            cur_offset += nbytes;
        }

        /* Use AES-NI (arch/x86/crypto/aesni-intel_asm.S) */
        else {
            if (crypt_type == KAVA_ENCRYPT)
                aesni_ecb_enc(aes_ctx(ctx->aesni_aes_ctx), wdst, wsrc, nbytes & AES_BLOCK_MASK);
            else
                aesni_ecb_dec(aes_ctx(ctx->aesni_aes_ctx), wdst, wsrc, nbytes & AES_BLOCK_MASK);
            nbytes &= AES_BLOCK_SIZE - 1;
        }

		err = blkcipher_walk_done(desc, walk, nbytes);
	}
	kernel_fpu_end();

	return err;
}

/* AES-NI version */
static int crypto_ecb_aesni_crypt_caller(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes, unsigned int offset,
                  enum kava_crypt_type crypt_type)
{
	struct blkcipher_walk walk;
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_kava_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);

	blkcipher_walk_init(&walk, dst, src, nbytes);
	return crypto_ecb_aesni_crypt(desc, &walk, tfm, ctx, offset, crypt_type);
}

/* CPU version */
static int crypto_ecb_crypt(struct blkcipher_desc *desc,
			    struct blkcipher_walk *walk,
			    struct crypto_cipher *tfm,
			    void (*fn)(struct crypto_tfm *, u8 *, const u8 *))
{
	int bsize = crypto_cipher_blocksize(tfm);
	unsigned int nbytes;
	int err;

	err = blkcipher_walk_virt(desc, walk);

	while ((nbytes = walk->nbytes)) {
		u8 *wsrc = walk->src.virt.addr;
		u8 *wdst = walk->dst.virt.addr;

		do {
			fn(crypto_cipher_tfm(tfm), wdst, wsrc);
			wsrc += bsize;
			wdst += bsize;
		} while ((nbytes -= bsize) >= bsize);

		err = blkcipher_walk_done(desc, walk, nbytes);
	}

	return err;
}

/* CPU version */
static int crypto_ecb_encrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{
	struct blkcipher_walk walk;
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_kava_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
	struct crypto_cipher *child = ctx->child;

	blkcipher_walk_init(&walk, dst, src, nbytes);
	return crypto_ecb_crypt(desc, &walk, child,
				crypto_cipher_alg(child)->cia_encrypt);
}

/* CPU version */
static int crypto_ecb_decrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{
	struct blkcipher_walk walk;
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_kava_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
	struct crypto_cipher *child = ctx->child;

	blkcipher_walk_init(&walk, dst, src, nbytes);
	return crypto_ecb_crypt(desc, &walk, child,
				crypto_cipher_alg(child)->cia_decrypt);
}

//HF: TODO
static int crypto_kava_cuda_crypt(struct blkcipher_desc *desc,
			    struct scatterlist *dst, struct scatterlist *src,
			    unsigned int split_size, enum kava_crypt_type crypt_type,
                CUstream stream, unsigned int offset, char *const buf) //TODO: check buf size
{
	struct blkcipher_walk walk;
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_kava_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
    char *buf_t;
    unsigned int cur = 0;
	unsigned int nbytes;
	int err;
    CUdeviceptr g_plain_text;
    int block_x = split_size >= BPT_BYTES_PER_BLOCK ?
                BPT_BYTES_PER_BLOCK / 16 : split_size / 16;
    int grid_x = split_size >= BPT_BYTES_PER_BLOCK ?
                split_size / BPT_BYTES_PER_BLOCK : 1;

    // Allocate GPU memory
    g_plain_text = (CUdeviceptr)((char *)ctx->g_buf + offset);

    // Copy plain text to buffer
	blkcipher_walk_init(&walk, dst, src, split_size + offset);
	err = blkcipher_walk_virt(desc, &walk);

    buf_t = (char *)buf;
	while (cur < split_size + offset && (nbytes = walk.nbytes)) {
        if (cur >= offset) {
		    u8 *wsrc = walk.src.virt.addr;
            memcpy((void *)buf_t, wsrc, nbytes);
            buf_t += nbytes;
		}

        cur += nbytes;
		err = blkcipher_walk_done(desc, &walk, 0);
	}

    //TODO: check buf size
    cuMemcpyHtoDAsync(g_plain_text, buf, split_size, stream);

    // Execute encrypt/decrypt kernel
    if (crypt_type == KAVA_ENCRYPT) {
        //void *args[] = { &ctx->g_key_enc, &ctx->key_round, &g_plain_text };
        //cuLaunchKernel(ctx->g_inst->encrypt_fn, grid_x, 1, 1, block_x, 1, 1, 0, stream, args, NULL);

        //int num_block = (size / 16 + kBaseThreadNum-1) / kBaseThreadNum;
        //dim3 numBlocks(num_block);
        //dim3 dimBlocks(kBaseThreadNum);
        //AES_GCM_xcrypt_kernel<<<numBlocks, dimBlocks>>>(dst, d_engine.sbox, d_engine.aes_roundkey, nonce, src, size);

        int num_block = (split_size / 16 + kBaseThreadNum-1) / kBaseThreadNum;

        //TODO: we need an output array
        void *args[] = { &ctx->g_key_enc, &ctx->key_round, &g_plain_text };
        cuLaunchKernel(ctx->g_inst->xcrypt_fn, num_block, 1, 1, kBaseThreadNum, 1, 1, 0, stream, args, NULL);

    }
    else {
        void *args_2[] = { &ctx->g_key_dec, &ctx->key_round, &g_plain_text };
        cuLaunchKernel(ctx->g_inst->decrypt_fn, grid_x, 1, 1, block_x, 1, 1, 0, stream, args_2, NULL);
    }

    // Retrieve crypto text
    cuMemcpyDtoHAsync((void *)buf, g_plain_text, split_size, stream);

    // dump_stack();

	return err;
}

/**
 * crypto_kava_cuda_copy_back - Copy back encrypted/decrypted data
 *
 * This function was used to copy back a single split, but now it is used to copy back
 * the whole file.
 */
static int crypto_kava_cuda_copy_back(struct blkcipher_desc *desc,
			    struct scatterlist *dst, struct scatterlist *src,
			    unsigned int total_size, char const *const cpu_buf)
{
    int err = 0;
    int cur = 0;
	struct blkcipher_walk walk;
    unsigned int nbytes = 0;
    char *buf_t = (char *)cpu_buf;

	blkcipher_walk_init(&walk, dst, src, total_size);
	err = blkcipher_walk_virt(desc, &walk);

	while (cur < total_size && (nbytes = walk.nbytes)) {
        u8 *wdst = walk.dst.virt.addr;
        memcpy(wdst, buf_t, nbytes);

        buf_t += nbytes;
        cur += nbytes;
		err = blkcipher_walk_done(desc, &walk, 0);
	}

    return err;
}

static int crypto_kava_ecb_crypt(struct blkcipher_desc *desc,
			    struct scatterlist *dst, struct scatterlist *src,
			    unsigned int nbytes, enum kava_crypt_type crypt_type)
{
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_kava_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
	int err = 0;
    unsigned int left_bytes = nbytes;
    int i = 0;
    unsigned int offset = 0;
    unsigned int size;
    char *cpu_buf;

    // Allocate CPU memory
    cpu_buf = (char *)kava_alloc(nbytes);

    while (left_bytes > 0) {
        /* Split buffer when the size is larger than 1.5x split threshold */
        if ((left_bytes >> PAGE_SHIFT) >= (split_threshold + (split_threshold >> 1)))
            size = split_threshold << PAGE_SHIFT;
        else
            size = left_bytes;

        err = crypto_kava_cuda_crypt(desc, dst, src, size,
                crypt_type, ctx->g_inst->g_stream[i], offset, cpu_buf + offset);
        if (err < 0)
            break;

        i++;
        if (i >= split_max)
            i = 0;
        offset += size;
        left_bytes -= size;
    }

    cuCtxSynchronize();
    crypto_kava_cuda_copy_back(desc, dst, src, nbytes, cpu_buf);
    kava_free(cpu_buf);

    return err;
}

static int kthread_crypto_ecb_aesni_crypt(void *args)
{
    struct kthread_ecb_arg *kargs = (struct kthread_ecb_arg *)args;
    struct crypto_kava_ecb_inst *inst;
    int err;

    printk(KERN_INFO "AES-NI kthread starts, reading args at %lx\n", (uintptr_t)kargs);

    while (!kthread_should_stop()) {
        wait_event_interruptible(aesni_crypt_wq_head, aesni_crypt_called || kthread_should_stop());
        if (kthread_should_stop()) break;

        aesni_crypt_called = 0;
        inst = ((struct crypto_kava_ecb_ctx *)crypto_blkcipher_ctx(kargs->desc->tfm))->g_inst;

        err = crypto_ecb_aesni_crypt_caller(kargs->desc, kargs->dst,
                kargs->src, kargs->nbytes, kargs->offset, kargs->crypt_type);
        complete(&inst->aesni_completion);
    }
    return 0;
}

static int crypto_kava_ecb_mixed_crypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes, enum kava_crypt_type crypt_type)
{
	struct crypto_blkcipher *tfm = desc->tfm;
	struct crypto_kava_ecb_ctx *ctx = crypto_blkcipher_ctx(tfm);
    unsigned int gpu_nbytes, aesni_nbytes;
    unsigned int gpu_npages;
    struct kthread_ecb_arg *kargs = &ctx->g_inst->aesni_kargs;
    int err = 0;

    /* Encrypt first part of the file with GPU. */
    gpu_npages = (nbytes >> PAGE_SHIFT) * (100 - aesni_fraction) / 100;
    gpu_nbytes = gpu_npages << PAGE_SHIFT;
    aesni_nbytes = nbytes - gpu_nbytes;

    kargs->desc = desc;
    kargs->dst = dst;
    kargs->src = src;
    kargs->nbytes = aesni_nbytes;
    kargs->offset = gpu_nbytes;
    kargs->crypt_type = crypt_type;

    /* Run AES-NI in a new kthread; run GPU in this thread. */
    aesni_crypt_called = 1;
    wake_up_interruptible(&aesni_crypt_wq_head);
    if (gpu_nbytes > 0)
        err = crypto_kava_ecb_crypt(desc, dst, src, gpu_nbytes, crypt_type);
    wait_for_completion_interruptible(&ctx->g_inst->aesni_completion);
    reinit_completion(&ctx->g_inst->aesni_completion);

    return err;
}

static int crypto_kava_ecb_encrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{
    if (!KAVA_ENABLE || nbytes <= KAVA_ECB_SIZE_THRESHOLD) {
        return crypto_ecb_encrypt(desc, dst, src, nbytes);
    }

    switch (aesni_fraction) {
        case 0:
            return crypto_kava_ecb_crypt(desc, dst, src, nbytes, KAVA_ENCRYPT);
        case 100:
            return crypto_ecb_aesni_crypt_caller(desc, dst, src, nbytes, 0, KAVA_ENCRYPT);
        default:
            return crypto_kava_ecb_mixed_crypt(desc, dst, src, nbytes, KAVA_ENCRYPT);
    }
}

static int crypto_kava_ecb_decrypt(struct blkcipher_desc *desc,
			      struct scatterlist *dst, struct scatterlist *src,
			      unsigned int nbytes)
{
    if (!KAVA_ENABLE || nbytes <= KAVA_ECB_SIZE_THRESHOLD) {
        return crypto_ecb_decrypt(desc, dst, src, nbytes);
    }

    switch (aesni_fraction) {
        case 0:
	        return crypto_kava_ecb_crypt(desc, dst, src, nbytes, KAVA_DECRYPT);
        case 100:
            return crypto_ecb_aesni_crypt_caller(desc, dst, src, nbytes, 0, KAVA_DECRYPT);
        default:
            return crypto_kava_ecb_mixed_crypt(desc, dst, src, nbytes, KAVA_DECRYPT);
    }
}

static int crypto_kava_ecb_init_kava(struct crypto_kava_ecb_inst *inst)
{
    CUresult res;
    int i;

    if (aesni_fraction < 0)
        aesni_fraction = 0;
    if (aesni_fraction > 100)
        aesni_fraction = 100;

    if (aesni_fraction > 0) {
        inst->aesni_kth = kthread_run(kthread_crypto_ecb_aesni_crypt, (void *)&inst->aesni_kargs, "kava_aesni_crypt");
        init_completion(&inst->aesni_completion);
        init_waitqueue_head(&aesni_crypt_wq_head);
    }

    if (!dev_initialized) {
        cuInit(0);
        dev_initialized = 1;
    }
    res = cuDeviceGet(&inst->device, 0);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: acquire GPU device 0\n");
        return -ENODEV;
    }

    res = cuCtxCreate(&inst->context, 0, inst->device);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create GPU context\n");
        return -EBUSY;
    }

    res = cuModuleLoad(&inst->module,
            "/home/hfingler/hf-HACK/kava/driver/hack_fs/crypto_gcm/gcm.cubin");
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: load AES-ECB CUDA module (%d)\n", res);
        return -ENOENT;
    }

    res = cuModuleGetFunction(&inst->xcrypt_fn, inst->module, "_Z21AES_GCM_xcrypt_kernelPhS_S_S_S_j");
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: load encrypt kernel\n");
        return -ENOSYS;
    }
    res = cuModuleGetFunction(&inst->mac_fn, inst->module, "_Z18AES_GCM_mac_kernelPmS_S_iPhjS0_");
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: load decrypt kernel\n");
        return -ENOSYS;
    }
    res = cuModuleGetFunction(&inst->key_expansion, inst->module, "_Z24AES_key_expansion_kernelPhS_S_S_");
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: load decrypt kernel\n");
        return -ENOSYS;
    }
    res = cuModuleGetFunction(&inst->setup_table, inst->module, "_Z34AES_GCM_setup_gf_mult_table_kernelPmPhS_S_S_S_S_S_");
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: load decrypt kernel\n");
        return -ENOSYS;
    }
    res = cuModuleGetFunction(&inst->encrypt_oneblock, inst->module, "_Z28AES_encrypt_one_block_kernelPhS_S_");
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: load decrypt kernel\n");
        return -ENOSYS;
    }
    

    res = cuFuncSetCacheConfig(inst->xcrypt_fn, CU_FUNC_CACHE_PREFER_L1);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: set encrypt_fn cache config\n");
        return -ENOSYS;
    }
    res = cuFuncSetCacheConfig(inst->mac_fn, CU_FUNC_CACHE_PREFER_L1);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: set decrypt_fn cache config\n");
        return -ENOSYS;
    }

    inst->g_stream = (CUstream *)kmalloc(sizeof(CUstream) * split_max, GFP_KERNEL);
    for (i = 0; i < split_max; i++) {
        res = cuStreamCreate(&inst->g_stream[i], 0);
        if (res != CUDA_SUCCESS) {
            printk(KERN_ERR "[kava] Error: create CUDA stream [%d]\n", i);
            return -ENOSYS;
        }
    }

    return 0;
}

static int crypto_kava_ecb_init_tfm(struct crypto_tfm *tfm)
{
	struct crypto_instance *inst = (void *)tfm->__crt_alg;
	struct crypto_spawn *spawn = crypto_instance_ctx(inst);
	struct crypto_kava_ecb_ctx *ctx = crypto_tfm_ctx(tfm);
	struct crypto_cipher *cipher;
#if KAVA_ENABLE
    CUresult res = CUDA_SUCCESS;
#endif

	cipher = crypto_spawn_cipher(spawn);
	if (IS_ERR(cipher))
		return PTR_ERR(cipher);

	ctx->child = cipher;

#if KAVA_ENABLE
    ctx->g_inst = &kava_ecb_inst;

    res = cuCtxSetCurrent(ctx->g_inst->context);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: switch to AES_CTX GPU context\n");
        return -ENOMEM;
    }

    // res = cuMemAlloc(&ctx->g_aes_ctx, sizeof(struct crypto_aes_ctx));
    // if (res != CUDA_SUCCESS) {
    //     printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
    //     return -ENOMEM;
    // }

    //ctx->g_key_enc = (CUdeviceptr)((struct crypto_aes_ctx *)ctx->g_aes_ctx)->key_enc;
    //ctx->g_key_dec = (CUdeviceptr)((struct crypto_aes_ctx *)ctx->g_aes_ctx)->key_dec;

    res = cuMemAlloc(&(kava_ecb_inst.d_engine.sbox), SBOX_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.rsbox), SBOX_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.Rcon), RCON_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.key), AESGCM_KEYLEN);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.aes_roundkey), AES_ROUNDKEYLEN);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.gcm_h), 16);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }

    res = cuMemAlloc(&(kava_ecb_inst.d_engine.HL), AESGCM_BLOCK_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.HH), AESGCM_BLOCK_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.HL_long), AESGCM_BLOCK_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.HH_long), AESGCM_BLOCK_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.HL_sqr_long), AESGCM_BLOCK_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.HH_sqr_long), AESGCM_BLOCK_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.gf_last4), AESGCM_BLOCK_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }

    res = cuMemAlloc(&(kava_ecb_inst.d_engine.buffer1), AESGCM_BLOCK_SIZE * AES_GCM_STEP * AES_GCM_STEP);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.buffer2), AESGCM_BLOCK_SIZE * AES_GCM_STEP);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemAlloc(&(kava_ecb_inst.d_engine.nonce_device), 12);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }

    //now copy stuff
    res = cuMemcpyHtoD(kava_ecb_inst.d_engine.sbox, sbox_host, SBOX_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemcpyHtoD(kava_ecb_inst.d_engine.rsbox, rsbox_host, SBOX_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemcpyHtoD(kava_ecb_inst.d_engine.Rcon, Rcon_host, RCON_SIZE);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    res = cuMemcpyHtoD(kava_ecb_inst.d_engine.key, key, AESGCM_KEYLEN);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }
    //TODO: this doesn't make sense, but it works..
    res = cuMemcpyHtoD(kava_ecb_inst.d_engine.nonce_device, kava_ecb_inst.d_engine.nonce_host, AESGCM_KEYLEN);
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create AES_CTX GPU buffer\n");
        return -ENOMEM;
    }

    cuCtxSynchronize();

    /* Why +1 here: we only make a split when the left bytes are more than half of
     * the split threshold, so that we will not send a small piece of tailing bytes
     * within a separate split. That means, the last split may be larger than the
     * split threshold.
     */
    //TODO: we might have to add bytes here for MAC
    res = cuMemAlloc(&ctx->g_buf, (split_max + 1)* (split_threshold << PAGE_SHIFT));
    if (res != CUDA_SUCCESS) {
        printk(KERN_ERR "[kava] Error: create reserved GPU buffer\n");
        return -ENOMEM;
    }
    pr_info("crypto_kava_ecb_init_tfm succeeded\n");
#endif

	return 0;
}

static void crypto_kava_ecb_exit_inst(struct crypto_kava_ecb_inst *inst)
{
    int i;
    for (i = 0; i < split_max; i++)
        cuStreamDestroy(inst->g_stream[i]);
    kfree(inst->g_stream);
    cuCtxDestroy(inst->context);

    if (aesni_fraction > 0) {
        kthread_stop(inst->aesni_kth);
    }
}

static void crypto_kava_ecb_exit_tfm(struct crypto_tfm *tfm)
{
	struct crypto_kava_ecb_ctx *ctx = crypto_tfm_ctx(tfm);
	crypto_free_cipher(ctx->child);

#if KAVA_ENABLE
    // Free GPU context
    cuMemFree(ctx->g_buf);
    cuMemFree(ctx->g_aes_ctx);
#endif
}

static struct crypto_instance *crypto_kava_ecb_alloc(struct rtattr **tb)
{
	struct crypto_instance *inst;
	struct crypto_alg *alg;
	int err;

	err = crypto_check_attr_type(tb, CRYPTO_ALG_TYPE_BLKCIPHER);
	if (err)
		return ERR_PTR(err);

	alg = crypto_get_attr_alg(tb, CRYPTO_ALG_TYPE_CIPHER,
				  CRYPTO_ALG_TYPE_MASK);
	if (IS_ERR(alg))
		return ERR_CAST(alg);

	inst = crypto_alloc_instance("kava_ecb", alg);
	if (IS_ERR(inst))
		goto out_put_alg;

	inst->alg.cra_flags = CRYPTO_ALG_TYPE_BLKCIPHER;
	inst->alg.cra_priority = alg->cra_priority;
	inst->alg.cra_blocksize = alg->cra_blocksize;
	inst->alg.cra_alignmask = alg->cra_alignmask;
	inst->alg.cra_type = &crypto_blkcipher_type;

	//inst->alg.cra_blkcipher.min_keysize = alg->cra_cipher.cia_min_keysize;
	//inst->alg.cra_blkcipher.max_keysize = alg->cra_cipher.cia_max_keysize;

    inst->alg.cra_blkcipher.min_keysize = alg->cra_cipher.cia_min_keysize;
    inst->alg.cra_blkcipher.max_keysize = alg->cra_cipher.cia_max_keysize;

	inst->alg.cra_ctxsize = sizeof(struct crypto_kava_ecb_ctx);

	inst->alg.cra_init = crypto_kava_ecb_init_tfm;
	inst->alg.cra_exit = crypto_kava_ecb_exit_tfm;

	inst->alg.cra_blkcipher.setkey = crypto_kava_ecb_setkey;
	inst->alg.cra_blkcipher.encrypt = crypto_kava_ecb_encrypt;
	inst->alg.cra_blkcipher.decrypt = crypto_kava_ecb_decrypt;

#if KAVA_ENABLE
    err = crypto_kava_ecb_init_kava(&kava_ecb_inst);
    if (err)
        return ERR_PTR(err);
#endif

out_put_alg:
	crypto_mod_put(alg);
	return inst;
}

static void crypto_kava_ecb_free(struct crypto_instance *inst)
{
#if KAVA_ENABLE
    crypto_kava_ecb_exit_inst(&kava_ecb_inst);
#endif
	crypto_drop_spawn(crypto_instance_ctx(inst));
	kfree(inst);
}

static struct crypto_template crypto_kava_ecb_tmpl = {
	.name = "kava_ecb",
	.alloc = crypto_kava_ecb_alloc,
	.free = crypto_kava_ecb_free,
	.module = THIS_MODULE,
};

static int __init crypto_kava_ecb_module_init(void)
{
    kava_aesni_init();
	return crypto_register_template(&crypto_kava_ecb_tmpl);
}

static void __exit crypto_kava_ecb_module_exit(void)
{
	crypto_unregister_template(&crypto_kava_ecb_tmpl);
    kava_aesni_exit();
}

module_init(crypto_kava_ecb_module_init);
module_exit(crypto_kava_ecb_module_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("ECB block cipher algorithm accelerated by KAvA");
MODULE_ALIAS_CRYPTO("kava_ecb");
