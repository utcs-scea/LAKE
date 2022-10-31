#include "gcm_cuda.h"

#ifdef __KERNEL__
#include <linux/err.h>
#include <linux/kernel.h>
#include <linux/vmalloc.h>
#include "lake_shm.h"
#else
#include <errno.h>
#endif

#define MAX_PAGES_PER_OP 4096

static u64 gf_last4_host[16] = {
  0x0000, 0x1c20, 0x3840, 0x2460, 0x7080, 0x6ca0, 0x48c0, 0x54e0,
  0xe100, 0xfd20, 0xd940, 0xc560, 0x9180, 0x8da0, 0xa9c0, 0xb5e0  };

static u8 sbox_host[256] = {
  //0     1    2      3     4    5     6     7      8    9     A      B    C     D     E     F
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 };

static u8 rsbox_host[256] = {
  0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
  0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
  0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
  0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
  0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
  0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
  0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
  0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
  0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
  0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
  0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
  0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
  0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
  0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
  0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
  0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d };

static u8 Rcon_host[11] = {
  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

//TODO: init nonce, figure out when to call next_nonce
void lake_AES_GCM_init(struct AES_GCM_engine_ctx* d_engine) {
    u8* kbuf = kava_alloc(SBOX_SIZE);

    gpuErrchk(cuMemAlloc(&d_engine->sbox, SBOX_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->rsbox, SBOX_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->Rcon,  RCON_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->key,   AESGCM_KEYLEN));
    gpuErrchk(cuMemAlloc(&d_engine->aes_roundkey, AES_ROUNDKEYLEN));
    gpuErrchk(cuMemAlloc(&d_engine->gcm_h, 16));

    gpuErrchk(cuMemAlloc(&d_engine->HL,          AESGCM_BLOCK_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->HH,          AESGCM_BLOCK_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->HL_long,     AESGCM_BLOCK_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->HH_long,     AESGCM_BLOCK_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->HL_sqr_long, AESGCM_BLOCK_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->HH_sqr_long, AESGCM_BLOCK_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->gf_last4,    AESGCM_BLOCK_SIZE));
    gpuErrchk(cuMemAlloc(&d_engine->nonce_device, 12));
    
    gpuErrchk(cuMemAlloc(&d_engine->d_src, MAX_PAGES_PER_OP*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES)));
    gpuErrchk(cuMemAlloc(&d_engine->d_dst, MAX_PAGES_PER_OP*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES)));

    gpuErrchk(cuMemAlloc(&d_engine->buffer1, AESGCM_BLOCK_SIZE * AES_GCM_STEP * AES_GCM_STEP));
    gpuErrchk(cuMemAlloc(&d_engine->buffer2, AESGCM_BLOCK_SIZE * AES_GCM_STEP));

    memcpy(kbuf, sbox_host, SBOX_SIZE);
    gpuErrchk(cuMemcpyHtoD(d_engine->sbox, kbuf, SBOX_SIZE));
    //gpuErrchk(cuMemcpyHtoD(d_engine->sbox, sbox_host, SBOX_SIZE));

    memcpy(kbuf, rsbox_host, SBOX_SIZE);
    gpuErrchk(cuMemcpyHtoD(d_engine->rsbox, kbuf, SBOX_SIZE));
    //gpuErrchk(cuMemcpyHtoD(d_engine->rsbox, rsbox_host, SBOX_SIZE));

    memcpy(kbuf, Rcon_host, RCON_SIZE);
    gpuErrchk(cuMemcpyHtoD(d_engine->Rcon, kbuf, RCON_SIZE));
    //gpuErrchk(cuMemcpyHtoD(d_engine->Rcon, Rcon_host, RCON_SIZE));
    kava_free(kbuf);
}

void lake_AES_GCM_destroy(struct AES_GCM_engine_ctx* d_engine) {
    gpuErrchk(cuMemFree(d_engine->sbox));
    gpuErrchk(cuMemFree(d_engine->rsbox));
    gpuErrchk(cuMemFree(d_engine->Rcon));
    gpuErrchk(cuMemFree(d_engine->key));
    gpuErrchk(cuMemFree(d_engine->aes_roundkey));
    gpuErrchk(cuMemFree(d_engine->gcm_h));

    gpuErrchk(cuMemFree(d_engine->HL));
    gpuErrchk(cuMemFree(d_engine->HH));
    gpuErrchk(cuMemFree(d_engine->HL_long));
    gpuErrchk(cuMemFree(d_engine->HH_long ));
    gpuErrchk(cuMemFree(d_engine->HL_sqr_long));
    gpuErrchk(cuMemFree(d_engine->HH_sqr_long));
    gpuErrchk(cuMemFree(d_engine->gf_last4));
    gpuErrchk(cuMemFree(d_engine->nonce_device));
    
    gpuErrchk(cuMemFree(d_engine->buffer1));
    gpuErrchk(cuMemFree(d_engine->buffer2));

    gpuErrchk(cuMemFree(d_engine->d_src));
    gpuErrchk(cuMemFree(d_engine->d_dst));
}

void lake_AES_GCM_setkey(struct AES_GCM_engine_ctx* d_engine, const u8* key) {
    u8* nonce_host;
    int i = 0;
    u8* kbuf = kava_alloc(AESGCM_KEYLEN);

    void *args[] = { &d_engine->sbox, &d_engine->Rcon, &d_engine->key, &d_engine->aes_roundkey};
    void *args2[] = { &d_engine->sbox, &d_engine->aes_roundkey, &d_engine->gcm_h};
    void *args3[] = { &d_engine->gf_last4, &d_engine->gcm_h,
        &d_engine->HL, &d_engine->HH, &d_engine->HL_long,
        &d_engine->HH_long, &d_engine->HL_sqr_long, &d_engine->HH_sqr_long};

#ifdef __KERNEL__
    nonce_host = vmalloc(16);
#else
    nonce_host = malloc(16);
#endif
    
    for(i = 0 ; i < 12 ; i++)
        nonce_host[i] = i;

    gpuErrchk(cuMemAlloc((CUdeviceptr*)&d_engine->nonce_device, 12));

    memcpy(kbuf, nonce_host, crypto_aead_aes256gcm_NPUBBYTES);
    gpuErrchk(cuMemcpyHtoD(d_engine->nonce_device, kbuf, crypto_aead_aes256gcm_NPUBBYTES));
    //gpuErrchk(cuMemcpyHtoD(d_engine->nonce_device, nonce_host, crypto_aead_aes256gcm_NPUBBYTES));

    memcpy(kbuf, key, AESGCM_KEYLEN);
    gpuErrchk(cuMemcpyHtoD(d_engine->key, kbuf, AESGCM_KEYLEN));
    //gpuErrchk(cuMemcpyHtoD(d_engine->key, key, AESGCM_KEYLEN));

    for(i = 0 ; i < 16 ; i++)
        nonce_host[i] = 0;

    memcpy(kbuf, nonce_host, 16);
    gpuErrchk(cuMemcpyHtoD(d_engine->gcm_h, kbuf, 16));
    //gpuErrchk(cuMemcpyHtoD(d_engine->gcm_h, nonce_host, 16));

    cuLaunchKernel(d_engine->key_expansion_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args, 0);
    cuLaunchKernel(d_engine->encrypt_oneblock_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args2, 0);

    memcpy(kbuf, gf_last4_host, AESGCM_BLOCK_SIZE);
    cuMemcpyHtoD(d_engine->gf_last4, kbuf, AESGCM_BLOCK_SIZE);
    //cuMemcpyHtoD(d_engine->gf_last4, gf_last4_host, AESGCM_BLOCK_SIZE);

    cuLaunchKernel(d_engine->setup_table_kernel, 1, 1, 1, 1, 1, 1, 0, 0, args3, 0);
    gpuErrchk(cuCtxSynchronize());

    #ifdef __KERNEL__
        vfree(nonce_host);
    #else
        free(nonce_host);
    #endif
    kava_free(kbuf);
}

static void lake_AES_GCM_xcrypt(struct AES_GCM_engine_ctx* d_engine, CUdeviceptr d_dst, CUdeviceptr d_src, u32 size) {
    int num_block = (size / 16 + kBaseThreadNum-1) / kBaseThreadNum;
    void *args[] = { &d_dst, &d_engine->sbox, &d_engine->aes_roundkey,
        &d_engine->nonce_device, &d_src, &size };
    cuLaunchKernel(d_engine->xcrypt_kernel, num_block, 1, 1, 
            kBaseThreadNum, 1, 1, 0, 0, args, 0);
}

static void lake_AES_GCM_compute_mac(struct AES_GCM_engine_ctx* d_engine, CUdeviceptr dst, CUdeviceptr src, u32 size) {
    int sq_step = AES_GCM_STEP * AES_GCM_STEP;
    u32 num_block = size / 16;
    int nparts = AES_GCM_STEP;
    int one = 1;
    void *args[] = { &d_engine->gf_last4, &d_engine->HL_sqr_long,
        &d_engine->HH_sqr_long, &sq_step, &src, &num_block, &d_engine->buffer1};

    void *args2[] = { &d_engine->gf_last4, &d_engine->HL_long,
        &d_engine->HH_long, &nparts, &d_engine->buffer1, &sq_step, &d_engine->buffer2 };

    void *args3[] = { &d_engine->gf_last4, &d_engine->HL,
        &d_engine->HH, &one, &d_engine->buffer2, &nparts, &dst };

    void *args4[] = { &d_engine->gf_last4, &d_engine->HL,
        &d_engine->HH, &d_engine->sbox, &d_engine->aes_roundkey, &d_engine->nonce_device,
        &dst, &size, &src };

    cuLaunchKernel(d_engine->mac_kernel, AES_GCM_STEP, 1, 1, 
            AES_GCM_STEP, 1, 1, 0, 0, args, 0);
    cuLaunchKernel(d_engine->mac_kernel, AES_GCM_STEP / 8, 1, 1, 
            8, 1, 1, 0, 0, args2, 0);
    cuLaunchKernel(d_engine->mac_kernel, 1, 1, 1, 
            1, 1, 1, 0, 0, args3, 0);
    cuLaunchKernel(d_engine->final_mac_kernel, 1, 1, 1, 
            1, 1, 1, 0, 0, args4, 0);
}

//TODO: set IV
void lake_AES_GCM_encrypt(struct AES_GCM_engine_ctx* d_engine, CUdeviceptr d_dst, CUdeviceptr d_src, u32 size) {
    //assert(size % AES_BLOCKLEN == 0);
    lake_AES_GCM_xcrypt(d_engine, d_dst, d_src, size);
    //TODO: enable MAC. using it as is breaks correctness
    //lake_AES_GCM_compute_mac(d_engine, d_dst+size, d_src, size);
}

void lake_AES_GCM_alloc_pages(CUdeviceptr* src, u32 size) {
    gpuErrchk(cuMemAlloc(src, size));
}

void lake_AES_GCM_free(CUdeviceptr src) {
    cuMemFree(src);
}

void lake_AES_GCM_copy_to_device(CUdeviceptr dst, u8* buf, u32 size) {
    int left = size;
    int max = 1024*PAGE_SIZE;
    CUdeviceptr cur = dst;
    u8* cur_buf = buf;
    while (1) {
        if (left <= max) {
            cuMemcpyHtoDAsync(cur, cur_buf, left, 0);
            break;
        }
        cuMemcpyHtoDAsync(cur, cur_buf, max, 0);
        cur += max;
        cur_buf += max;
        left -= max;
    }
    //cuMemcpyHtoDAsync(dst, buf, size, 0);
}

void lake_AES_GCM_copy_from_device(u8* buf, CUdeviceptr src, u32 size) {
    cuMemcpyDtoHAsync(buf, src, size, 0);
}

void lake_AES_GCM_decrypt(struct AES_GCM_engine_ctx* d_engine, CUdeviceptr d_dst, CUdeviceptr d_src, u32 size) {
    //assert(size % AES_BLOCKLEN == 0);
    //TODO: enable MAC. using it as is breaks correctness
    //lake_AES_GCM_compute_mac(d_engine, d_dst, d_src, size);
    // TODO verify mac for i in crypto_aead_aes256gcm_ABYTES: (dst == src[size])
    lake_AES_GCM_xcrypt(d_engine, d_dst, d_src, size);
}

static volatile char dev_initialized = 0;

int lake_AES_GCM_init_fns(struct AES_GCM_engine_ctx *d_engine, char *cubin_path) {
    int res=0;

    if (!dev_initialized) {
        cuInit(0);
        dev_initialized = 1;
    }
    else
        return 0;

    res = cuDeviceGet(&d_engine->device, 0);
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: acquire GPU device 0 %d\n", res);
        return -ENODEV;
    }

    res = cuCtxCreate(&d_engine->context, 0, d_engine->device);
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: creating ctx\n");
        return -ENODEV;
    }

    res = cuCtxCreate(&d_engine->context, 0, d_engine->device);
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: create GPU context\n");
        return -EBUSY;
    }

    res = cuModuleLoad(&d_engine->module, cubin_path);
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: load AES-ECB CUDA module (%d)\n", res);
        return -ENOENT;
    }

    res = cuModuleGetFunction(&d_engine->xcrypt_kernel, d_engine->module, "_Z21AES_GCM_xcrypt_kernelPhS_S_S_S_j");
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: load encrypt kernel\n");
        return -ENOSYS;
    }
    res = cuModuleGetFunction(&d_engine->mac_kernel, d_engine->module, "_Z18AES_GCM_mac_kernelPmS_S_iPhjS0_");
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: load decrypt kernel\n");
        return -ENOSYS;
    }
    res = cuModuleGetFunction(&d_engine->final_mac_kernel, d_engine->module, "_Z24AES_GCM_mac_final_kernelPmS_S_PhS0_S0_S0_jS0_");
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: load decrypt kernel\n");
        return -ENOSYS;
    }
    res = cuModuleGetFunction(&d_engine->key_expansion_kernel, d_engine->module, "_Z24AES_key_expansion_kernelPhS_S_S_");
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: load decrypt kernel\n");
        return -ENOSYS;
    }
    res = cuModuleGetFunction(&d_engine->setup_table_kernel, d_engine->module, "_Z34AES_GCM_setup_gf_mult_table_kernelPmPhS_S_S_S_S_S_");
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: load decrypt kernel\n");
        return -ENOSYS;
    }
    res = cuModuleGetFunction(&d_engine->encrypt_oneblock_kernel, d_engine->module, "_Z28AES_encrypt_one_block_kernelPhS_S_");
    if (res != CUDA_SUCCESS) {
        PRINT("[lake] Error: load decrypt kernel\n");
        return -ENOSYS;
    }
    return 0;
}

