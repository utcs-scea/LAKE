#ifndef __GCM_H__
#define __GCM_H__

#include <cuda.h>

typedef unsigned long long int u64;
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define Nb 4
#define Nk 8
#define Nr 14

//#define FIXED_SIZE_FULL (0x1UL << 20) // 1 MB

#define SBOX_SIZE 256
#define RCON_SIZE 11
#define AESGCM_BLOCK_SIZE 16

#define AESGCM_KEYLEN 32u
#define AES_ROUNDKEYLEN 240u
#define AES_BLOCKLEN 16u
#define AES_MACLEN 12u
#define AES_GCM_STEP 64u

const int kBaseThreadBits = 8;
const int kBaseThreadNum  = 1 << kBaseThreadBits;

#define crypto_aead_aes256gcm_NPUBBYTES 12U
#define crypto_aead_aes256gcm_ABYTES 16U

typedef u8 state_t[4][4];

struct AES_GCM_engine_ctx {
    CUdeviceptr sbox;
    CUdeviceptr rsbox;
    CUdeviceptr Rcon;
    CUdeviceptr key;
    CUdeviceptr aes_roundkey;
    CUdeviceptr gcm_h;

    CUdeviceptr HL;
    CUdeviceptr HH;
    CUdeviceptr HL_long;
    CUdeviceptr HH_long;
    CUdeviceptr HL_sqr_long;
    CUdeviceptr HH_sqr_long;
    CUdeviceptr gf_last4;
    CUdeviceptr nonce_device;

    CUdeviceptr buffer1;
    CUdeviceptr buffer2;

    //XXX
    //u8 key[AESGCM_KEYLEN];
    //u8 nonce_host[crypto_aead_aes256gcm_NPUBBYTES];

    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction xcrypt_kernel;
    CUfunction mac_kernel;
    CUfunction final_mac_kernel;
    CUfunction key_expansion_kernel;
    CUfunction setup_table_kernel;
    CUfunction encrypt_oneblock_kernel;
    CUstream *g_stream;
};

void lake_AES_GCM_encrypt(struct AES_GCM_engine_ctx* d_engine, u8* d_dst, u8* d_src, u32 size);
void lake_AES_GCM_decrypt(struct AES_GCM_engine_ctx* d_engine, u8* d_dst, u8* d_src, u32 size);
void lake_AES_GCM_init(struct AES_GCM_engine_ctx* d_engine);
void lake_AES_GCM_setkey(struct AES_GCM_engine_ctx* d_engine, u8* key);

#ifdef __CUDACC__
#define ENDIAN_SELECTOR 0x00000123
#define GETU32(plaintext) __byte_perm(*(u32*)(plaintext), 0, ENDIAN_SELECTOR)
#define PUTU32(ciphertext, st) {*(u32*)(ciphertext) = __byte_perm((st), 0, ENDIAN_SELECTOR);}
#endif
#endif
