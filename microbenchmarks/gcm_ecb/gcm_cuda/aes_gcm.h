#ifndef _AES_GCM_H_
#define _AES_GCM_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define SBOX_SIZE 256
#define RCON_SIZE 11
#define BLOCK_SIZE 16

#define FIXED_SIZE_FULL (0x1UL << 20) // 1 MB

#define AES_KEYLEN 32u
#define AES_ROUNDKEYLEN 240u
#define AES_BLOCKLEN 16u
#define AES_MACLEN 12u
#define AES_GCM_STEP 64u

const int kBaseThreadBits = 8;
const int kBaseThreadNum  = 1 << kBaseThreadBits;

#define crypto_aead_aes256gcm_NPUBBYTES 12U
#define crypto_aead_aes256gcm_ABYTES 16U

struct AES_GCM_engine {
    uint8_t sbox[SBOX_SIZE];
    uint8_t rsbox[SBOX_SIZE];
    uint8_t Rcon[RCON_SIZE];
    uint8_t key[AES_KEYLEN];
    uint8_t aes_roundkey[AES_ROUNDKEYLEN];
    uint8_t gcm_h[16];

    uint64_t HL[BLOCK_SIZE];
    uint64_t HH[BLOCK_SIZE];
    uint64_t HL_long[BLOCK_SIZE];
    uint64_t HH_long[BLOCK_SIZE];
    uint64_t HL_sqr_long[BLOCK_SIZE];
    uint64_t HH_sqr_long[BLOCK_SIZE];
    uint64_t gf_last4[BLOCK_SIZE];

    uint8_t buffer1[BLOCK_SIZE * AES_GCM_STEP * AES_GCM_STEP];
    uint8_t buffer2[BLOCK_SIZE * AES_GCM_STEP];
};

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
};

void AES_GCM_init(uint8_t* key);
void AES_GCM_decrypt(uint8_t* dst, uint8_t* src, uint32_t size);
void AES_GCM_encrypt(uint8_t* dst, uint8_t* src, uint32_t size);

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#endif
