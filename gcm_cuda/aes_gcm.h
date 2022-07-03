#ifndef _AES_GCM_H_
#define _AES_GCM_H_

#include <stdint.h>
#include "common.h"

#define SBOX_SIZE 256
#define RCON_SIZE 11
#define BLOCK_SIZE 16

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

// void AES_GCM_init(AES_GCM_engine** engine, const uint8_t* key, hipStream_t stream);

// void AES_GCM_destroy(AES_GCM_engine* engine);

// void AES_GCM_encrypt(hip_launch_batch_t* batch, uint8_t* dst, const AES_GCM_engine* engine, const uint8_t* nonce,
//         const uint8_t* src, uint32_t size, hipStream_t stream);

// void AES_GCM_decrypt(hip_launch_batch_t* batch, uint8_t* dst, const AES_GCM_engine* engine, const uint8_t* nonce,
//         const uint8_t* src, uint32_t size, hipStream_t stream);

// void AES_GCM_next_nonce(hip_launch_batch_t* batch, uint8_t* nonce, hipStream_t stream);

#endif
