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


#endif
