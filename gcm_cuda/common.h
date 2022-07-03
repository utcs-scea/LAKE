#ifndef _COMMON_H_
#define _COMMON_H_

#define AES_KEYLEN 32u
#define AES_ROUNDKEYLEN 240u
#define AES_BLOCKLEN 16u
#define AES_MACLEN 12u
#define AES_GCM_STEP 64u

constexpr int kBaseThreadBits = 8;
constexpr int kBaseThreadNum  = 1 << kBaseThreadBits;

#define crypto_aead_aes256gcm_NPUBBYTES 12U
#define crypto_aead_aes256gcm_ABYTES 16U

#endif
