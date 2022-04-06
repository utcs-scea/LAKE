#ifndef __KAVA_ECB_H__
#define __KAVA_ECB_H__

#define KAVA_ENABLE             (1)
#define KAVA_ECB_SIZE_THRESHOLD (PAGE_SIZE - 1)

#include <crypto/aes.h>

#define BPT_BYTES_PER_BLOCK 4096

enum kava_crypt_type {
    KAVA_ENCRYPT,
    KAVA_DECRYPT,
};

static inline void cvt_cpu_to_be32(u32 *key, int n) {
    int i;

    for (i = 0; i < n; i++)
        key[i] = cpu_to_be32(key[i]);
}

struct kthread_ecb_arg {
    struct blkcipher_desc *desc;
    struct scatterlist *dst;
    struct scatterlist *src;
    unsigned int nbytes;
    unsigned int offset;
    enum kava_crypt_type crypt_type;
};

#endif
