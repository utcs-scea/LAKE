#include "gcm_cuda.h"
#include <stdio.h>
#include <stdlib.h>

u32 PAGE_SIZE = 4096;
u32 n = 16;

void gen_pages(char* buf, int n) {
    srand(123);
    for (int i = 0 ; i < n*PAGE_SIZE ; i++) {
        if (i < PAGE_SIZE) buf[i] = 0;
        else buf[i] = rand();
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Need argument: path to cubin");
    }

    struct AES_GCM_engine_ctx ctx;
    lake_AES_GCM_init_fns(&ctx, argv[1]);
    lake_AES_GCM_init(&ctx);

    u8 key[AESGCM_KEYLEN];
    for (int i = 0 ; i < AESGCM_KEYLEN ; i++)
        key[i] = i % 255;

    lake_AES_GCM_setkey(&ctx, key);

    u8 input[n*PAGE_SIZE];
    for (int i = 0 ; i < AESGCM_KEYLEN ; i++)
        input[i] = i % 255;

    CUdeviceptr src, dst, dec;
    lake_AES_GCM_alloc_pages(&src, n*PAGE_SIZE);
    lake_AES_GCM_alloc_pages(&dst, n*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));
    
    lake_AES_GCM_encrypt(&ctx, src, dst, n*PAGE_SIZE);

    lake_AES_GCM_alloc_pages(&dec, n*(PAGE_SIZE));

    

}


