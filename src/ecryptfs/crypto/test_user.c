/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */



#include "gcm_cuda.h"
#include <stdio.h>
#include <stdlib.h>

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
        return -1;
    }

    struct AES_GCM_engine_ctx ctx;
    lake_AES_GCM_init_fns(&ctx, argv[1]);
    lake_AES_GCM_init(&ctx);

    u8 key[AESGCM_KEYLEN];
    for (int i = 0 ; i < AESGCM_KEYLEN ; i++)
        key[i] = i % 255;

    lake_AES_GCM_setkey(&ctx, key);

    u8 input[n*PAGE_SIZE];
    for (int i = 0 ; i < n*PAGE_SIZE ; i++)
        input[i] = i % 255;

    CUdeviceptr src, dst, dec;
    lake_AES_GCM_alloc_pages(&src, n*PAGE_SIZE);
    lake_AES_GCM_alloc_pages(&dst, n*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES));

    lake_AES_GCM_copy_to_device(src, input, n*PAGE_SIZE);
    lake_AES_GCM_encrypt(&ctx, dst, src, n*PAGE_SIZE);
    cuCtxSynchronize();

    lake_AES_GCM_alloc_pages(&dec, n*(PAGE_SIZE));
    lake_AES_GCM_decrypt(&ctx, dec, dst, n*PAGE_SIZE);
    cuCtxSynchronize();
    lake_AES_GCM_copy_from_device(input, dec, n*PAGE_SIZE);
    cuCtxSynchronize();

    for (int i = 0 ; i < n*PAGE_SIZE ; i++)
        if(input[i] != i % 255) {
            printf("Wrong value at idx %d (%d != %d)\n", i, input[i], i%255);
            return -1;
        }

    printf("User test passed!\n");
    return 0;
}


