#include <byteswap.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Hashes
#include "jhash.h"
#include "xxhash.h"

#define PAGE_SIZE     (4096)
#define PAGE_INT_SIZE (PAGE_SIZE/4)
#define PAGE_NUM      (1<<18)

uint64_t iter = 1024;
uint32_t PAGE[PAGE_INT_SIZE];
uint32_t PAGES[PAGE_NUM][PAGE_INT_SIZE];

void run_test(size_t len)
{
    uint64_t i;
    uint32_t hash = 0;
    uint64_t hash64 = 0;
    clock_t start, end;

    printf("- - -\n");
    printf("input size: %lu, loop count: %lu\n", len, iter);

    if (len % 4 == 0) {
        start = clock() * 1000000 / CLOCKS_PER_SEC;
        for (i = 0; i < iter; i++) {
            hash = jhash2(&PAGE[hash % (PAGE_INT_SIZE / 2)], len / 4, 17);
        }
        end = clock() * 1000000 / CLOCKS_PER_SEC;

        printf("jhash2:   0x%" PRIx32
               "\t\ttime: %6lu ms, th: %.2f MiB/s\n",
            hash, (end - start) / 1000,
            len * iter * 1.0 / (end - start));
    } else {
        start = clock() * 1000000 / CLOCKS_PER_SEC;
        for (i = 0; i < iter; i++) {
            hash = jhash(&PAGE[hash % (PAGE_INT_SIZE / 2)], len, 17);
        }
        end = clock() * 1000000 / CLOCKS_PER_SEC;

        printf("jhash:    0x%" PRIx32
               "\t\ttime: %6lu ms, th: %.2f MiB/s\n",
            hash, (end - start) / 1000,
            len * iter * 1.0 / (end - start));
    }

    start = clock() * 1000000 / CLOCKS_PER_SEC;
    for (i = 0; i < iter; i++) {
        hash = xxh32(&PAGE[hash % (PAGE_INT_SIZE / 2)], len, 17);
    }
    end = clock() * 1000000 / CLOCKS_PER_SEC;

    printf("xxhash32: 0x%" PRIx32 "\t\ttime: %6lu ms, th: %.2f MiB/s\n",
        hash, (end - start) / 1000, len * iter * 1.0 / (end - start));

    start = clock() * 1000000 / CLOCKS_PER_SEC;
    for (i = 0; i < iter; i++) {
        hash64 = xxh64(&PAGE[hash % (PAGE_INT_SIZE / 2)], len, 17);
    }
    end = clock() * 1000000 / CLOCKS_PER_SEC;

    printf("xxhash64: 0x%" PRIx64 "\ttime: %6lu ms, th: %.2f MiB/s\n",
        hash64, (end - start) / 1000, len * iter * 1.0 / (end - start));
}

void run_page_test(size_t len)
{
    uint64_t i;
    uint32_t hash = 0;
    uint64_t hash64 = 0;
    clock_t start, end;

    printf("- - -\n");
    printf("input page number: %lu, loop count: %lu\n", len, iter);

    start = clock() * 1000000 / CLOCKS_PER_SEC;
    for (i = 0; i < iter; i++) {
        hash = jhash2(PAGES[hash % (PAGE_NUM - len)], len * PAGE_INT_SIZE, 0);
    }
    end = clock() * 1000000 / CLOCKS_PER_SEC;

    printf("jhash2:   0x%" PRIx32
           "\t\ttime: %6lu ms, th: %.2f MiB/s\n",
        hash, (end - start) / 1000,
        len * PAGE_SIZE * iter * 1.0 / (end - start));

    start = clock() * 1000000 / CLOCKS_PER_SEC;
    for (i = 0; i < iter; i++) {
        hash = jhash(PAGES[hash % (PAGE_NUM - len)], len * PAGE_SIZE, 0);
    }
    end = clock() * 1000000 / CLOCKS_PER_SEC;

    printf("jhash:    0x%" PRIx32
           "\t\ttime: %6lu ms, th: %.2f MiB/s\n",
        hash, (end - start) / 1000,
        len * PAGE_SIZE * iter * 1.0 / (end - start));

    start = clock() * 1000000 / CLOCKS_PER_SEC;
    for (i = 0; i < iter; i++) {
        hash = xxh32(PAGES[hash % (PAGE_NUM - len)], len * PAGE_SIZE, 0);
    }
    end = clock() * 1000000 / CLOCKS_PER_SEC;

    printf("xxhash32: 0x%" PRIx32 "\t\ttime: %6lu ms, th: %.2f MiB/s\n",
        hash, (end - start) / 1000,
        len * PAGE_SIZE * iter * 1.0 / (end - start));

    start = clock() * 1000000 / CLOCKS_PER_SEC;
    for (i = 0; i < iter; i++) {
        hash64 = xxh64(PAGES[hash64 % (PAGE_NUM - len)], len * PAGE_SIZE, 0);
    }
    end = clock() * 1000000 / CLOCKS_PER_SEC;

    printf("xxhash64: 0x%" PRIx64 "\ttime: %6lu ms, th: %.2f MiB/s\n",
        hash64, (end - start) / 1000,
        len * PAGE_SIZE * iter * 1.0 / (end - start));
}

void reinit_page(void)
{
    uint32_t i, j;

    for (i = 0; i < PAGE_INT_SIZE; i++)
        PAGE[i] = rand() * rand();

    for (i = 0; i < PAGE_NUM; i++)
        for (j = 0; j < PAGE_INT_SIZE; j++)
            PAGES[i][j] = rand() * rand();
}

int main()
{
    uint64_t i;

    srand(time(NULL));
    iter *= 1024 * 256;

#if 1
    uint32_t input_lenths[]
        = { 3, 4, 8, 11, 12, 16, 17, 33, 36, 64 };

    for (i = 0; i < (sizeof(input_lenths) / sizeof(uint32_t)); i++) {
        reinit_page();
        run_test(input_lenths[i]);
    }
#endif

    iter /= 256;

#if 1
    uint32_t input_page_nums[]
        = { 1, 2, 4, 8 };

    for (i = 0; i < (sizeof(input_page_nums) / sizeof(uint32_t)); i++) {
        reinit_page();
        run_page_test(input_page_nums[i]);
    }
#endif

    return 0;
}
