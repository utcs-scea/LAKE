#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <openssl/aes.h>
#include <cuda.h>
#include <cudaProfiler.h>

#include "ecb.h"

#define AES_MAX_KEYLENGTH     (15 * 16)
#define AES_MAX_KEYLENGTH_U32 (AES_MAX_KEYLENGTH / sizeof(u32))

#ifndef AES_BLOCK_SIZE
#define AES_BLOCK_SIZE        (16)
#endif

#define BPT_BYTES_PER_BLOCK   4096

struct crypto_aes_ctx {
    u32 key_enc[AES_MAX_KEYLENGTH_U32];
    u32 key_dec[AES_MAX_KEYLENGTH_U32];
    u32 key_length;
};

struct crypto_aes_ctx aes_ctx = {
    .key_length = 16,
};

CUdevice device;
CUcontext context;
CUmodule module;
CUfunction encrypt_fn, decrypt_fn;
CUdeviceptr g_plain_text;
CUdeviceptr g_aes_ctx;
CUdeviceptr g_key_enc;
CUdeviceptr g_key_dec;
u32 key_round;
CUstream *g_streams;

char *testfile_name = "testfile.txt";
char *text;
size_t file_size; // in KB
size_t block_size;
size_t stream_size;
int block_x, grid_x;

struct timeval start_t, end_t;

int init_gpu_context(void)
{
    int res;
    int i;

    cuInit(0);
    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[kava] Error: acquire GPU device 0\n");
        return res;
    }

    res = cuCtxCreate(&context, 0, device);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[kava] Error: create GPU context\n");
        return res;
    }

    res = cuModuleLoad(&module, "ecb.cubin");
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[kava] Error: load AES-ECB CUDA module\n");
        return res;
    }

    res = cuModuleGetFunction(&encrypt_fn, module, "_Z15aes_encrypt_bptPjiPh");
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[kava] Error: load encrypt kernel\n");
        return res;
    }
    res = cuModuleGetFunction(&decrypt_fn, module, "_Z15aes_decrypt_bptPjiPh");
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "[kava] Error: load decrypt kernel\n");
        return res;
    }

    cuFuncSetCacheConfig(encrypt_fn, CU_FUNC_CACHE_PREFER_L1);
    cuFuncSetCacheConfig(decrypt_fn, CU_FUNC_CACHE_PREFER_L1);

    block_x = (block_size << 10) >= BPT_BYTES_PER_BLOCK ?
                BPT_BYTES_PER_BLOCK / 16 : (block_size << 10) / 16;
    grid_x = (block_size << 10) >= BPT_BYTES_PER_BLOCK ?
                (block_size << 10) / BPT_BYTES_PER_BLOCK : 1;

    AES_set_encrypt_key("hello_key", aes_ctx.key_length, (AES_KEY *)aes_ctx.key_enc);
    AES_set_decrypt_key("hello_key", aes_ctx.key_length, (AES_KEY *)aes_ctx.key_dec);

    cuMemAlloc(&g_aes_ctx, sizeof(struct crypto_aes_ctx));
    cuMemcpyHtoD(g_aes_ctx, &aes_ctx, sizeof(struct crypto_aes_ctx));

    g_key_enc = (CUdeviceptr)((struct crypto_aes_ctx *)g_aes_ctx)->key_enc;
    g_key_dec = (CUdeviceptr)((struct crypto_aes_ctx *)g_aes_ctx)->key_dec;
    key_round = aes_ctx.key_length / 4 + 6;

    g_streams = (CUstream *)malloc(sizeof(CUstream) * stream_size);
    for (i = 0; i < stream_size; i++) {
        res = cuStreamCreate(&g_streams[i], CU_STREAM_NON_BLOCKING);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "[kava] Error: create stream #%d\n", i);
            return res;
        }
    }

    return 0;
}

void fini_gpu_context(void)
{
    int i;
    for (i = 0; i < stream_size; i++)
        cuStreamDestroy(g_streams[i]);
    free(g_streams);
    cuMemFree(g_aes_ctx);
    cuCtxDestroy(context);
}

void init_text(void)
{
    char *p;
    int i;

    text = (char *)malloc(file_size << 10);
    srand(time(NULL));
    for (p = text; p < text + (file_size << 10); p++) {
        *p = (rand() % 26) + 'a';
    }

#ifdef VERBOSE
    printf("Plain text:\t");
    for (i = 0; i < 32; i++)
        printf("%02hhx ", text[i]);
    printf("\n");
#endif
}

void fini_text(void)
{
    free(text);
}

void test_gpu_write(void)
{
    size_t left_size = file_size;
    size_t write_size;
    char *buf = text;
    char *crypt_text = (char *)malloc(file_size << 10);
    char *buf_c = crypt_text;
    int file = open(testfile_name, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
    int res;
    int i;

    res = cuMemAlloc(&g_plain_text, block_size << 10);

    gettimeofday(&start_t, NULL);

    while (left_size > 0) {
        write_size = (left_size >= block_size) ? block_size : left_size;

        res = cuMemcpyHtoD(g_plain_text, buf, write_size << 10);
        void *args[] = { &g_key_enc, &key_round, &g_plain_text };
        res = cuLaunchKernel(encrypt_fn, grid_x, 1, 1, block_x, 1, 1, 0, NULL, args, NULL);

        res = cuMemcpyDtoH(buf_c, g_plain_text, write_size << 10);
        write(file, buf_c, write_size << 10);

        buf += (write_size << 10);
        buf_c += (write_size << 10);
        left_size -= write_size;
    }

    gettimeofday(&end_t, NULL);
    double duration = (end_t.tv_sec - start_t.tv_sec) +
                        (end_t.tv_usec - start_t.tv_usec) / 1000000.0;
    printf("Write throughput:\t %lf MiB/s\n", file_size * 1.0 / duration / 1024);

    res = cuMemFree(g_plain_text);
    fsync(file);
    close(file);

#ifdef VERBOSE
    printf("Crypto text:\t");
    for (i = 0; i < 32; i++)
        printf("%02hhx ", crypt_text[i]);
    printf("\n");
#endif

    free(crypt_text);
}

void test_gpu_write_async(void)
{
    size_t left_size = file_size;
    size_t write_size;
    size_t offset = 0;
    char *crypt_text = (char *)malloc(file_size << 10);
    int file = open(testfile_name, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
    int res;
    int i;
    int stream_id = 0;
    CUstream stream;

    res = cuMemAlloc(&g_plain_text, file_size << 10);

    gettimeofday(&start_t, NULL);

    while (left_size > 0) {
        CUdeviceptr g_text = (CUdeviceptr)((char *)g_plain_text + offset);
        write_size = (left_size >= block_size) ? block_size : left_size;

        if (stream_id == stream_size - 1) {
            stream_id = 0;
        }
        stream = g_streams[stream_id++];
        res = cuMemcpyHtoDAsync(g_text, text + offset, write_size << 10, stream);
        void *args[] = { &g_key_enc, &key_round, &g_text };
        res = cuLaunchKernel(encrypt_fn, grid_x, 1, 1, block_x, 1, 1, 0, stream, args, NULL);

        res = cuMemcpyDtoHAsync(crypt_text + offset, g_text, write_size << 10, stream);

        offset += (write_size << 10);
        left_size -= write_size;
    }

    cuCtxSynchronize();
    write(file, crypt_text, file_size << 10);

    gettimeofday(&end_t, NULL);
    double duration = (end_t.tv_sec - start_t.tv_sec) +
                        (end_t.tv_usec - start_t.tv_usec) / 1000000.0;
    printf("Async write throughput:\t %lf MiB/s\n", file_size * 1.0 / duration / 1024);

    res = cuMemFree(g_plain_text);
    fsync(file);
    close(file);

#ifdef VERBOSE
    printf("Crypto text:\t");
    for (i = 0; i < 32; i++)
        printf("%02hhx ", crypt_text[i]);
    printf("\n");
#endif

    free(crypt_text);
}

void test_gpu_read()
{
    int file = open(testfile_name, O_RDONLY, NULL);
    char *crypt_text = (char *)malloc(file_size << 10);
    char *buf_c = crypt_text;
    char *trans_text = (char *)malloc(file_size << 10);
    char *buf_t = trans_text;
    size_t left_size = file_size;
    size_t read_size;
    int i;
    int res;

    res = cuMemAlloc(&g_plain_text, block_size << 10);

    gettimeofday(&start_t, NULL);

    while (left_size > 0) {
        read_size = (left_size >= block_size) ? block_size : left_size;
        read_size = read(file, buf_c, read_size << 10);
        if (read_size <= 0 || read_size % (1<<10) != 0) {
            fprintf(stderr, "Error: Read from file %s\n", testfile_name);
            break;
        }

        res = cuMemcpyHtoD(g_plain_text, buf_c, read_size);
        void *args_2[] = { &g_key_dec, &key_round, &g_plain_text };
        res = cuLaunchKernel(decrypt_fn, grid_x, 1, 1, block_x, 1, 1, 0, NULL, args_2, NULL);

        res = cuMemcpyDtoH(buf_t, g_plain_text, read_size);

        buf_c += read_size;
        buf_t += read_size;
        left_size -= (read_size >> 10);
    }

    gettimeofday(&end_t, NULL);
    double duration = (end_t.tv_sec - start_t.tv_sec) +
                        (end_t.tv_usec - start_t.tv_usec) / 1000000.0;
    printf("Read throughput:\t %lf MiB/s\n", file_size / duration / 1024);

    res = cuMemFree(g_plain_text);
    close(file);

#ifdef VERBOSE
    printf("Decrypt text:\t");
    for (i = 0; i < 32; i++)
        printf("%02hhx ", trans_text[i]);
    printf("\n");
#endif

    // Verify
    for (i = 0; i < file_size; i++)
        if (trans_text[i] != text[i]) {
            printf("Conflict byte at %d: buf[%d]=%hhx but orig_buf[%d]=%hhx\n",
                    i, i, trans_text[i], i, text[i]);
            break;
        }

    free(trans_text);
    free(crypt_text);
}

void test_gpu_read_async()
{
    int file = open(testfile_name, O_RDONLY, NULL);
    char *crypt_text = (char *)malloc(file_size << 10);
    char *trans_text = (char *)malloc(file_size << 10);
    size_t offset = 0;
    size_t left_size = file_size;
    size_t read_size;
    int i;
    int stream_id = 0;
    CUstream stream;

    cuMemAlloc(&g_plain_text, file_size << 10);

    gettimeofday(&start_t, NULL);

    read_size = read(file, crypt_text, file_size << 10);
    if (read_size != (file_size << 10)) {
        fprintf(stderr, "Error: Read from file %s\n", testfile_name);
        return;
    }

    while (left_size > 0) {
        CUdeviceptr g_text = (CUdeviceptr)((char *)g_plain_text + offset);
        read_size = (left_size >= block_size) ? block_size : left_size;

        if (stream_id == stream_size - 1) {
            stream_id = 0;
        }
        stream = g_streams[stream_id++];
        cuMemcpyHtoDAsync(g_text, crypt_text + offset, read_size << 10, stream);
        void *args_2[] = { &g_key_dec, &key_round, &g_text };
        cuLaunchKernel(decrypt_fn, grid_x, 1, 1, block_x, 1, 1, 0, stream, args_2, NULL);

        cuMemcpyDtoHAsync(trans_text + offset, g_text, read_size << 10, stream);

        offset += (read_size << 10);
        left_size -= read_size;
    }

    gettimeofday(&end_t, NULL);
    cuCtxSynchronize();

    //gettimeofday(&end_t, NULL);
    double duration = (end_t.tv_sec - start_t.tv_sec) +
                        (end_t.tv_usec - start_t.tv_usec) / 1000000.0;
    printf("Async read throughput:\t %lf MiB/s\n", file_size / duration / 1024);

    cuMemFree(g_plain_text);
    close(file);

#ifdef VERBOSE
    printf("Decrypt text:\t");
    for (i = 0; i < 32; i++)
        printf("%02hhx ", trans_text[i]);
    printf("\n");
#endif

    // Verify
    for (i = 0; i < file_size; i++)
        if (trans_text[i] != text[i]) {
            printf("Conflict byte at %d: buf[%d]=%hhx but orig_buf[%d]=%hhx\n",
                    i, i, trans_text[i], i, text[i]);
            break;
        }

    free(trans_text);
    free(crypt_text);
}

int main(int argc, char *argv[]) {
    file_size = (16 << 10); //  16 MB
    block_size = 128;       // 128 KB
    stream_size = 32;

    if (argc < 2 || file_size % atoi(argv[1]) != 0) {
        fprintf(stderr, "Usage: %s <block_size (KB)>\nDefault block size=128\n", argv[0]);
    }
    else {
        block_size = atoi(argv[1]);
    }

    //cuProfilerStart();

    assert(init_gpu_context() == 0);
    init_text();

    // Test
    sync();
    test_gpu_write();
    sync();
    test_gpu_read();
    sync();
    test_gpu_write_async();
    sync();
    test_gpu_read_async();

    // Free
    fini_text();
    fini_gpu_context();
    //cuProfilerStop();

    return 0;
}
