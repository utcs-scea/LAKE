#include <sys/mman.h>
#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <jhash.h>
#include <chrono>
#include <iostream>
#include <sys/time.h>

#define PAGE_SIZE 4096 // getpagesize()

uint64_t current_timestamp() {
    //struct timeval te; 
    //gettimeofday(&te, NULL); // get current time_cont
    struct timespec te; 
    clock_gettime(CLOCK_MONOTONIC, &te);
    uint64_t milliseconds = te.tv_sec*1000000000 + te.tv_nsec; // calculate ns
    return milliseconds;
}

int main(int argc, char **argv)
{
    int res;
    uint64_t seed = 17;
    uint64_t max_concurrency = 1;
    //uint64_t batch_size = 4096;
    uint64_t batch_size = 128;
    uint64_t n_pages = 128 * batch_size;
    uint64_t grid_x = 4;
    uint64_t grid_y = 4;
    uint64_t grid_z = 4;
    uint64_t block_z = 1;
    uint64_t block_y = 1;
    uint64_t block_x = batch_size / (grid_x * grid_y * grid_z * block_y * block_z);

    uint64_t start_app, start_exec, end;
    start_app = current_timestamp();

    // CUDA setup
    res = cuInit(0);
    if (res)
    {
        printf("Error cuinit (%d)\n", res);
    }
    CUdevice cuDevice;
    res = cuDeviceGet(&cuDevice, 0);
    if (res)
    {
        printf("Error getDev (%d)\n", res);
    }
    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res)
    {
        printf("Error ctxCreat (%d)\n", res);
    }
    CUmodule cuModule;
    res = cuModuleLoad(&cuModule, "jhash.cubin");
    if (res)
    {
        printf("Error loadMod (%d), %s\n", res, cudaGetErrorString((cudaError_t)res));
    }
    CUstream streams[max_concurrency];
    for (int i = 0; i < max_concurrency; ++i)
    {
        res = cuStreamCreate(&streams[i], 0);
        if (res)
        {
            printf("Error streamCreate (%d), stream no %d\n", res, i);
        }
    }

    // Create a model identical page
    char *page = (char *)malloc(PAGE_SIZE);
    for (int i = 0; i < PAGE_SIZE; ++i)
    {
        page[i] = i;
    }

    // mmap pages
    void *pages = mmap(NULL, PAGE_SIZE * n_pages,
                       PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    for (uint64_t i = 0; i < n_pages; ++i)
    {
        char *page_start = ((char *)pages) + PAGE_SIZE * i;
        memcpy(page_start, page, PAGE_SIZE);
    }


    // alloc checksum buf
    uint32_t *h_checksum = (uint32_t *)malloc(batch_size * max_concurrency * sizeof(uint32_t));

    void *h_pages;
    h_pages = malloc(batch_size * PAGE_SIZE * max_concurrency);
    if (!h_pages)
        printf("Couldn't allocatepages\n");

    // Device side mem
    CUdeviceptr d_pages;
    CUdeviceptr d_checksum;
    res = cuMemAlloc(&d_pages, batch_size * max_concurrency * PAGE_SIZE);
    if (res) {
        printf("Error memalloc 1 (%d)\n", res);
    }
    res = cuMemAlloc(&d_checksum, batch_size * max_concurrency * sizeof(uint32_t));
    if (res)
    {
        printf("Error memalloc 2 (%d)\n", res);
    }

    CUfunction xxh;
    res = cuModuleGetFunction(&xxh, cuModule, "_Z6jhash2PvPj");
    if (res != CUDA_SUCCESS)
    {
        printf("Error getting function (%d)\n", res);
    }

    // Measure tpt
    //uint64_t n_samples = 5000;
    int n_iterations = 2000;
    //uint64_t batches_per_sample = n_iterations * n_pages * max_concurrency / (n_samples * batch_size);
    //uint64_t *pages_checksummed = (uint64_t *)malloc(n_samples * sizeof(uint64_t));
    //std::chrono::high_resolution_clock::time_point *times = 
    //    (std::chrono::high_resolution_clock::time_point *)malloc(n_samples * sizeof(std::chrono::high_resolution_clock::time_point));
    printf("allocating %d\n", n_iterations * 2 *(n_pages/batch_size));
    uint64_t *pages_checksummed = (uint64_t *)malloc(n_iterations * 2 *(n_pages/batch_size));
    uint64_t *times = (uint64_t *)malloc(n_iterations * 2*(n_pages/batch_size));
    double total_memcpy_time = 0.0;
    // Time start
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();

    int j = 0;
    uint64_t i = 0;
    int sample_cnt = 0;

    start_exec = current_timestamp();

    // Run all the batches on GPU
    for (int k = 0; k < n_iterations; ++k)
    {
        //for every batch size in total # of pages, 128 (n_pages) times
        for (i = 0; i < n_pages; i += batch_size * max_concurrency) 
        {
            CUdeviceptr concur_pages = d_pages + 0 * batch_size * PAGE_SIZE;
            CUdeviceptr concur_checksum = d_checksum + 0 * batch_size * sizeof(uint32_t);
            for (j = 0; j < max_concurrency; ++j) //by default only one
            {
                if (j == 0 && k == 0) {
                    //memcpy(((char *)h_pages) + j * batch_size * PAGE_SIZE, ((char *)pages) + (i + j * batch_size) * PAGE_SIZE, batch_size * PAGE_SIZE);
                    // Copy to dev
                    res = cuMemcpyHtoDAsync(concur_pages, h_pages, batch_size * PAGE_SIZE, streams[j]);
                    if (res)
                        printf("Error memcpy htod 1 (%d)\n", res);
                    cuStreamSynchronize(streams[j]);
                    pages_checksummed[0] = 0;
                    //times[0] = std::chrono::high_resolution_clock::now();
                    times[0] = current_timestamp();
                }

                // Launch kernel
                void *args[] = {&concur_pages, &concur_checksum};
                res = cuLaunchKernel(xxh, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                                     0, streams[j], args, NULL);
                if (res)
                    printf("Error launching kernel (%d)\n", res);
    

                if (res)
                    printf("Error memcpy dtoh (%d)\n", res);
            }
            res = cuMemcpyDtoHAsync(h_checksum + 0 * batch_size, concur_checksum,
                                            batch_size * sizeof(uint32_t), streams[0]);
        }

        cuStreamSynchronize(streams[0]);
        //pages_checksummed[sample_cnt + 1] = pages_checksummed[sample_cnt] + batches_per_sample * batch_size;
        pages_checksummed[sample_cnt + 1] = n_pages;
        //times[sample_cnt + 1] = std::chrono::high_resolution_clock::now();
        times[sample_cnt + 1] = current_timestamp();
        sample_cnt++;
    }

    cuCtxDestroy(cuContext);

    // Time end
    // std::chrono::high_resolution_clock::time_point t2 =
    //     std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> gpu_time =
    //     std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    // ============================ End GPU checksum ============================

    // Check results
    //   uint32_t checksum = jhash2((uint32_t *) pages, PAGE_SIZE / sizeof(uint32_t), seed);
    //   for (int i = 0; i < batch_size * max_concurrency; ++i) {
    //     if (checksum != h_checksum[i]) {
    //       printf("Checksums don't match, idx = %d, should be %x, is %x\n", i,
    //         checksum, h_checksum[i]);
    //     }
    //   }

    // ============================ CPU checksum ============================
    // uint32_t *host_checksum_vals = (uint32_t *)malloc(n_pages * PAGE_SIZE);

    // // Time start
    // std::chrono::high_resolution_clock::time_point t3 =
    //     std::chrono::high_resolution_clock::now();

    // // hash all the pages on cpu
    // for (int i = 0; i < n_pages; ++i)
    // {
    //     uint32_t *h_page = ((uint32_t *)pages) + i / sizeof(uint32_t);
    //     host_checksum_vals[i] = jhash2(h_page, PAGE_SIZE, seed);
    // }

    // // Time end
    // std::chrono::high_resolution_clock::time_point t4 =
    //     std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> cpu_time =
    //     std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3);

    // ============================ End CPU checksum ============================

    end = current_timestamp();

    free(h_pages);

    // Print tpt
    FILE *f = fopen("uspace.out", "w");

    fprintf(f, "%lu,%lu,%lu\n", start_app, start_exec, end);
    for (int l = 1; l < sample_cnt; ++l) {
        fprintf(f, "%lu,%lu\n", times[l], pages_checksummed[l]);
        //fprintf(f, "%f,%f\n", std::chrono::duration_cast<std::chrono::duration<double>>(times[l] - times[0]).count(), (double)(pages_checksummed[l] - pages_checksummed[l - 1]) / (std::chrono::duration_cast<std::chrono::duration<double>>(times[l] - times[l - 1]).count()));
        // printf("%f,%f\n", std::chrono::duration_cast<std::chrono::duration<double>>(times[l] - times[0]).count(), (double) (pages_checksummed[l] - pages_checksummed[l - 1]) / (std::chrono::duration_cast<std::chrono::duration<double>>(times[l] - times[l - 1]).count()));
    }

    printf("total time %lu ms\n", (end-start_app)/1000);
    printf("total gpu time %lu ms\n", (end-start_exec)/1000);
    fclose(f);
}
