#include <random>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include <thread>
#include <sstream>
#include <iostream>

#include "nvcomp/lz4.h"
#include "nvcomp/snappy.h"
#include "nvcomp/gdeflate.h"
#include "lz4.h"

uint32_t PAGE_SIZE = 4096;
/********************************************/
// manually set these numbers
/********************************************/
int max_batch = 2048;
int batch_sizes[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
uint64_t page_bytes = max_batch*PAGE_SIZE;
int nrounds = 2;
int nwarm = 1;
/********************************************/


void gen_pages(char* buf, int n) {
    srand(123);
    for (int i = 0 ; i < n*PAGE_SIZE ; i++) {
        if (i < PAGE_SIZE) buf[i] = 0;
        else buf[i] = rand();
    }
}

void lz4_cpu(char* pages, std::stringstream& csv) {
    const int max_dst_size = LZ4_compressBound(PAGE_SIZE);

    csv << "CPU_LZ4";

    for (int &batch_size : batch_sizes) {
        char* tmp = (char*) malloc(batch_size*max_dst_size);
        uint64_t time_sum = 0;
        for (int i = 0 ; i < nrounds+nwarm ; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200)); 
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            
            for (int j = 0 ; j < batch_size ; j++) {
                //compress a single page
                char* src = pages + (j*PAGE_SIZE);
                char* dst = tmp + (j*max_dst_size);
                LZ4_compress_default(src, dst, PAGE_SIZE, max_dst_size);
            }

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

            if (i >= nwarm) {
                time_sum += total_time;
            }
        }

        std::cout << "Avg CPU time for " << batch_size << ": " << time_sum/nrounds << std::endl;
        csv << "," <<  time_sum/nrounds;
        free(tmp);
    }

    csv << std::endl;
}

struct gpu_compress_ops {
    const char* name;
    nvcompStatus_t (*get_temp_size)(
        size_t batch_size,
        size_t max_chunk_bytes,
        size_t * temp_bytes);

    nvcompStatus_t (*get_max_output_size)(
        const size_t max_chunk_size,
        size_t* const max_compressed_size);

    nvcompStatus_t (*compress_async)(  
        const void* const* device_uncompressed_ptrs,    
        const size_t* device_uncompressed_bytes,  
        size_t max_uncompressed_chunk_bytes,  
        size_t batch_size,  
        void* device_temp_ptr,  
        size_t temp_bytes,  
        void* const* device_compressed_ptrs,  
        size_t* device_compressed_bytes,  
        cudaStream_t stream);
};

/* 
 *    LZ4 ops
 */
nvcompStatus_t lz4_get_temp_size(size_t batch_size, size_t max_chunk_bytes, size_t* temp_bytes) {
    return nvcompBatchedLZ4CompressGetTempSize(batch_size, max_chunk_bytes, nvcompBatchedLZ4DefaultOpts, temp_bytes);
        
}

nvcompStatus_t lz4_get_max_output_size(const size_t max_chunk_size, size_t* const max_compressed_size) {
    return nvcompBatchedLZ4CompressGetMaxOutputChunkSize(max_chunk_size, nvcompBatchedLZ4DefaultOpts, max_compressed_size);
}

nvcompStatus_t lz4_compress_async(  
        const void* const* device_uncompressed_ptrs,    
        const size_t* device_uncompressed_bytes,  
        size_t max_uncompressed_chunk_bytes,  
        size_t batch_size,  
        void* device_temp_ptr,  
        size_t temp_bytes,  
        void* const* device_compressed_ptrs,  
        size_t* device_compressed_bytes,  
        cudaStream_t stream) {
    return nvcompBatchedLZ4CompressAsync(  
                device_uncompressed_ptrs,    
                device_uncompressed_bytes,  
                max_uncompressed_chunk_bytes, // The maximum chunk size  
                batch_size,  
                device_temp_ptr,  
                temp_bytes,  
                device_compressed_ptrs,  
                device_compressed_bytes,  
                nvcompBatchedLZ4DefaultOpts,  
                stream);
}

struct gpu_compress_ops lz4_ops = {
    .name                = "LZ4",
    .get_temp_size       = lz4_get_temp_size,
    .get_max_output_size = lz4_get_max_output_size,
    .compress_async      = lz4_compress_async
};


/* 
 *    Snappy ops
 */

nvcompStatus_t snappy_get_temp_size(size_t batch_size, size_t max_chunk_bytes, size_t* temp_bytes) {
    return nvcompBatchedSnappyCompressGetTempSize(batch_size, max_chunk_bytes, nvcompBatchedSnappyDefaultOpts, temp_bytes);
        
}

nvcompStatus_t snappy_get_max_output_size(const size_t max_chunk_size, size_t* const max_compressed_size) {
    return nvcompBatchedSnappyCompressGetMaxOutputChunkSize(max_chunk_size, nvcompBatchedSnappyDefaultOpts, max_compressed_size);
}

nvcompStatus_t snappy_compress_async(  
        const void* const* device_uncompressed_ptrs,    
        const size_t* device_uncompressed_bytes,  
        size_t max_uncompressed_chunk_bytes,  
        size_t batch_size,  
        void* device_temp_ptr,  
        size_t temp_bytes,  
        void* const* device_compressed_ptrs,  
        size_t* device_compressed_bytes,  
        cudaStream_t stream) {
    return nvcompBatchedSnappyCompressAsync(  
                device_uncompressed_ptrs,    
                device_uncompressed_bytes,  
                max_uncompressed_chunk_bytes, // The maximum chunk size  
                batch_size,  
                device_temp_ptr,  
                temp_bytes,  
                device_compressed_ptrs,  
                device_compressed_bytes,  
                nvcompBatchedSnappyDefaultOpts,  
                stream);
}

struct gpu_compress_ops snappy_ops = {
    .name                = "Snappy",
    .get_temp_size       = snappy_get_temp_size,
    .get_max_output_size = snappy_get_max_output_size,
    .compress_async      = snappy_compress_async
};


/* 
 *    GDeflate ops
 */

nvcompStatus_t gdeflate_get_temp_size(size_t batch_size, size_t max_chunk_bytes, size_t* temp_bytes) {
    return nvcompBatchedGdeflateCompressGetTempSize(batch_size, max_chunk_bytes, nvcompBatchedGdeflateDefaultOpts, temp_bytes);
        
}

nvcompStatus_t gdeflate_get_max_output_size(const size_t max_chunk_size, size_t* const max_compressed_size) {
    return nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(max_chunk_size, nvcompBatchedGdeflateDefaultOpts, max_compressed_size);
}

nvcompStatus_t gdeflate_compress_async(  
        const void* const* device_uncompressed_ptrs,    
        const size_t* device_uncompressed_bytes,  
        size_t max_uncompressed_chunk_bytes,  
        size_t batch_size,  
        void* device_temp_ptr,  
        size_t temp_bytes,  
        void* const* device_compressed_ptrs,  
        size_t* device_compressed_bytes,  
        cudaStream_t stream) {
    return nvcompBatchedGdeflateCompressAsync(  
                device_uncompressed_ptrs,    
                device_uncompressed_bytes,  
                max_uncompressed_chunk_bytes, // The maximum chunk size  
                batch_size,  
                device_temp_ptr,  
                temp_bytes,  
                device_compressed_ptrs,  
                device_compressed_bytes,  
                nvcompBatchedGdeflateDefaultOpts,  
                stream);
}

struct gpu_compress_ops gdeflate_ops = {
    .name                = "Gdeflate",
    .get_temp_size       = gdeflate_get_temp_size,
    .get_max_output_size = gdeflate_get_max_output_size,
    .compress_async      = gdeflate_compress_async
};


void compress_gpu(struct gpu_compress_ops& ops, char* pages, std::stringstream& csv) {
 
    csv << "GPU_" << ops.name;

    char* d_pages;
    cudaMalloc((void**)&d_pages, page_bytes);
    cudaMemcpy(d_pages, pages, page_bytes, cudaMemcpyHostToDevice);

    for (int &batch_size : batch_sizes) {
        //set compressed ptrs on host
        void* host_uncompressed_ptrs[batch_size];
        size_t host_uncompressed_bytes[batch_size];
        for (int i = 0 ; i < batch_size ; i++) {
            host_uncompressed_ptrs[i] = d_pages + (PAGE_SIZE*i);
            host_uncompressed_bytes[i] = PAGE_SIZE;
        }

        //set ptrs on device
        void** d_uncompressed_ptrs;
        size_t* d_uncompressed_bytes;
        cudaMalloc((void**)&d_uncompressed_ptrs,  batch_size*sizeof(void*));
        cudaMalloc((void**)&d_uncompressed_bytes, batch_size*sizeof(size_t));
        cudaMemcpy(d_uncompressed_ptrs,  host_uncompressed_ptrs,  batch_size*sizeof(void*),  cudaMemcpyHostToDevice);
        cudaMemcpy(d_uncompressed_bytes, host_uncompressed_bytes, batch_size*sizeof(size_t), cudaMemcpyHostToDevice);

        //get temp size
        size_t temp_bytes;
        ops.get_temp_size(batch_size, PAGE_SIZE, &temp_bytes);
        void* d_temp_ptr;
        cudaMalloc(&d_temp_ptr, temp_bytes);

        // get the max output size for each chunk
        size_t max_out_bytes;
        ops.get_max_output_size(PAGE_SIZE, &max_out_bytes);

        void** host_compressed_ptrs;
        cudaMallocHost((void**)&host_compressed_ptrs, sizeof(size_t) * batch_size);
        for(size_t ix_chunk = 0; ix_chunk < batch_size; ++ix_chunk) {
            cudaMalloc(&host_compressed_ptrs[ix_chunk], max_out_bytes);
        }

        void** d_compressed_ptrs;
        cudaMalloc((void**)&d_compressed_ptrs, sizeof(size_t) * batch_size);
        cudaMemcpy(d_compressed_ptrs, host_compressed_ptrs, 
            sizeof(size_t) * batch_size,cudaMemcpyHostToDevice);

        size_t* d_compressed_bytes;
        cudaMalloc((void**)&d_compressed_bytes, sizeof(size_t) * batch_size);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        std::this_thread::sleep_for(std::chrono::milliseconds(100)); 

        uint64_t time_sum = 0;
        for (int i = 0 ; i < nrounds+nwarm ; i++) {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            nvcompStatus_t comp_res = ops.compress_async(  
                d_uncompressed_ptrs,    
                d_uncompressed_bytes,  
                PAGE_SIZE, // The maximum chunk size  
                batch_size,  
                d_temp_ptr,  
                temp_bytes,  
                d_compressed_ptrs,  
                d_compressed_bytes,  
                stream);

            if (comp_res != nvcompSuccess) {
                std::cerr << "Failed compression!" << std::endl;
                assert(comp_res == nvcompSuccess);
            }
            cudaStreamSynchronize(stream);

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

            // only record non-warmups
            if (i >= nwarm) {
                time_sum += total_time;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
        }

        // if very last one, check compression ratio
        if (batch_size == max_batch) {
            size_t compressed_bytes[max_batch];
            cudaMemcpy(compressed_bytes, d_compressed_bytes, sizeof(size_t) * max_batch, cudaMemcpyDeviceToHost);
            //std::cout << "Compressed bytes: ";
            //for (int k = 0 ; k < max_batch ; k++) {
            //    std::cout << compressed_bytes[k] << ", ";
            //}
            //std::cout << std::endl;

            double sum=0, max=0, min=0;
            for (int k = 0 ; k < max_batch ; k++) {
                double ratio = PAGE_SIZE / compressed_bytes[k];

                if (ratio > max) max = ratio;
                if (ratio < min || min == 0) min = ratio;
                sum += ratio;
            }
            std::cout << "Stats for compression algorithm " << ops.name << " (avg, max, min):" << std::endl;
            std::cout << "   " << sum/max_batch << ", " << max << ", " << min << std::endl;
        }

        std::cout << "Avg GPU time for " << batch_size << " on " << ops.name <<  ": " << time_sum/nrounds << std::endl;
        csv << "," <<  time_sum/nrounds;

        cudaFree(d_uncompressed_ptrs);
        cudaFree(d_uncompressed_bytes);
        cudaFree(d_temp_ptr);
        cudaFreeHost(host_compressed_ptrs);
        cudaFree(d_compressed_ptrs);
        cudaFree(d_compressed_bytes);
    }

    cudaFree(d_pages);
    csv << std::endl;
}


int main() {
    std::stringstream csv;
    char* pages;
    
    pages = (char*) malloc(page_bytes);
    gen_pages(pages, max_batch);

    lz4_cpu(pages, csv);
    compress_gpu(lz4_ops, pages, csv);
    compress_gpu(snappy_ops, pages, csv);
    compress_gpu(gdeflate_ops, pages, csv);

    std::cout << "CSV:\n" << csv.str();

    free(pages);
}