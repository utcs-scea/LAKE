#include <stdio.h>
#include <cuda.h>
#include <random>
#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <chrono>
#include <thread>
#include <sstream>
#include <iostream>

#include "gcm_cuda/aes_gcm.h"
#include "crypto_ecb/ecb.h"

uint32_t PAGE_SIZE = 4096;
/********************************************/
// manually set these numbers
/********************************************/
uint64_t max_batch = 4096;
//int block_sizes[] = {1,2,4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}; //how many pages to en/decrypt
uint64_t block_sizes[] = {1,2,4,8,16,32,64, 128, 256, 512, 1024, 2048, 4096}; //how many pages to en/decrypt
uint64_t multiplier = 1; //how many kernels en/decrypting the total size we launch concurrently
int nrounds = 1;
int nwarm = 0;
/********************************************/
char* pages;

void gen_pages(char* buf, int n) {
    srand(123);
    for (int i = 0 ; i < n*PAGE_SIZE ; i++) {
        if (i < PAGE_SIZE) buf[i] = 0;
        else buf[i] = rand();
    }
}

uint8_t gcm_key[AES_KEYLEN];
uint8_t *d_gcm_input, *d_gcm_cypher;
void gcm_setup() {
    srand(0);
    for (int i = 0 ; i < AES_KEYLEN ; i++) {
        gcm_key[i] = i%255;
    }
    AES_GCM_init(gcm_key);

    gpuErrchk(cudaMalloc(&d_gcm_cypher, max_batch*multiplier*(PAGE_SIZE+crypto_aead_aes256gcm_ABYTES)));
    gpuErrchk(cudaMalloc(&d_gcm_input,  max_batch*multiplier*PAGE_SIZE));

    gpuErrchk(cudaMemcpy(d_gcm_input, pages, max_batch*multiplier*PAGE_SIZE, cudaMemcpyHostToDevice));
}

void gcm_clean(){
    cudaFree(d_gcm_input);
    cudaFree(d_gcm_cypher);
}

void gcm_encrypt(uint64_t block_size) {    
    uint64_t sz = block_size*PAGE_SIZE;
    for (int i = 0 ; i < multiplier ; i++) {
        AES_GCM_encrypt(
            d_gcm_cypher+ i*sz, //dst
            d_gcm_input+ i*(sz+crypto_aead_aes256gcm_ABYTES), //src
            sz);
    }
}

void gcm_decrypt(uint64_t block_size) {    
    uint64_t sz = block_size*PAGE_SIZE;
    for (int i = 0 ; i < multiplier ; i++) {
        AES_GCM_decrypt(
            d_gcm_input+ i*sz, //dst
            d_gcm_cypher+ i*(sz+crypto_aead_aes256gcm_ABYTES),
            sz);
    }
}

uint8_t *d_ecb_input, *d_ecb_cypher;
void ecb_setup() {
    ECB_init();

    gpuErrchk(cudaMalloc(&d_ecb_cypher, max_batch*multiplier*PAGE_SIZE));
    gpuErrchk(cudaMalloc(&d_ecb_input,  max_batch*multiplier*PAGE_SIZE));
    gpuErrchk(cudaMemcpy(d_ecb_input, pages, max_batch*multiplier*PAGE_SIZE, cudaMemcpyHostToDevice));
}

void ecb_clean(){
    cudaFree(d_ecb_input);
    cudaFree(d_ecb_cypher);
}

void ecb_encrypt(uint64_t block_size) {
    uint64_t sz = block_size*PAGE_SIZE;
    for (int i = 0 ; i < multiplier ; i++) {
        ECB_encrypt(
            d_ecb_cypher+ i*sz, //dst
            d_ecb_input+ i*sz, //src
            sz);
    }
}

void ecb_decrypt(uint64_t block_size) {
    uint64_t sz = block_size*PAGE_SIZE;
    for (int i = 0 ; i < multiplier ; i++) {
        ECB_decrypt(
            d_ecb_input+ i*sz, //src
            d_ecb_cypher+ i*sz, //dst
            sz);
    }
}

#define GCM 0
#define ECB 0
int main() {
    std::stringstream csv, dec_csv;
    pages = (char*) malloc(max_batch*PAGE_SIZE*multiplier);
    gen_pages(pages, max_batch);
    
    for (uint64_t &block_size : block_sizes) {
        csv << "," << block_size << "*" << multiplier;
        dec_csv << "," << block_size << "*" << multiplier;
    }
    csv << "\n";
    dec_csv << "\n";

    for(int alg = 0 ; alg < 2 ; alg++) {
        if (alg == GCM) { 
            csv << "gcm";
            dec_csv << "gcm";
            gcm_setup();
        }
        else {
            csv << "ecb";
            dec_csv << "gcm";
            ecb_setup();
        }

        for (uint64_t &block_size : block_sizes) {
            double enc_tput, dec_tput;
            double total_bytes;
            for (int i = 0 ; i < nrounds+nwarm ; i++) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500)); 
                std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
                if (alg == GCM) gcm_encrypt(block_size);
                else            ecb_encrypt(block_size);
                cudaDeviceSynchronize();
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                double total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
                //std::cout << "time: " << total_time << "us" << std::endl;

                //this assumes we only do one run
                total_bytes = (double)multiplier*PAGE_SIZE*block_size; //total bytes encrypted
                total_bytes = total_bytes/(1024*1024*1024); //convert to GB
                //printf("%f GB\n", total_bytes);
                enc_tput = total_bytes/(total_time/1000000); //divide by seconds to get GB/s

                std::this_thread::sleep_for(std::chrono::milliseconds(500)); 
                begin = std::chrono::steady_clock::now();
                if (alg == GCM) gcm_decrypt(block_size);
                else            ecb_decrypt(block_size);
                cudaDeviceSynchronize();
                end = std::chrono::steady_clock::now();
                total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

                dec_tput = total_bytes/(total_time/1000000); //divide by seconds to get GB/s
            }
            std::cout << "Encryption throughput for " << (alg==0?"GCM":"ECB") << " and "<< block_size*multiplier << " pages: " << enc_tput << "Gb/s" << std::endl;
            csv << "," <<  enc_tput;
            dec_csv << "," <<  dec_tput;
        }
        if (alg == GCM) gcm_clean();
        else            ecb_clean();
    
        csv << "\n";
    }

    std::cout << "Encryption:\n" << csv.str();
    std::cout << "\n\nDecryption\n" << csv.str();
}