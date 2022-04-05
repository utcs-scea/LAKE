#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include "consts.h"


__global__ void mllb_infer_v1(float* input, float* w1, float* b1,
    int n, int k, float* w2, float b2)
{
    //multiply 2 matrices  (1,15) x (15, 10)
    //add matrix element wise  (1x10)
    //ReLU: for each element  (x > 0 ?  x : 0)
    //multiply 2 matrices (1x10, 10x1)
    //add matrix element wise (1,1)

    __shared__ float sm[64];
    //lets assume we launch 10 threads
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    //n = 15, k = 10
    if (id < 10) {
        float acc = 0;
        //multiply 2 matrices  (1,14) x (14, 10)
        for(int i = 0; i < 15; i++) {
            //printf(" [%d]:  0,%d x %d,%d   (=%.3f x %.3f)\n", id, i, i, id, input[i], w1[i*k + id]);
            acc += input[i] * w1[i*k + id];
        }
        
        //add matrix element wise  (1x10)
        acc += b1[id];
        //ReLU: for each element  (x > 0 ?  x : 0)
        acc = acc > 0 ? acc : 0;
        sm[id] = acc;
    }

    __syncthreads();
    // if (id == 0){
    //     for(int i = 0; i < k; i++) {
    //         printf("gpu:  %d = %.3f\n", i, sm[i]);
    //     }
    // }

    if (id == 0) {
        float res = 0;
        //multiply 2 matrices (1x10, 10x1)
        for(int i = 0; i < 10; i++) {
            res += sm[i] * w2[i];
        }
        input[0] = res+b2;
    }
}


__global__ void mllb_infer_v2(float* inputs, 
    float* w1, float* b1, float* w2, float b2, float* results)
{
    // int id = blockIdx.x * blockDim.x + threadIdx.x;
    // if (id < 2) {
    //     inputs[id] = id;
    //     w1[id] = id;
    //     b1[id] = id;
    //     w2[id] = id;
    //     results[id] = id;
    // }

    //multiply 2 matrices  (1,15) x (15, 10)
    //add matrix element wise  (1x10)
    //ReLU: for each element  (x > 0 ?  x : 0)
    //multiply 2 matrices (1x10, 10x1)
    //add matrix element wise (1,1)

    //each block shouldn't have more than 8 inputs (128 threads per block)
    __shared__ float sm[10*8];
    //lets assume we launch 16*x threads (blockDim.x is a multiple of 16). we only use 15(10) so one will be idle
    int tid = threadIdx.x;
    int intrablock_idx = tid % 16;
    int inputs_per_block = blockDim.x / 16;
    int input_base_idx = blockIdx.x * inputs_per_block + intrablock_idx;
    //save input base pointer so we don't have complicated indexing
    float* input = inputs + input_base_idx;
    float* sm_base = sm + intrablock_idx;

    //n = 15, k = 10
    if (tid < 10) {
        float acc = 0;
        //multiply 2 matrices  (1,14) x (14, 10) into 1x10
        for(int i = 0; i < 15; i++) {
            //printf(" [%d]:  0,%d x %d,%d   (=%.3f x %.3f)\n", id, i, i, id, input[i], w1[i*k + id]);
            acc += input[i] * w1[i*15 + tid];
        }
        
        //add matrix element wise  (1x10)
        acc += b1[tid];
        //ReLU: for each element  (x > 0 ?  x : 0)
        acc = acc > 0 ? acc : 0;
        sm_base[tid] = acc;
    }

    __syncthreads();

    if (tid == 0) {
        float res = 0;
        //multiply 2 matrices (1x10, 10x1)
        for(int i = 0; i < 10; i++) {
            res += sm_base[i] * w2[i];
        }
        results[input_base_idx] = res+b2;
    }
}
