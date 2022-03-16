#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include "consts.h"

// https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu


__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}


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


__global__ void mllb_infer_v1_dbl(double* input, double* w1, double* b1,
    int n, int k, double* w2, double b2)
{
    __shared__ double sm[64];
    //lets assume we launch 10 threads
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    //n = 15, k = 10
    if (id < 10) {
        double acc = 0;
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
        double res = 0;
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


float* d_inputs;
float* d_w1;
float* d_b1;
float* d_w2;
float* d_results;

void gpu_setup(int n_inputs) {
    cudaMalloc(&d_inputs, NR_FEAT*sizeof(float));
    cudaMalloc(&d_w1, NR_FEAT*10*sizeof(float));
    cudaMalloc(&d_b1, 10*sizeof(float));
    cudaMalloc(&d_w2, 10*sizeof(float));
    cudaMalloc(&d_results, n_inputs*sizeof(float));
    
    cudaMemcpy(d_w1, w1, NR_FEAT*10*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(d_b1, b1, 10*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(d_w2, w2, 10*sizeof(float), cudaMemcpyHostToDevice );
}

void gpu_clean() {
    cudaFree(d_inputs);
    cudaFree(d_results);
    cudaFree(d_w1);
    cudaFree(d_b1);
    cudaFree(d_w2);
}

void gpu_setup_inputs(float* inputs, int n) {
    cudaMemcpy(d_inputs, inputs, n*NR_FEAT*sizeof(float), cudaMemcpyHostToDevice);
}

float gpu_inference() {
    dim3 b(1);
    dim3 t(10);

    mllb_infer_v1<<<b, t, 64*sizeof(float)>>>(d_inputs, d_w1, d_b1, NR_FEAT, 10, d_w2, *b2);
    cudaDeviceSynchronize();
    return 0;
    //float res;
    //cudaMemcpy(&res, d_inputs, sizeof(float), cudaMemcpyDeviceToHost);
    //return res;
}

float gpu_inference_many(int n_inputs) {
    int total_threads = n_inputs * 16;
    int blocks = total_threads / 128;
    dim3 b(blocks);
    dim3 t(128);

    mllb_infer_v2<<<b, t, 10*8*sizeof(float)>>>(d_inputs, d_w1, d_b1, d_w2, *b2, d_results);
    cudaDeviceSynchronize();

    return 0;
}

float gpu_get_result(int n_inputs) {
    float res[n_inputs];
    cudaMemcpy(&res, d_results, n_inputs*sizeof(float), cudaMemcpyDeviceToHost);
    return 0;
}
