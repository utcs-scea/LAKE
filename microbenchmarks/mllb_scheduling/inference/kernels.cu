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


float* d_inputs;
float* d_w1;
float* d_b1;
float* d_w2;

double* d_inputs_dbl;
double* d_w1_dbl;
double* d_b1_dbl;
double* d_w2_dbl;

void gpu_setup() {
    cudaMalloc(&d_inputs, NR_FEAT*sizeof(float));
    cudaMalloc(&d_w1, NR_FEAT*10*sizeof(float));
    cudaMalloc(&d_b1, 10*sizeof(float));
    cudaMalloc(&d_w2, 10*sizeof(float));

    cudaMemcpy(d_w1, w1, NR_FEAT*10*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(d_b1, b1, 10*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy(d_w2, w2, 10*sizeof(float), cudaMemcpyHostToDevice );
}

void gpu_setup_double() {
    cudaMalloc(&d_inputs_dbl, NR_FEAT*sizeof(double));
    cudaMalloc(&d_w1_dbl, NR_FEAT*10*sizeof(double));
    cudaMalloc(&d_b1_dbl, 10*sizeof(double));
    cudaMalloc(&d_w2_dbl, 10*sizeof(double));

    cudaMemcpy(d_w1, w1_dbl, NR_FEAT*10*sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_b1, b1_dbl, 10*sizeof(double), cudaMemcpyHostToDevice );
    cudaMemcpy(d_w2, w2_dbl, 10*sizeof(double), cudaMemcpyHostToDevice );
}

void gpu_clean() {
    cudaFree(d_inputs);
    cudaFree(d_w1);
    cudaFree(d_b1);
    cudaFree(d_w2);
}

void gpu_clean_double() {
    cudaFree(d_inputs_dbl);
    cudaFree(d_w1_dbl);
    cudaFree(d_b1_dbl);
    cudaFree(d_w2_dbl);
}

void gpu_setup_input(float* inputs) {
    cudaMemcpy(d_inputs, inputs, NR_FEAT*sizeof(float), cudaMemcpyHostToDevice);
}

void gpu_setup_input_double(float* inputs) {
    double inp[NR_FEAT];
    for (int j = 0 ; j < NR_FEAT ; j++) {
        inp[j] = inputs[j];
    }

    cudaMemcpy(d_inputs, inp, NR_FEAT*sizeof(double), cudaMemcpyHostToDevice);
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

float gpu_inference_double() {
    dim3 b(1);
    dim3 t(10);

    mllb_infer_v1_dbl<<<b, t, 64*sizeof(float)>>>(d_inputs_dbl, d_w1_dbl, d_b1_dbl, NR_FEAT, 10, d_w2_dbl, *b2_dbl);
    cudaDeviceSynchronize();

    return 0;

    //float res;
    //cudaMemcpy(&res, d_inputs, sizeof(float), cudaMemcpyDeviceToHost);
    //return res;
}