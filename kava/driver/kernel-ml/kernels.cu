#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "weights.h"

__global__ void matrix_mult_constant(float *src, float constant, float *dest) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] * constant;
}

__global__ void matrix_add(float *src, float *add, float *dest) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] + add[blockId*dim + threadId];
}

__global__ void matrix_div_constant(float *src, float constant, float *dest) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] / constant;
}

__global__ void set_matrix_with_matrix(float *src, float *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId];
}

__global__ void matrix_sub(float *src, float *sub, float *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] - sub[blockId*dim + threadId];
}

__global__ void matrix_elementwise_mult(float *m1, float *m2, float *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = m1[blockId*dim + threadId] * m2[blockId*dim + threadId];
}

__global__ void matrix_elementwise_div(float *m1, float *m2, float *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = m1[blockId*dim + threadId] / m2[blockId*dim + threadId];
}

__global__ void matrix_map(float *src, float *dest) { 
	int threadId = threadIdx.x;
    float r = 1;
    float x = src[threadId];
    long long i = *(long long *)&x;
    i = 0x5fe6eb50c7b537a9 - (i >> 1);
    r = *(float *)&i;
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    r = r * (1.5f - 0.5f * x * r * r);
    dest[threadId] =  r * x; //func_f(src[blockId*dim + threadId]);
}

__global__ void matrix_transpose(float *m, float *ret, int rows_ret, int cols_ret) { 
    int blockId = blockIdx.x; // row-id of ret
	int threadId = threadIdx.x; // col-id of ret
    ret[blockId*cols_ret + threadId] = m[threadId * rows_ret + blockId];
}

__global__ void matrix_repmat(float *m, int row_repeat, int col_repeat, int m_rows, int m_cols, float *ret) { 
    //int *ret = allocate(row_repeat*m_rows*col_repeat*m_cols);
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    if (col_repeat > 1) {
        for (int col_copy = 0; col_copy < col_repeat *m_cols; col_copy += m_cols) {
            ret[blockId*dim + (threadId +col_copy )] = m[blockId*dim + threadId];
        }
    }else {
        ret[blockId*dim + threadId] = m[blockId*dim + threadId];
    }
    if(row_repeat > 1) {
        for (int row_copy = m_rows; row_copy < m_rows*row_repeat; row_copy += m_rows) { 
            ret[(row_copy + blockId)*dim + threadId] = m[blockId*dim + threadId];
        }
    }
}

__global__ void matrix_mult(float *a, float *b, float *c, int m, int n, int k)
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

__global__ void get_average(float *avg_init, int k1, int k2, int batch_size, float *inp, float * avg_final) {
	int threadId = threadIdx.x;
    float sum = 0;
    for(int j = 0; j < batch_size; j++) {
        sum += inp[j*5 + threadId];
    }
    avg_final[threadId] = avg_init[threadId] * k1 /(k2  + batch_size)
        + sum / (k2 + batch_size);
}

__global__ void get_variance(float *var_init, float k1, float k2, 
        int batch_size ,float *inp, float *data_last_values, float * var_final) {
	int threadId = threadIdx.x;
    float sum_diff = 0;
    for(int j = 0; j < batch_size; j++) {
        sum_diff += (data_last_values[threadId] -  inp[ j*5 + threadId])
        *(data_last_values[threadId] -  inp[j*5 + threadId]) ;
    }
    var_final[threadId] = var_init[threadId] * k1 /(k2 + batch_size) 
        + sum_diff/ (k2 + batch_size);
}

__global__ void normalize_data(float *inp, float *avg, float *std_dev, float *out) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
    int dim = blockDim.x;
    out[blockId * dim + threadId] = (inp[blockId * dim + threadId] - avg[threadId])
        / std_dev[threadId];
}

__global__ void add_bias(float *wx, float *bias, float *out) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
    int dim = blockDim.x;
    out[blockId * dim + threadId] = wx[blockId * dim + threadId] + bias[threadId];
}

__global__ void matrix_argmax(float *src, int cols, int *max_col_array) {
    int threadId = threadIdx.x; // row_index
    int max_col = 0;
    int max = INT_MIN;
    for(int i = 0; i < cols; i++) {
        if(max < src[threadId * cols + i]) {
            max = src[threadId * cols + i];
            max_col = i;
        }
    }
    max_col_array[threadId] = max_col;

}
