#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include "weights.h"

__global__ void matrix_mult_constant(double *src, double constant, double *dest) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] * constant;
}

__global__ void matrix_add(double *src, double *add, double *dest) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] + add[blockId*dim + threadId];
}

__global__ void matrix_div_constant(double *src, double constant, double *dest) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] / constant;
}

__global__ void set_matrix_with_matrix(double *src, double *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId];
}

__global__ void matrix_sub(double *src, double *sub, double *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] - sub[blockId*dim + threadId];
}

__global__ void matrix_elementwise_mult(double *m1, double *m2, double *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = m1[blockId*dim + threadId] * m2[blockId*dim + threadId];
}

__global__ void matrix_elementwise_div(double *m1, double *m2, double *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = m1[blockId*dim + threadId] / m2[blockId*dim + threadId];
}

__global__ void matrix_map(double *src, double (*func_f)(double), double *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = func_f(src[blockId*dim + threadId]);
}

__global__ void matrix_transpose(double *m, double *ret) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    ret[blockId*dim + threadId] = m[threadId * dim + blockId];
}

__global__ void matrix_repmat(double *m, int row_repeat, int col_repeat, int m_rows, int m_cols, double *ret) { 
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

__global__ void matrix_mult(double *a,double *b, double *c, int m, int n, int k)
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
