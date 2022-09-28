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


//fused

//lets do 160 per block, that's 32*5, so 32 inputs per block
//input before was each line was 5 inputs, with batch rows
//need at least 32 threads, we must ensure no OOB
__global__ void normalize_fused(int batch_size, float* inputs, float* avg_base, float* avg_out, 
        float* last_values, float* var_out, float* final_out) {
    int tid = threadIdx.x;
    int uid = blockIdx.x * blockDim.x + threadIdx.x;
    int intrablock_idx = tid % 5;
    int inputs_per_block = 32;
    int base_idx = blockIdx.x * inputs_per_block + intrablock_idx;

    float* input  = inputs    + base_idx;
    float* output = final_out + base_idx;

    const int MAX = 5*batch_size;
    const int k1 = 10; //n_seconds
    const int k2 = 9; // n_1_seconds

    //calculate averages
    if (uid < 5) {
        float sum = 0;
        //do average of each point in all inputs
        for(int j = 0; j < batch_size; j++) {
            sum += inputs[j*5 + tid];
        }
        avg_out[uid] = avg_base[uid] * k1 /(k2  + batch_size) + sum / (k2 + batch_size);
    }

    //calculate variance, stddev through the magic thingy
    //use second warp
    if (uid > 15 && uid <= 20) {
        float* var_init = avg_base;
        int idx = uid - 16;

        float sum_diff = 0;
        for(int j = 0; j < batch_size; j++) {
            sum_diff += (last_values[idx] -  inputs[j*5 + idx]) * (last_values[idx] -  inputs[j*5 + idx]) ;
        }
        var_out[idx] = var_init[idx] * k1 /(k2 + batch_size) + sum_diff/ (k2 + batch_size);

        float r = 1;
        float x = var_out[idx];
        long long i = *(long long *)&x;
        i = 0x5fe6eb50c7b537a9 - (i >> 1);
        r = *(float *)&i;
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        var_out[idx] =  r * x;
    }

    __syncthreads();

    if (uid < MAX) {
        *output = (*input - avg_out[intrablock_idx]) / var_out[intrablock_idx];
    }
}

//fused linear
__global__ void fused_forward(float *input, int* result, int batch_size, 
        float* d_w0, float* d_b0, float* wt0,
        float* d_w1, float* d_b1, float* wt1,
        float* d_w2, float* d_b2, float* wt2,
        float* d_out0, float* d_out1, float* d_out2) {
    // const int w0_rows = 15;
    // const int w0_cols = 5;
    // const int b0_rows = 15;
    // const int b0_cols = 1;
    // const int w1_rows = 5;
    // const int w1_cols = 15;
    // const int b1_rows = 5;
    // const int b1_cols = 1; 
    // const int w2_rows = 4;
    // const int w2_cols = 5;
    // const int b2_rows = 4;
    // const int b2_cols = 1;

    int tid = threadIdx.x;

    //TODO: something is probably wrong, I think matrices are all transposed
    //matrix mult input (batch x 5) and wt0 (5 x 15) (apparently its 15 x 5)
    // out is (batch x 15)
    {
        float* my_row = input + blockIdx.x * 5;
        float* my_out = d_out0 + blockIdx.x * 15;

        if (tid < 15) {
            float acc = 0;
            for(int i = 0; i < 5; i++) {
                acc += my_row[i] * wt0[i*5 + tid];
            }
            my_out[tid] = acc + d_b0[tid];
        }
    }

    __syncthreads();

    //matrix mult d_out0 (batch x 15) and wt1 (15 x 5)
    // out is (batch x 5)
    {
        float* my_row = d_out0 + blockIdx.x * 15;
        float* my_out = d_out1 + blockIdx.x * 5;

        if (tid < 5) {
            float acc = 0;
            for(int i = 0; i < 15; i++) {
                acc += my_row[i] * wt1[i*15 + tid];
            }
            my_out[tid] = acc + d_b1[tid];
        }
    }

    __syncthreads();

    //matrix mult d_out1 (batch x 5) and wt2 (5 x 4)
    // out is (batch x 4)
    {
        float* my_row = d_out1 + blockIdx.x * 5;
        float* my_out = d_out2 + blockIdx.x * 4;

        if (tid < 4) {
            float acc = 0;
            for(int i = 0; i < 15; i++) {
                acc += my_row[i] * wt1[i*15 + tid];
            }
            my_out[tid] = acc + d_b2[tid];
        }
    }

    __syncthreads();

    //final output is sized: batch size
    {
        float* my_row = d_out2 + blockIdx.x * 4;
        if (tid == 0) {
            int idx = 0;
            for(int i = 1; i < 3; i++) {
                if (my_row[i] > my_row[idx])
                    idx = i;    
            }
            result[blockIdx.x] = idx;
        }
    }
}
