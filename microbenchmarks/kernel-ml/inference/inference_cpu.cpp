#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <limits.h> 
#include "weights.h"
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <cuda_runtime.h>

static int w0_rows, w0_cols, b0_rows, b0_cols, w1_rows, w1_cols, b1_rows, b1_cols, w2_rows, w2_cols, b2_rows, b2_cols, input_rows;
static int out0_rows, out0_cols, out1_rows, out1_cols, out2_rows, out2_cols;
static float *out0, *out1, *out2, *w0, *w1, *w2, *b0, *b1, *b2, *stats;
static float* batch_input;
static int *result_cols;

float* allocate(int size) {
    float* ptr = (float*) malloc(size * sizeof(float));
    return ptr;
}

void matrix_map(float *src, float *dest, int cols) {  
    for (int j = 0; j < cols; j++) {
        float r = 1;
        float x = src[j];
        long long i = *(long long *)&x;
        i = 0x5fe6eb50c7b537a9 - (i >> 1);
        r = *(float *)&i;
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        r = r * (1.5f - 0.5f * x * r * r);
        dest[j] =  r * x;
    }
}

void matrix_argmax(float *src, int cols, int rows, int *max_col_array) {
    for(int j = 0; j < rows; j ++) {
        int max_col = 0;
        int max = INT_MIN;
        for(int i = 0; i < cols; i++) {
            if(max < src[j * cols + i]) {
                max = src[j * cols + i];
                max_col = i;
            }
        }
        max_col_array[j] = max_col;
    }
}

void matrix_transpose(float *m, float *ret, int rows, int cols) { 
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ret[i*cols + j] = m[j*rows + i];
        }
    }
}

void matrix_mult(float *a,float *b, float *c, int m, int n, int k)
{ 
    int i, j, k_;
    for(i = 0; i < m; ++i)
        for(j = 0; j < k; ++j)
            for(k_ = 0; k_ < n; ++k_)
            {
                c[i*k + j] += a[i *n + k_] * b[k_*k+ j];
            }

}


void add_bias(float *wx, float *bias, float *out, int rows, int cols) {
    int i, j;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            out[i*cols + j] = wx[i *cols + j] + bias[j];
        }
    }
}

void normalize_data(float *inp, float *avg, float *std_dev, float *out, int rows, int cols) {
    int i, j;
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            out[i*cols + j] = (inp[i * cols + j] - avg[j]) / std_dev[j];
        }
    }
}

void get_variance(float *var_init, float k1, float k2, 
        int batch_size ,float *inp, float *data_last_values, float * var_final, int cols) {
	for(int i = 0; i < cols; i++) {
        float sum_diff = 0;
        for(int j = 0; j < batch_size; j++) {
            sum_diff += (data_last_values[i] -  inp[ j*5 + i])
            *(data_last_values[i] -  inp[j*5 + i]) ;
    }
    var_final[i] = var_init[i] * k1 /(k2 + batch_size) 
        + sum_diff/ (k2 + batch_size);
    }
}

void get_average(float *avg_init, int k1, int k2, int batch_size, float *inp, float * avg_final, int cols) {
    for(int i = 0; i < cols; i++) { 
        float sum = 0;
        for(int j = 0; j < batch_size; j++) {
            sum += inp[j*5 + i];
        }
        avg_final[i] = avg_init[i] * k1 /(k2  + batch_size)
        + sum / (k2 + batch_size);
    }
}

void readahead_normalized_online_data(float *readahead_online_data, int readahead_online_data_cols,
 float *readahead_norm_online_data, int batch_size) {
    float *diff, *local_average, *local_std_dev, *local_variance, *readahead_norm_online_data_last_values;

    local_average = allocate(readahead_online_data_cols);
    local_std_dev = allocate(readahead_online_data_cols);
    local_variance = allocate(readahead_online_data_cols);
    readahead_norm_online_data_last_values = allocate(readahead_online_data_cols * batch_size);
    int n_seconds = 10;
    int n_1_seconds = 9;

    get_average(stats, n_seconds, n_1_seconds, 
        batch_size, readahead_online_data, local_average, readahead_online_data_cols);

    get_variance(stats, n_seconds, n_1_seconds, batch_size, readahead_online_data, 
        readahead_norm_online_data_last_values, local_variance, readahead_online_data_cols);
    
    matrix_map(local_variance, local_std_dev, readahead_online_data_cols);
    normalize_data(readahead_online_data, local_average, 
        local_std_dev, readahead_norm_online_data, batch_size, readahead_online_data_cols);
    
    free(local_average);
    free(local_std_dev);
    free(local_variance);
}


int readahead_online_data_cols, readahead_std_dev_rows, readahead_std_dev_cols, readahead_avg_rows, 
readahead_avg_cols, readahead_variance_rows, readahead_variance_cols;

void get_normalized_readahead_data(float *readahead,
                                       float *d_readahead_norm_online_data, int batch_size) {
    int readahead_online_data_cols = 5;
    readahead_normalized_online_data(readahead, readahead_online_data_cols, 
    d_readahead_norm_online_data, batch_size);
}

void linear_layer_forward(float *x, float *linear_w, int linear_w_rows, 
int linear_w_columns, float *bias_vector, int layer_index, float *out, int batch_size) {
    float *wx;
    float *wt;
    wt = allocate(linear_w_columns * linear_w_rows);
    matrix_transpose(linear_w, wt, linear_w_columns, linear_w_rows);
    //dimensions of wt linear_w_columns * linear_w_rows

    // wx+b
    wx = allocate(batch_size *linear_w_rows);
    //wx = matrix_mult(x, wt);
    matrix_mult(x, wt, wx, batch_size, linear_w_columns, linear_w_rows);

    add_bias(wx, bias_vector, out, batch_size, linear_w_rows);
    // set input & output
    //linear->input = x;
    if (layer_index == 0) {
        out0_rows = batch_size;
        out0_cols = linear_w_rows;
    }
    if (layer_index == 1) {
        out1_rows = batch_size;
        out1_cols = linear_w_rows;
    }
    if (layer_index == 2) {
        out2_rows = batch_size;
        out2_cols = linear_w_rows;
    }

    free(wx);
    free(wt);

}

void autodiff_forward(float *input, int batch_size) { 
    // layer 0
    out0 = allocate(w0_rows * batch_size);
    linear_layer_forward(input, w0, w0_rows, w0_cols, b0, 0, out0, batch_size);
    //layer 1
    out1 = allocate(w1_rows * out0_rows);
    linear_layer_forward(out0, w1, w1_rows, w1_cols, b1, 1, out1, batch_size);
    //layer 2
    out2 = allocate(w2_rows * out1_rows);
    linear_layer_forward(out1, w2, w2_rows, w2_cols, b2, 2, out2, batch_size);
    matrix_argmax(out2, w2_rows,out2_rows, result_cols);
}

void readahead_class_net_inference(float *input, int batch_size) {
    autodiff_forward(input, batch_size);
}

void predict_readahead_class(float *input, int batch_size) {
    int readahead_online_data_cols = 5;
    float *d_readahead_norm_online_data = allocate(readahead_online_data_cols * batch_size);
    get_normalized_readahead_data(input, d_readahead_norm_online_data, batch_size);
    readahead_class_net_inference(d_readahead_norm_online_data, batch_size);
}
void cleanup() {
    free(w0);
    free(w1);
    free(w2);
    free(b0);
    free(b1);
    free(b2);
    free(out0);
    free(out1);
    free(out2);
}


void setup_cpu() {
    //dimensions of weights of layer 0 : 15 *5
    //dimensions of bias of layer 0 : 15 * 1
    //dimensions of weights of layer 1 : 15 * 5
    //dimensions of bias of layer 1 : 5 * 1
    //dimensions of weights of layer 2 : 4 * 5
    //dimensions of bias of layer 2 : 4 * 1
    w0_rows = 15;
    w0_cols = 5;
    b0_rows = 15;
    b0_cols = 1;
    w1_rows = 5;
    w1_cols = 15;
    b1_rows = 5;
    b1_cols = 1; 
    w2_rows = 4;
    w2_cols = 5;
    b2_rows = 4;
    b2_cols = 1;
    w0 = &w0_arr[0][0];
    b0 = &b0_arr[0][0];
    w1 = &w1_arr[0][0];
    b1 = &b1_arr[0][0];
    w2 = &w2_arr[0][0];
    b2 = &b2_arr[0][0];
    int batch_size = 1024;
    int input_features = 5;
    input_rows = input_features;

    // float input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};
    
    // batch_input = allocate(batch_size * 5);
    // int i ,j;
    // for(i = 0; i < batch_size; i++) {
    //     for(j = 0 ; j < 5; j++) {
    //         batch_input[i*5 + j] = input[j];
    //     }
    // }
    stats = &intial_stats[0];
    //predict_readahead_class(batch_input, 1);
    //cleanup();
}

void setup_input(int batch_size) {
    float input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};
    batch_input = allocate(batch_size * 5);
    int i ,j;
    for(i = 0; i < batch_size; i++) {
        for(j = 0 ; j < 5; j++) {
            batch_input[ i*5 + j] = input[j];
        }
    }
    result_cols = (int*) malloc(batch_size * sizeof(int));
}

int main(int argc, char** argv) {
    setup_cpu();
    int batch_size = 5;
    

    std::stringstream csv;
    csv << ", inference, inference+transfer\n";


    int cpu_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int RUNS = 10;

    for (int &N_INPUTS_BATCH : cpu_sizes) {
        uint32_t cpubatch_total(0);
        for (int i = 0 ; i < RUNS ; i++) {
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            setup_input(N_INPUTS_BATCH);
            predict_readahead_class(batch_input, N_INPUTS_BATCH);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
            cpubatch_total += total_time;
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); 
        }
        
        std::cout << "CPU time for " << N_INPUTS_BATCH << " inferences: " << cpubatch_total/RUNS << "us. " << std::endl;
        csv << "cpu" <<N_INPUTS_BATCH<<", " << cpubatch_total/RUNS << "," << cpubatch_total/RUNS << std::endl;
    }
    std::cout << "CSV:\n" << csv.str();
    return 0;
}

