#include <stdio.h>
#include <string.h>
#include "weights.h"
#define BLOCK_SIZE 16



static double *d_w0, *d_b0, *d_w1, *d_b1, *d_w2, *d_b2, *d_out0, *d_out1, *d_out2;
static int w0_rows, w0_cols, b0_rows, b0_cols, w1_rows, w1_cols, b1_rows, b1_cols, w2_rows, w2_cols, b2_rows, b2_cols, input_rows;
static int out0_rows, out0_cols, out1_rows, out1_cols, out2_rows, out2_cols;

void readahead_normalized_online_data(double *readahead_online_data, int readahead_online_data_cols, int readahead_std_dev_rows, int readahead_std_dev_cols, 
    int readahead_avg_rows, int readahead_avg_cols, int readahead_variance_rows, int readahead_variance_cols,
                                      int readahead_val , double *readahead_norm_online_data) {
    double *diff, *local_average, *local_std_dev, *local_variance, *readahead_norm_online_data_last_values;
    
    cudaMalloc((void**)&diff, sizeof(double) * readahead_online_data_cols);
    cudaMalloc((void**)&local_average, sizeof(double) * readahead_avg_rows * readahead_avg_cols);
    cudaMalloc((void**)&local_std_dev, sizeof(double) * readahead_std_dev_rows * readahead_std_dev_cols);
    cudaMalloc((void**)&local_variance, sizeof(double) * readahead_variance_rows * readahead_variance_cols);
    cudaMalloc((void**)&readahead_norm_online_data, sizeof(double) * readahead_online_data_cols);
    cudaMalloc((void**)&readahead_norm_online_data_last_values, sizeof(double) * readahead_online_data_cols);
 
    int n_seconds = 10;

    matrix_mult_constant<<<1, readahead_online_data_cols>>>(local_average, n_seconds, diff);
    matrix_add<<<1, readahead_online_data_cols>>>(readahead_online_data, diff, diff);
    matrix_div_constant<<<1, readahead_online_data_cols>>>(diff, n_seconds, diff);
    set_matrix_with_matrix<<<1, readahead_online_data_cols>>>(diff, local_average);
    // print_matrix(readahead->norm_data_stat.average);

    matrix_sub<<<1, readahead_online_data_cols>>>(readahead_online_data, readahead_norm_online_data_last_values,
                diff);
    matrix_elementwise_mult<<<1, readahead_online_data_cols>>>(diff, diff, diff);
    matrix_mult_constant<<<readahead_variance_rows, readahead_variance_cols>>>(local_variance, n_seconds, local_variance);
    matrix_add<<<1, readahead_online_data_cols>>>(local_variance, diff, local_variance);
    matrix_div_constant<<<readahead_variance_rows, readahead_variance_cols>>>(local_variance, n_seconds, local_variance);


    matrix_map<<<readahead_variance_rows, readahead_variance_cols>>>(local_variance, fast_sqrt_d, local_std_dev);
    // print_matrix(readahead->norm_data_stat.std_dev);
    matrix_sub<<<1, readahead_online_data_cols>>>(readahead_online_data, local_average,
                readahead_norm_online_data);
    matrix_elementwise_div<<<1, readahead_online_data_cols>>>(readahead_norm_online_data, local_std_dev,
                            readahead_norm_online_data);

    set_matrix_with_matrix<<<1, readahead_online_data_cols>>>(readahead_online_data,
                         readahead_norm_online_data_last_values);

    cudaFree(diff);
    cudaFree(local_average);
    cudaFree(local_std_dev);
    cudaFree(local_variance);
}

int readahead_online_data_cols, readahead_std_dev_rows, readahead_std_dev_cols, readahead_avg_rows, readahead_avg_cols, readahead_variance_rows, readahead_variance_cols;

    double *get_normalized_readahead_data(double *readahead,
                                        int current_readahead_val, double *d_readahead_norm_online_data) {
    double *normalized_data = NULL;
    int readahead_online_data_cols = 5, readahead_std_dev_rows = 1, 
        readahead_std_dev_cols = 5, readahead_avg_rows = 1, readahead_avg_cols = 5, readahead_variance_rows = 1, readahead_variance_cols = 5;
    
    cudaMalloc((void**)&d_readahead_norm_online_data, sizeof(double) *readahead_online_data_cols);
    
    readahead_normalized_online_data(readahead, readahead_online_data_cols, readahead_std_dev_rows, readahead_std_dev_cols,
        readahead_avg_rows, readahead_avg_cols, readahead_variance_rows, readahead_variance_cols, current_readahead_val, d_readahead_norm_online_data);
    return normalized_data;
}

void linear_layer_forward(double *x, double *linear_w, int linear_w_rows, int x_rows, int linear_w_columns, double *bias_vector, int bias_vector_rows, int bias_vector_cols, int layer_index, double *out) {
    double *wx, *bias;
    double *wt;
    cudaMalloc((void**)&wt, sizeof(double) *x_rows * linear_w_rows);
    matrix_transpose<<<linear_w_rows, linear_w_columns>>>(linear_w, wt);
    //dimensions of wt linear_w_columns * linear_w_rows
    cudaMalloc((void**)&bias, sizeof(double) * bias_vector_rows *x_rows * bias_vector_cols);

    // wx+b
    cudaMalloc((void**)&wx, sizeof(double) * x_rows *linear_w_rows);
    //wx = matrix_mult(x, wt);
    unsigned int grid_rows = (x_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (linear_w_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    matrix_mult<<<dimGrid, dimBlock>>>( x, wt, wx, x_rows, linear_w_columns,linear_w_rows);
    matrix_repmat<<<bias_vector_rows, bias_vector_cols>>>(bias_vector, x_rows, 1, bias_vector_rows, bias_vector_cols, bias);
    //bias = matrix_repmat(linear->bias_vector, wx->rows, 1);
    matrix_add<<<x_rows, linear_w_rows>>>(wx, bias, out);

    // set input & output
    //linear->input = x;
    if (layer_index == 0) {
        out0_rows = x_rows;
        out0_cols = linear_w_rows;
    }
    if (layer_index == 1) {
        out1_rows = x_rows;
        out1_cols = linear_w_rows;
    }
    if (layer_index == 2) {
        out2_rows = x_rows;
        out2_cols = linear_w_rows;
    }

    cudaFree(wx);
    cudaFree(bias);
    cudaFree(wt);

}


double *autodiff_forward(double *input) { 
    // layer 0
    cudaMalloc((void**)&d_out0, sizeof(double) *w0_rows * input_rows);
    linear_layer_forward(input, d_w0, w0_rows, input_rows, w0_cols, d_b0, b0_rows, b0_cols, 0, d_out0);
    //layer 1
    cudaMalloc((void**)&d_out1, sizeof(double) *w1_rows * out0_rows);
    linear_layer_forward(d_out0, d_w1, w1_rows, out0_rows, w1_cols, d_b1, b1_rows, b1_cols, 1, d_out1);
    //layer 2
    cudaMalloc((void**)&d_out2, sizeof(double) *w2_rows * out1_rows);
    linear_layer_forward(d_out1, d_w2, w2_rows, out1_rows, w2_cols, d_b2, b2_rows, b2_cols, 2, d_out2);
    double *out2= (double*) malloc(sizeof(double)*out2_cols *out2_rows);
    cudaMemcpy(out2, d_out2, sizeof(int) * out2_cols *out2_rows, cudaMemcpyDeviceToHost);
    return out2;
}

double *readahead_class_net_inference(double *input) {
    return autodiff_forward(input);
}


int predict_readahead_class(double *input) {
    double *d_readahead_norm_online_data;
    int readahead_online_data_cols = 5;
    cudaMalloc((void**)&d_readahead_norm_online_data, sizeof(double) *readahead_online_data_cols);
    get_normalized_readahead_data(input, 1, d_readahead_norm_online_data);
    int cls = 0;
    double *indv_result = readahead_class_net_inference(d_readahead_norm_online_data);
    cls = matrix_argmax(indv_result, out2_rows, out2_cols);

    return cls;
}


int main() {
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

    double *w0, *w1, *w2, *b0, *b1, *b2;
    w0 = &w0_arr[0][0];
    b0 = &b0_arr[0][0];
    w1 = &w1_arr[0][0];
    b1 = &b1_arr[0][0];
    w2 = &w2_arr[0][0];
    b2 = &b2_arr[0][0];

    int input_features = 5;
    input_rows = input_features;
    cudaMalloc((void**)&d_w0, sizeof(double) *w0_rows * w0_cols);
    cudaMemcpy(d_w0, w0, sizeof(double) * w0_rows * w0_cols, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_b0, sizeof(double) *b0_rows * b0_cols);
    cudaMemcpy(d_b0, b0, sizeof(double) * b0_rows * b0_cols, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_w1, sizeof(double) *w1_rows * w1_cols);
    cudaMemcpy(d_w1, w1, sizeof(double) * w1_rows * w1_cols, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_b1, sizeof(double) *b1_rows * b1_cols);
    cudaMemcpy(d_b1, b1, sizeof(double) * b1_rows * b1_cols, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_w2, sizeof(double) *w2_rows * w2_cols);
    cudaMemcpy(d_w2, w2, sizeof(double) * w2_rows * w2_cols, cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_b2, sizeof(double) *b2_rows * b2_cols);
    cudaMemcpy(d_b2, b2, sizeof(double) * b2_rows * b2_cols, cudaMemcpyHostToDevice);

    double input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};
    double *d_input;
    cudaMalloc((void**)&d_input, sizeof(double) *input_features);
    cudaMemcpy(d_input, input, sizeof(double) * input_features, cudaMemcpyHostToDevice);
    printf("\n %d \n", predict_readahead_class(d_input));

    cleanup();

    return 0;
}