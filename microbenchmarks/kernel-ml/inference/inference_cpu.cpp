#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <limits.h> 
#include "weights.h"

#define BLOCK_SIZE 16

double fast_sqrt_d(double x) {
  double r;
  int64_t i = *(int64_t *)&x;
  i = 0x5fe6eb50c7b537a9 - (i >> 1);
  r = *(double *)&i;
  r = r * (1.5f - 0.5f * x * r * r);
  r = r * (1.5f - 0.5f * x * r * r);
  r = r * (1.5f - 0.5f * x * r * r);
  r = r * (1.5f - 0.5f * x * r * r);
  r = r * (1.5f - 0.5f * x * r * r);
  return r * x;
}

double* allocate(int size) {
    double* ptr = (double*) malloc(size * sizeof(double));
    return ptr;
}

void matrix_mult_constant(double *src, double constant, double *dest, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i*cols + j] = src[i*cols + j] * constant;
        }
    }
}

void matrix_add(double *src, double *add, double *dest, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i*cols + j] = src[i*cols + j] + add[i*cols + j];
        }
    }
}

void matrix_div_constant(double *src, double constant, double *dest, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i*cols + j] = src[i*cols + j] / constant;
        }
    }
}

void set_matrix_with_matrix(double *src, double *dest, int rows, int cols) { 
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i*cols + j] = src[i*cols + j];
        }
    }
}

void matrix_sub(double *src, double *sub, double *dest, int rows, int cols) { 
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i*cols + j] = src[i*cols + j] - sub[i*cols + j];
        }
    }
}

void matrix_elementwise_mult(double *m1, double *m2, double *dest, int rows, int cols) { 
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i*cols + j] = m1[i*cols + j] * m2[i*cols + j];
        }
    }
}

void matrix_elementwise_div(double *m1, double *m2, double *dest, int rows, int cols) { 
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i*cols + j] = m1[i*cols + j] / m2[i*cols + j];
        }
    }
}

void matrix_map(double *src, double (*func_f)(double), double *dest, int rows, int cols) { 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dest[i * cols + j] = func_f(src[i * cols + j]);
        }
    }
}

void matrix_transpose(double *m, double *ret, int rows, int cols) { 
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            ret[i*cols + j] = m[j*rows + i];
        }
    }
}

void matrix_repmat(double *m, int row_repeat, int col_repeat, int m_rows, int m_cols, double *ret) { 
    //int *ret = allocate(row_repeat*m_rows*col_repeat*m_cols);
    if (col_repeat > 1) {
        for(int i = 0; i < m_rows; i++) {
            for (int col_copy = 0; col_copy < col_repeat *m_cols; col_copy += m_cols) { 
                for(int col_idx = 0; col_idx < m_cols; col_idx++) {
                    ret[i*(m_cols * col_repeat) + col_copy + col_idx] = m[i*m_cols + col_idx];
                }
            }
        }
    }else {
        for(int i = 0; i < m_rows; i++) {
            for (int j = 0; j < m_cols; j++)
                ret[i*m_cols + j] = m[i*m_cols + j];
        }
    }
    if(row_repeat > 1) {
        for (int row_copy = m_rows; row_copy < m_rows*row_repeat; row_copy += m_rows) { 
            for(int i = 0; i < m_rows; i++) {
                for(int j = 0; j < m_cols*col_repeat; j++){
                    ret[(i + row_copy)*(m_cols * col_repeat) + j] = m[i *m_cols + j];
                }
            }
        }
    }
}

void matrix_mult(double *a,double *b, double *c, int m, int n, int k)
{ 
    int i, j, k_;
    for(i = 0; i < m; ++i)
        for(j = 0; j < k; ++j)
            for(k_ = 0; k_ < n; ++k_)
            {
                c[i*k + j] += a[i *n + k_] * b[k_*k+ j];
            }

}

int matrix_argmax(double *src, int rows, int cols) { 
    int max = INT_MIN;
    int max_row = 0, max_col = 0;
    for(int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if(max < src[i*rows + j]) {
                max = src[i*rows + j];
                max_row =i;
                max_col = j;
            }
        }
    }
    return max_col;
}

static int w0_rows, w0_cols, b0_rows, b0_cols, w1_rows, w1_cols, b1_rows, b1_cols, w2_rows, w2_cols, b2_rows, b2_cols, input_rows;
static int out0_rows, out0_cols, out1_rows, out1_cols, out2_rows, out2_cols;
static double *out0, *out1, *out2, *w0, *w1, *w2, *b0, *b1, *b2;

void readahead_normalized_online_data(double *readahead_online_data, int readahead_online_data_cols, int readahead_std_dev_rows, int readahead_std_dev_cols, 
    int readahead_avg_rows, int readahead_avg_cols, int readahead_variance_rows, int readahead_variance_cols,
                                      int readahead_val , double *readahead_norm_online_data) {
    double *diff, *local_average, *local_std_dev, *local_variance, *readahead_norm_online_data_last_values;

    diff = allocate(readahead_online_data_cols);
    local_average = allocate(readahead_avg_rows * readahead_avg_rows);
    local_std_dev = allocate(readahead_std_dev_rows * readahead_std_dev_cols);
    local_variance = allocate(readahead_variance_rows * readahead_variance_cols);
    readahead_norm_online_data = allocate(readahead_online_data_cols);
    readahead_norm_online_data_last_values = allocate(readahead_online_data_cols);
    int n_seconds = 10;

    matrix_mult_constant(local_average, n_seconds, diff, 1, readahead_online_data_cols);
    matrix_add(readahead_online_data, diff, diff, 1, readahead_online_data_cols);
    matrix_div_constant(diff, n_seconds, diff, 1, readahead_online_data_cols);
    set_matrix_with_matrix(diff, local_average, 1, readahead_online_data_cols);

    matrix_sub(readahead_online_data, readahead_norm_online_data_last_values,
        diff, 1, readahead_online_data_cols);
    matrix_elementwise_mult(diff, diff, diff, 1, readahead_online_data_cols);
    matrix_mult_constant(local_variance, n_seconds, local_variance,
        readahead_variance_rows, readahead_variance_cols);
    matrix_add(local_variance, diff, local_variance, 1, readahead_online_data_cols);
    matrix_div_constant(local_variance, n_seconds, local_variance,
        readahead_variance_rows, readahead_variance_cols);
    matrix_map(local_variance, fast_sqrt_d, local_std_dev,
        readahead_variance_rows, readahead_variance_cols);

    matrix_sub(readahead_online_data, local_average,
        readahead_norm_online_data, 1, readahead_online_data_cols);
    matrix_elementwise_div(readahead_norm_online_data, local_std_dev,
        readahead_norm_online_data, 1, readahead_online_data_cols);

    set_matrix_with_matrix(readahead_online_data,
        readahead_norm_online_data_last_values, 1, readahead_online_data_cols);

    free(diff);
    free(local_average);
    //free(local_std_dev);
    free(local_variance);
}


int readahead_online_data_cols, readahead_std_dev_rows, readahead_std_dev_cols, readahead_avg_rows, 
readahead_avg_cols, readahead_variance_rows, readahead_variance_cols;

    double* get_normalized_readahead_data(double *readahead,
                                      int current_readahead_val, double *d_readahead_norm_online_data) {
    double *normalized_data = NULL;
    int readahead_online_data_cols = 5, readahead_std_dev_rows = 1, 
      readahead_std_dev_cols = 5, readahead_avg_rows = 1, 
      readahead_avg_cols = 5, readahead_variance_rows = 1, 
      readahead_variance_cols = 5;
    d_readahead_norm_online_data = allocate(readahead_online_data_cols);
    readahead_normalized_online_data(readahead, readahead_online_data_cols, readahead_std_dev_rows, readahead_std_dev_cols,
        readahead_avg_rows, readahead_avg_cols, readahead_variance_rows, readahead_variance_cols, current_readahead_val, d_readahead_norm_online_data);
    return normalized_data;
}

void linear_layer_forward(double *x, double *linear_w, int linear_w_rows, int x_rows, int linear_w_columns, double *bias_vector, int bias_vector_rows, int bias_vector_cols, int layer_index, double *out) {
    double *wx, *bias;
    double *wt;
    wt = allocate(x_rows * linear_w_rows);
    matrix_transpose(linear_w, wt, linear_w_rows, linear_w_columns);
    //dimensions of wt linear_w_columns * linear_w_rows
    bias = allocate(bias_vector_rows *x_rows * bias_vector_cols);

    // wx+b
    wx = allocate(x_rows *linear_w_rows);
    //wx = matrix_mult(x, wt);
    matrix_mult( x, wt, wx, x_rows, linear_w_columns,linear_w_rows);
    matrix_repmat(bias_vector, x_rows, 1, bias_vector_rows, bias_vector_cols, bias);
    //bias = matrix_repmat(linear->bias_vector, wx->rows, 1);
    matrix_add(wx, bias, out, x_rows, linear_w_rows);

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

    free(wx);
    free(bias);
    free(wt);

}

double *autodiff_forward(double *input) { 
    // layer 0
    out0 = allocate(w0_rows * input_rows);
    linear_layer_forward(input, w0, w0_rows, input_rows, w0_cols, b0, b0_rows, b0_cols, 0, out0);
    //layer 1
    out1 = allocate(w1_rows * out0_rows);
    linear_layer_forward(out0, w1, w1_rows, out0_rows, w1_cols, b1, b1_rows, b1_cols, 1, out1);
    //layer 2
    out2 = allocate(w2_rows * out1_rows);
    linear_layer_forward(out1, w2, w2_rows, out1_rows, w2_cols, b2, b2_rows, b2_cols, 2, out2);
    return out2;
}

double *readahead_class_net_inference(double *input) {
    return autodiff_forward(input);
}

int predict_readahead_class(double *input) {
    int readahead_online_data_cols = 5;
    double *d_readahead_norm_online_data = allocate(readahead_online_data_cols);
    get_normalized_readahead_data(input, 1, d_readahead_norm_online_data);
    int cls = 0;
    double *indv_result = readahead_class_net_inference(d_readahead_norm_online_data);
    cls = matrix_argmax(indv_result, out2_rows, out2_cols);

  return cls;
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
    w0 = &w0_arr[0][0];
    b0 = &b0_arr[0][0];
    w1 = &w1_arr[0][0];
    b1 = &b1_arr[0][0];
    w2 = &w2_arr[0][0];
    b2 = &b2_arr[0][0];
    
    int input_features = 5;
    input_rows = input_features;

    double input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};
    printf("\n %d \n", predict_readahead_class(input));

    //cleanup();

    return 0;
}