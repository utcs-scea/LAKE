#include <stdio.h>
#include <string.h>
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

int* allocate(int size) {
    int *ret;
    cudaMalloc((void**)&ret, sizeof(int) *size);
    return ret;
}


__global__ void matrix_mult_constant(int *src, int constant, int *dest) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] * constant;
}

__global__ void matrix_add(int *src, int *add, int *dest) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] + add[blockId*dim + threadId];
}

__global__ void matrix_div_constant(int *src, int constant, int *dest) {
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] / constant;
}

__global__ void set_matrix_with_matrix(int *src, int *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId];
}

__global__ void matrix_sub(int *src, int *sub, int *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = src[blockId*dim + threadId] - sub[blockId*dim + threadId];
}

__global__ void matrix_elementwise_mult(int *m1, int *m2, int *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = m1[blockId*dim + threadId] * m2[blockId*dim + threadId];
}

__global__ void matrix_elementwise_div(int *m1, int *m2, int *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = m1[blockId*dim + threadId] / m2[blockId*dim + threadId];
}

__global__ void matrix_map(int *src, double (*func_f)(double), int *dest) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    dest[blockId*dim + threadId] = func_f(src[blockId*dim + threadId]);
}

// __global__ void readahead_normalized_online_data() {
//     static int *matrix;
//     int rows = 10;
//     int cols = 10;
//     //cudaMalloc((void**)&matrix, sizeof(int) * rows*cols);
// }

__global__ void matrix_transpose(int *m, int *ret) { 
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    ret[blockId*dim + threadId] = m[threadId * dim + blockId];
}

__global__ void matrix_repmat(int *m, int row_repeat, int col_repeat, int m_rows, int m_cols, int *ret) { 
    //int *ret = allocate(row_repeat*m_rows*col_repeat*m_cols);
    int blockId = blockIdx.x;
	int threadId = threadIdx.x;
	int dim = blockDim.x;
    if (col_repeat > 1) {
        for (int col_copy = 0; col_copy < col_copy *m_cols; col_copy += m_cols) {
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

__global__ void matrix_mult(int *a,int *b, int *c, int m, int n, int k)
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

int matrix_argmax(int *src, int rows, int cols) { 
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

static int *d_w0, *d_b0, *d_w1, *d_b1, *d_w2, *d_b2, *d_out0, *d_out1, *d_out2;
static int w0_rows, w0_cols, b0_rows, b0_cols, w1_rows, w1_cols, b1_rows, b1_cols, w2_rows, w2_cols, b2_rows, b2_cols, input_rows;
static int out0_rows, out0_cols, out1_rows, out1_cols, out2_rows, out2_cols;

void readahead_normalized_online_data(int *readahead_online_data, int readahead_online_data_cols, int readahead_std_dev_rows, int readahead_std_dev_cols, 
    int readahead_avg_rows, int readahead_avg_cols, int readahead_variance_rows, int readahead_variance_cols,
                                      int readahead_val , int *readahead_norm_online_data) {
//   val n_seconds, n_1_seconds;
//   matrix *diff = allocate_matrix(1, readahead->online_data->cols,
//                                  readahead->online_data->type);
//   matrix *local_average =
//       allocate_matrix(readahead->norm_data_stat.average->rows,
//                       readahead->norm_data_stat.average->cols,
//                       readahead->norm_data_stat.average->type);
//   matrix *local_std_dev =
//       allocate_matrix(readahead->norm_data_stat.std_dev->rows,
//                       readahead->norm_data_stat.std_dev->cols,
//                       readahead->norm_data_stat.std_dev->type);
//   matrix *local_variance =
//       allocate_matrix(readahead->norm_data_stat.variance->rows,
//                       readahead->norm_data_stat.variance->cols,
//                       readahead->norm_data_stat.variance->type);
    int *diff, *local_average, *local_std_dev, *local_variance, *readahead_norm_online_data_last_values;
    
  cudaMalloc((void**)&diff, sizeof(int) * readahead_online_data_cols);
  cudaMalloc((void**)&local_average, sizeof(int) * readahead_avg_rows * readahead_avg_rows);
  cudaMalloc((void**)&local_std_dev, sizeof(int) * readahead_std_dev_rows * readahead_std_dev_cols);
  cudaMalloc((void**)&local_variance, sizeof(int) * readahead_variance_rows * readahead_variance_cols);
  cudaMalloc((void**)&readahead_norm_online_data, sizeof(int) * readahead_online_data_cols);
  cudaMalloc((void**)&readahead_norm_online_data_last_values, sizeof(int) * readahead_online_data_cols);
  // setting values
//   set_matrix_with_matrix(readahead->norm_data_stat.average, local_average);
//   set_matrix_with_matrix(readahead->norm_data_stat.std_dev, local_std_dev);
//   set_matrix_with_matrix(readahead->norm_data_stat.variance, local_variance);
  int n_seconds = 10;

//   readahead->online_data->vals.d[mat_index(readahead->online_data, 0,
//                                            readahead->online_data->cols - 1)] =
//       ((double)readahead_val) / 1024;
//   readahead->norm_data_stat.n_seconds++;
//   n_seconds.d = (double)readahead->norm_data_stat.n_seconds;
//   n_1_seconds.d = (double)(readahead->norm_data_stat.n_seconds - 1);

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


//   free_matrix(diff);
//   free_matrix(local_average);
//   free_matrix(local_std_dev);
//   free_matrix(local_variance);
    cudaFree(diff);
    cudaFree(local_average);
    cudaFree(local_std_dev);
    cudaFree(local_variance);
}
int readahead_online_data_cols, readahead_std_dev_rows, readahead_std_dev_cols, readahead_avg_rows, readahead_avg_cols, readahead_variance_rows, readahead_variance_cols;

int *get_normalized_readahead_data(int *readahead,
                                      int current_readahead_val, int *d_readahead_norm_online_data) {
  int *normalized_data = NULL;
//   readahead_normalized_online_data((readahead_net *)readahead,
//                                    current_readahead_val, false);
//   normalized_data = matrix_float_conversion(readahead->norm_online_data);

int readahead_online_data_cols, readahead_std_dev_rows, 
    readahead_std_dev_cols, readahead_avg_rows, readahead_avg_cols, readahead_variance_rows, readahead_variance_cols;
cudaMalloc((void**)&d_readahead_norm_online_data, sizeof(int) *readahead_online_data_cols);

    readahead_normalized_online_data(readahead, readahead_online_data_cols, readahead_std_dev_rows, readahead_std_dev_cols,
    readahead_avg_rows, readahead_avg_cols, readahead_variance_rows, readahead_variance_cols, current_readahead_val, d_readahead_norm_online_data);
  //normalized_data = matrix_float_conversion(readahead->norm_online_data);
    // int *normalized_data;
    // cudaMemcpy(normalized_data, d_readahead_norm_online_data, sizeof(int) * readahead_online_data_cols, cudaMemcpyDeviceToHost);

  return normalized_data;
}

void linear_layer_forward(int *x, int *linear_w, int linear_w_rows, int x_rows, int linear_w_columns, int *bias_vector, int bias_vector_rows, int bias_vector_cols, int layer_index, int *out) {
  int *wx, *bias;
  int *wt;
  cudaMalloc((void**)&wt, sizeof(int) *x_rows * linear_w_rows);
  matrix_transpose<<<linear_w_rows, linear_w_columns>>>(linear_w, wt);
  //dimensions of wt linear_w_columns * linear_w_rows
  cudaMalloc((void**)&bias, sizeof(int) * bias_vector_rows *x_rows * bias_vector_cols);

  // wx+b
  cudaMalloc((void**)&wx, sizeof(int) * x_rows *linear_w_rows);
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

// matrix *autodiff_forward(layers *layer_list, int *input) {
// //   layer *current_layer = NULL;
// //   matrix *output = NULL;

// //   traverse_layers_forward(layer_list, current_layer) {
// //         output = linear_layer_functions.forward(input, current_layer->internal);
// //         input = output;
// //   }
   

//   return output;
// }

int *autodiff_forward(int *input) { 
    // layer 0
    cudaMalloc((void**)&d_out0, sizeof(int) *w0_rows * input_rows);
    linear_layer_forward(input, d_w0, w0_rows, input_rows, w0_cols, d_b0, b0_rows, b0_cols, 0, d_out0);
    //layer 1
    cudaMalloc((void**)&d_out1, sizeof(int) *w1_rows * out0_rows);
    linear_layer_forward(d_out0, d_w1, w1_rows, out0_rows, w1_cols, d_b1, b1_rows, b1_cols, 1, d_out1);
    //layer 2
    cudaMalloc((void**)&d_out2, sizeof(int) *w2_rows * out1_rows);
    linear_layer_forward(d_out1, d_w2, w2_rows, out1_rows, w2_cols, d_b2, b2_rows, b2_cols, 2, d_out2);
    int *out2;
    cudaMemcpy(out2, d_out2, sizeof(int) * out2_cols *out2_rows, cudaMemcpyDeviceToHost);
    return out2;
}

int *readahead_class_net_inference(int *input) {
  return autodiff_forward(input);
}

// int predict_readahead_class(readahead_class_net *readahead,
//                             int current_readahead_val) {
//   matrix *normalized_data = NULL, *indv_result = NULL;
//   int class = 0;

//   normalized_data =
//       get_normalized_readahead_data(readahead, current_readahead_val);

//   // kml_debug("normalized per-disk data:\n");
//   // print_matrix(normalized_data);
//   indv_result = readahead_class_net_inference(normalized_data, readahead);
//   class = matrix_argmax(indv_result);

//   cleanup_autodiff(readahead->layer_list);
//   free_matrix(normalized_data);

//   return class;
// }

int predict_readahead_class(int *input) {
    int *d_readahead_norm_online_data;
    int readahead_online_data_cols;
    cudaMalloc((void**)&d_readahead_norm_online_data, sizeof(int) *readahead_online_data_cols);
    get_normalized_readahead_data(input, 1, d_readahead_norm_online_data);
  int cls = 0;
  int *indv_result = readahead_class_net_inference(d_readahead_norm_online_data);
  cls = matrix_argmax(indv_result, out2_rows, out2_cols);

  return cls;
}
void cleanup() {
    cudaFree(d_w0);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_b0);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_out0);
    cudaFree(d_out1);
    cudaFree(d_out2);
}


int main() {
    //dimensions of weights of layer 0 : 15 *5
    //dimensions of bias of layer 0 : 15 *1
    //dimensions of weights of layer 1 : 5 *10
    //dimensions of bias of layer 1 : 5 *1
    //dimensions of weights of layer 2 : 4 * 5
    //dimensions of bias of layer 2 : 4 * 1
    w0_rows = 15;
    w0_cols = 5;
    b0_rows = 15;
    b0_cols = 1;
    w1_rows = 5;
    w1_cols = 10;
    b1_rows = 5;
    b1_cols = 1; 
    w2_rows = 4;
    w2_cols = 5;
    b2_rows = 4;
    b2_cols = 1;
    input_rows = 15;
    cudaMalloc((void**)&d_w0, sizeof(int) *w0_rows * w0_cols);
    cudaMalloc((void**)&d_b0, sizeof(int) *b0_rows * b0_cols);
    cudaMalloc((void**)&d_w1, sizeof(int) *w1_rows * w1_cols);
    cudaMalloc((void**)&d_b1, sizeof(int) *b1_rows * b1_cols);
    cudaMalloc((void**)&d_w2, sizeof(int) *w2_rows * w2_cols);
    cudaMalloc((void**)&d_b2, sizeof(int) *b2_rows * b2_cols);
    int *d_input;
    cudaMalloc((void**)&d_input, sizeof(int) *input_rows);
    printf("\n %d", predict_readahead_class(d_input));

    cleanup();

    return 0;
}