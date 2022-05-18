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

// __global__ void readahead_normalized_online_data() {
//     static int *matrix;
//     int rows = 10;
//     int cols = 10;
//     //cudaMalloc((void**)&matrix, sizeof(int) * rows*cols);
// }

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

static double *d_w0, *d_b0, *d_w1, *d_b1, *d_w2, *d_b2, *d_out0, *d_out1, *d_out2;
static int w0_rows, w0_cols, b0_rows, b0_cols, w1_rows, w1_cols, b1_rows, b1_cols, w2_rows, w2_cols, b2_rows, b2_cols, input_rows;
static int out0_rows, out0_cols, out1_rows, out1_cols, out2_rows, out2_cols;

void readahead_normalized_online_data(double *readahead_online_data, int readahead_online_data_cols, int readahead_std_dev_rows, int readahead_std_dev_cols, 
    int readahead_avg_rows, int readahead_avg_cols, int readahead_variance_rows, int readahead_variance_cols,
                                      int readahead_val , double *readahead_norm_online_data) {
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
    double *diff, *local_average, *local_std_dev, *local_variance, *readahead_norm_online_data_last_values;
    
  cudaMalloc((void**)&diff, sizeof(double) * readahead_online_data_cols);
  cudaMalloc((void**)&local_average, sizeof(double) * readahead_avg_rows * readahead_avg_rows);
  cudaMalloc((void**)&local_std_dev, sizeof(double) * readahead_std_dev_rows * readahead_std_dev_cols);
  cudaMalloc((void**)&local_variance, sizeof(double) * readahead_variance_rows * readahead_variance_cols);
  cudaMalloc((void**)&readahead_norm_online_data, sizeof(double) * readahead_online_data_cols);
  cudaMalloc((void**)&readahead_norm_online_data_last_values, sizeof(double) * readahead_online_data_cols);
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

double *get_normalized_readahead_data(double *readahead,
                                      int current_readahead_val, double *d_readahead_norm_online_data) {
  double *normalized_data = NULL;
//   readahead_normalized_online_data((readahead_net *)readahead,
//                                    current_readahead_val, false);
//   normalized_data = matrix_double_conversion(readahead->norm_online_data);

int readahead_online_data_cols = 5, readahead_std_dev_rows = 1, 
    readahead_std_dev_cols = 5, readahead_avg_rows = 1, readahead_avg_cols = 5, readahead_variance_rows = 1, readahead_variance_cols = 5;
cudaMalloc((void**)&d_readahead_norm_online_data, sizeof(int) *readahead_online_data_cols);

    readahead_normalized_online_data(readahead, readahead_online_data_cols, readahead_std_dev_rows, readahead_std_dev_cols,
    readahead_avg_rows, readahead_avg_cols, readahead_variance_rows, readahead_variance_cols, current_readahead_val, d_readahead_norm_online_data);
  //normalized_data = matrix_double_conversion(readahead->norm_online_data);
    // int *normalized_data;
    // cudaMemcpy(normalized_data, d_readahead_norm_online_data, sizeof(int) * readahead_online_data_cols, cudaMemcpyDeviceToHost);

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

// matrix *autodiff_forward(layers *layer_list, int *input) {
// //   layer *current_layer = NULL;
// //   matrix *output = NULL;

// //   traverse_layers_forward(layer_list, current_layer) {
// //         output = linear_layer_functions.forward(input, current_layer->internal);
// //         input = output;
// //   }
   

//   return output;
// }

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

    double w0[15][5] = {
        {-0000011.669888, -0000000.249456, -0000000.038613, 00000005.642278, 00000007.012926},
        {-0000012.145359, 00000003.872515, 00000004.019009 ,00000012.550223, -0000002.621693},
        {00000008.960056, -0000000.016092, -0000000.562617, 00000015.599205, -0000000.361083},
        {00000014.456594, -0000013.322599, -0000013.313159, -0000019.182358, 00000001.948441},
        {-0000034.163129, 00000006.558228, 00000006.389304, -0000039.633775, 00000048.753821},
        {-0000013.934725, -0000007.519114, -0000007.329937, 00000011.234843, 00000004.198010},
        {00000009.273137, -0000027.453354, -0000027.124823, -0000005.129141, 00000026.243805},
        {00000009.617660, 00000013.975671, 00000014.221053, 00000029.918973, 00000018.547957},
        {-0000004.539204, -0000002.561855, -0000002.881559, -0000048.449138, -0000006.024052},
        {-0000006.996530, -0000003.499494, -0000004.001873, 00000025.095392, -0000000.873080},
        {-0000001.834595, -0000024.983553, -0000024.442274, -0000005.016328, 00000003.491228},
        {-0000015.516551, 00000001.166713, 00000001.190491, -0000023.617902, -0000002.510977},
        {00000002.785250, -0000003.971537, -0000004.621931, 00000019.081904, -0000002.339924},
        {-0000011.060982, -0000001.113953, -0000000.611457, -0000003.284909, -0000007.897714},
        {-0000023.032224, -0000007.876164, -0000008.074158, -0000001.024627, -0000002.299194}
    };

    double b0[15][1] = {
        {00000002.992922},
        {-0000004.898516},
        {00000011.470737},
        {-0000011.640097},
        {00000004.372712},
        {-0000009.422000},
        {-0000058.417139},
        {-0000005.164213},
        {00000001.074321},
        {00000019.854877},
        {00000003.557259},
        {-0000005.584953},
        {00000002.676954},
        {-0000004.404445},
        {00000010.303386}
    };

    double w1[5][15] = {
        {-0000010.416551, 00000010.207635, 00000014.135603, -0000004.558068, -0000008.942299, 00000007.590070, 00000002.929620, -0000008.077410, -0000028.280543, 00000017.591318, 00000009.423640, 00000001.113650, -0000003.239676, -0000010.321794, 00000008.653932},
        {00000012.242060, -0000014.918776, 00000009.806193, -0000021.781936, 00000008.023761, 00000014.767116, 00000007.154670, 00000017.055565, -0000013.487206, -0000010.394139, -0000009.279621, -0000021.993192, 00000024.587426, 00000001.663421, 00000000.846539},
        {00000006.164338, -0000014.798030, -0000023.657100, 00000001.923294, 00000001.072160, -0000007.740484, 00000007.343431, -0000002.514456, 00000010.546505, 00000000.892898, -0000001.336684, 00000009.573787, -0000003.529458, 00000004.246379, -0000018.515028},
        {00000000.025790, 00000001.776147, -0000008.665069, 00000020.336332, 00000005.798561, -0000004.780248, 00000034.501388, -0000010.909422, 00000014.693963, -0000002.547216, 00000005.864494, 00000027.920209, -0000040.353642, -0000002.663298, -0000009.888228},
        {00000010.370017, -0000005.095625, -0000009.117038, 00000018.484024, -0000018.706940, -0000001.956738, 00000014.861873, -0000011.606552, 00000028.985207, -0000029.506188, 00000002.412825, 00000020.593578, -0000022.539810, 00000010.869101, -0000013.389612}
    };

    double b1[5][1] = {
        {00000002.395463},
        {00000006.801791},
        {00000005.419590},
        {-0000001.822087},
        {00000018.575540}
    };

    double w2[4][5] = {
        {-0000013.901530, 00000047.290453, -0000018.861083, -0000037.144629, 00000048.703547},
        {00000032.041269, -0000016.726115, 00000008.311415, 00000024.001997, -0000057.524419},
        {00000004.571861, -0000020.739063, -0000020.472469, 00000015.574213, 00000005.334875},
        {-0000023.773152, -0000009.025779, 00000031.621847, -0000002.161713, 00000003.990320}
    };

    double b2[4][1] = {
        {-0000004.582073},
        {00000013.337211},
        {-0000002.816034},
        {-0000006.808400}
    };
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