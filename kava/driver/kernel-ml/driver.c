#include "weights.h"
#include <linux/delay.h>
#include <linux/ktime.h>
#include "helpers.h"
//#include <asm/fpu/api.h>
#define BLOCK_SIZE 16


static char *cubin_path = "kml.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to kml.cubin, default ./kml.cubin");

static int run_cpu(void) {
    return 0;
}

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

CUdeviceptr d_w0, d_b0, d_w1, d_b1, d_w2, d_b2, d_out0, d_out1, d_out2, d_input;


static void setup_gpu(int batch_size) {
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
    check_error(cuMemAlloc((CUdeviceptr*) &d_w0, sizeof(double) *w0_rows * w0_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_w0, w0, sizeof(double) * w0_rows * w0_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_b0, sizeof(double) *b0_rows * b0_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_b0, b0, sizeof(double) * b0_rows * b0_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_w1, sizeof(double) *w1_rows * w1_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_w1, w1, sizeof(double) * w1_rows * w1_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_b1, sizeof(double) *b1_rows * b1_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_b1, b1, sizeof(double) * b1_rows * b1_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_w2, sizeof(double) *w2_rows * w2_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_w2, w2, sizeof(double) * w2_rows * w2_cols), "cuMemcpyHtoD", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_b2, sizeof(double) *b2_rows * b2_cols), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_b2, b2, sizeof(double) * b2_rows * b2_cols), "cuMemcpyHtoD", __LINE__);
    
    double input[5] = { -0.586797, 5.456822, 5.456966, -0.297318, -1.184651};

    check_error(cuMemAlloc((CUdeviceptr*) &d_input, sizeof(double) *input_features), "cuMemAlloc ", __LINE__);
    check_error(cuMemcpyHtoD(d_input, input, sizeof(double) * input_features), "cuMemcpyHtoD", __LINE__);

}

void clean_batch(void) {

    cuMemFree(d_w0);
    cuMemFree(d_w1);
    cuMemFree(d_w2);
    cuMemFree(d_b0);
    cuMemFree(d_b1);
    cuMemFree(d_b2);
    cuMemFree(d_out0);
    cuMemFree(d_out1);
    cuMemFree(d_out2);
	cuMemFree(d_input);
}



void linear_layer_forward(double *x, double *linear_w, int linear_w_rows, int x_rows, int linear_w_columns, double *bias_vector, int bias_vector_rows, int bias_vector_cols, int layer_index, double *out,
CUfunction* matrix_transpose, CUfunction* matrix_mult, CUfunction* matrix_repmat, CUfunction* matrix_add ) {
    CUdeviceptr wx, bias;
    CUdeviceptr wt;
    check_error(cuMemAlloc((CUdeviceptr*) &wt, sizeof(double) *x_rows * linear_w_rows), "cuMemAlloc ", __LINE__);

    void *args[] = {
		&linear_w, &wt
	};
    check_error(cuLaunchKernel(*matrix_transpose, 
				linear_w_rows, 1, 1,          //blocks
				linear_w_columns, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_transpose<<<linear_w_rows, linear_w_columns>>>(linear_w, wt);
    //dimensions of wt linear_w_columns * linear_w_rows

    check_error(cuMemAlloc((CUdeviceptr*) &bias, sizeof(double) *bias_vector_rows *x_rows * bias_vector_cols), "cuMemAlloc ", __LINE__);
    // wx+b
    check_error(cuMemAlloc((CUdeviceptr*) &wx, sizeof(double) * x_rows *linear_w_rows), "cuMemAlloc ", __LINE__);

    //wx = matrix_mult(x, wt);
    unsigned int grid_rows = (x_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (linear_w_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    void *args[] = {
		&x, &wt, &wx, x_rows, linear_w_columns,linear_w_rows
	};

    check_error(cuLaunchKernel(*matrix_transpose, 
				dimGrid, 1, 1,          //blocks
				dimBlock, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

   // matrix_mult<<<dimGrid, dimBlock>>>( x, wt, wx, x_rows, linear_w_columns,linear_w_rows);

    int rep_val = 1;
    void *args[] = {
		&bias_vector, x_rows, rep_val, bias_vector_rows, bias_vector_cols, &bias
	};

    check_error(cuLaunchKernel(*matrix_repmat, 
				bias_vector_rows, 1, 1,          //blocks
				bias_vector_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_repmat<<<bias_vector_rows, bias_vector_cols>>>(bias_vector, x_rows, 1, bias_vector_rows, bias_vector_cols, bias);
    //bias = matrix_repmat(linear->bias_vector, wx->rows, 1);

    void *args[] = {
		&wx, &bias, &out
	};

    check_error(cuLaunchKernel(*matrix_add, 
				x_rows, 1, 1,          //blocks
				linear_w_rows, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_add<<<x_rows, linear_w_rows>>>(wx, bias, out);

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

    cuMemFree(wx);
    cuMemFree(bias);
    cuMemFree(wt);

}


double *autodiff_forward(CUdeviceptr d_readahead_norm_online_data, CUfunction* matrix_mult_constant, CUfunction* matrix_add, CUfunction* matrix_div_constant, CUfunction* set_matrix_with_matrix, 
    CUfunction* matrix_sub, CUfunction* matrix_elementwise_mult, CUfunction* matrix_elementwise_div, CUfunction* matrix_map, CUfunction* matrix_transpose, CUfunction* matrix_repmat,
    CUfunction* matrix_mult) { 
    // layer 0
    check_error(cuMemAlloc((CUdeviceptr*) &d_out0, sizeof(double) *w0_rows * input_rows), "cuMemAlloc ", __LINE__);
    linear_layer_forward(d_readahead_norm_online_data, d_w0, w0_rows, input_rows, w0_cols, d_b0, b0_rows, b0_cols, 0, d_out0,
    &matrix_transpose, &matrix_mult, &matrix_repmat, &matrix_add);
    //layer 1
    check_error(cuMemAlloc((CUdeviceptr*) &d_out1, sizeof(double) *w1_rows * out0_rows), "cuMemAlloc ", __LINE__);
    linear_layer_forward(d_out0, d_w1, w1_rows, out0_rows, w1_cols, d_b1, b1_rows, b1_cols, 1, d_out1,
    &matrix_transpose, &matrix_mult, &matrix_repmat, &matrix_add);
    //layer 2
    check_error(cuMemAlloc((CUdeviceptr*) &d_out2, sizeof(double) *w2_rows * out1_rows), "cuMemAlloc ", __LINE__);
    linear_layer_forward(d_out1, d_w2, w2_rows, out1_rows, w2_cols, d_b2, b2_rows, b2_cols, 2, d_out2,
    &matrix_transpose, &matrix_mult, &matrix_repmat, &matrix_add);

    double *out2 = (double*) kava_alloc(sizeof(double)*out2_cols *out2_rows);
    check_error(cuMemcpyDtoH(out2, d_out2, sizeof(double) * out2_cols *out2_rows), "cuMemcpyDtoH", __LINE__);
    //cudaMemcpy(out2, d_out2, sizeof(int) * out2_cols *out2_rows, cudaMemcpyDeviceToHost);
    return out2;
}

double *readahead_class_net_inference(CUdeviceptr d_readahead_norm_online_data, CUfunction* matrix_mult_constant, CUfunction* matrix_add, CUfunction* matrix_div_constant, CUfunction* set_matrix_with_matrix, 
    CUfunction* matrix_sub, CUfunction* matrix_elementwise_mult, CUfunction* matrix_elementwise_div, CUfunction* matrix_map, CUfunction* matrix_transpose, CUfunction* matrix_repmat,
    CUfunction* matrix_mult) {
    return autodiff_forward(d_readahead_norm_online_data, matrix_mult_constant, matrix_add, matrix_div_constant, set_matrix_with_matrix, 
    matrix_sub, matrix_elementwise_mult, matrix_elementwise_div, matrix_map, matrix_transpose, matrix_repmat,
    matrix_mult);
}

void readahead_normalized_online_data(int readahead_online_data_cols, int readahead_std_dev_rows, int readahead_std_dev_cols, 
    int readahead_avg_rows, int readahead_avg_cols, int readahead_variance_rows, int readahead_variance_cols,
                                      int readahead_val , double *readahead_norm_online_data,
                                      CUfunction* matrix_mult_constant, CUfunction* matrix_add, CUfunction* matrix_div_constant,
                                      CUfunction* set_matrix_with_matrix,CUfunction* matrix_sub,
                                      CUfunction* matrix_elementwise_mult, CUfunction* matrix_map,
                                      CUfunction* matrix_elementwise_div) {
    CUdeviceptr diff, local_average, local_std_dev, local_variance, readahead_norm_online_data_last_values;
    
    //cudaMalloc((void**)&diff, sizeof(double) * readahead_online_data_cols);
    check_error(cuMemAlloc((CUdeviceptr*) &diff, sizeof(double) * readahead_online_data_cols), "cuMemAlloc ", __LINE__);

    //cudaMalloc((void**)&local_average, sizeof(double) * readahead_avg_rows * readahead_avg_cols);
    check_error(cuMemAlloc((CUdeviceptr*) &local_average, sizeof(double) * readahead_avg_rows * readahead_avg_cols), "cuMemAlloc ", __LINE__);

    //cudaMalloc((void**)&local_std_dev, sizeof(double) * readahead_std_dev_rows * readahead_std_dev_cols);
    check_error(cuMemAlloc((CUdeviceptr*) &local_std_dev, sizeof(double) * readahead_std_dev_rows * readahead_std_dev_cols), "cuMemAlloc ", __LINE__);

    //cudaMalloc((void**)&local_variance, sizeof(double) * readahead_variance_rows * readahead_variance_cols);
    check_error(cuMemAlloc((CUdeviceptr*) &local_variance, sizeof(double) * readahead_variance_rows * readahead_variance_cols), "cuMemAlloc ", __LINE__);

    //cudaMalloc((void**)&local_average, sizeof(double) * readahead_online_data_cols);
    check_error(cuMemAlloc((CUdeviceptr*) &local_average, sizeof(double) * readahead_online_data_cols), "cuMemAlloc ", __LINE__);

    //cudaMalloc((void**)&readahead_norm_online_data_last_values, sizeof(double) * readahead_online_data_cols);
    check_error(cuMemAlloc((CUdeviceptr*) &readahead_norm_online_data_last_values, 
    sizeof(double) * readahead_online_data_cols), "cuMemAlloc ", __LINE__);
 
    int n_seconds = 10;

    void *args[] = {
		&local_average, &n_seconds, &diff
	};

    check_error(cuLaunchKernel(*matrix_mult_constant, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    //matrix_mult_constant<<<1, readahead_online_data_cols>>>(local_average, n_seconds, diff);
    void *args[] = {
		&d_input, &diff, &diff
	};

    check_error(cuLaunchKernel(*matrix_add, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_add<<<1, readahead_online_data_cols>>>(readahead_online_data, diff, diff);

    void *args[] = {
		&diff, &n_seconds, &diff
	};

    check_error(cuLaunchKernel(*matrix_div_constant, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_div_constant<<<1, readahead_online_data_cols>>>(diff, n_seconds, diff);

    void *args[] = {
		&diff, &local_average
	};

    check_error(cuLaunchKernel(*set_matrix_with_matrix, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //set_matrix_with_matrix<<<1, readahead_online_data_cols>>>(diff, local_average);
    // print_matrix(readahead->norm_data_stat.average);


    void *args[] = {
		&d_input, &readahead_norm_online_data_last_values, &diff
	};

    check_error(cuLaunchKernel(*matrix_sub, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_sub<<<1, readahead_online_data_cols>>>(readahead_online_data, readahead_norm_online_data_last_values, diff);

    void *args[] = {
		&diff, &diff, &diff
	};

    check_error(cuLaunchKernel(*matrix_elementwise_mult, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_elementwise_mult<<<1, readahead_online_data_cols>>>(diff, diff, diff);

    void *args[] = {
		&local_variance, &n_seconds, &local_variance
	};

    check_error(cuLaunchKernel(*matrix_mult_constant, 
				readahead_variance_rows, 1, 1,          //blocks
				readahead_variance_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_mult_constant<<<readahead_variance_rows, readahead_variance_cols>>>(local_variance, n_seconds, local_variance);

    void *args[] = {
		&local_variance, &diff, &local_variance
	};

    check_error(cuLaunchKernel(*matrix_add, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_add<<<1, readahead_online_data_cols>>>(local_variance, diff, local_variance);

    void *args[] = {
		&local_variance, &n_seconds, &local_variance
	};

    check_error(cuLaunchKernel(*matrix_div_constant, 
				readahead_variance_rows, 1, 1,          //blocks
				readahead_variance_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_div_constant<<<readahead_variance_rows, readahead_variance_cols>>>(local_variance, n_seconds, local_variance);


    void *args[] = {
		&local_variance, &fast_sqrt_d, &local_std_dev
	};

    check_error(cuLaunchKernel(*matrix_map, 
				readahead_variance_rows, 1, 1,          //blocks
				readahead_variance_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    //matrix_map<<<readahead_variance_rows, readahead_variance_cols>>>(local_variance, fast_sqrt_d, local_std_dev);
    // print_matrix(readahead->norm_data_stat.std_dev);
    void *args[] = {
		&d_input, &local_average, &readahead_norm_online_data
	};

    check_error(cuLaunchKernel(*matrix_map, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    //matrix_sub<<<1, readahead_online_data_cols>>>(readahead_online_data, local_average, readahead_norm_online_data);

    void *args[] = {
		&d_input, &local_std_dev, &readahead_norm_online_data
	};

    check_error(cuLaunchKernel(*matrix_elementwise_div, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //matrix_elementwise_div<<<1, readahead_online_data_cols>>>(readahead_norm_online_data, local_std_dev, readahead_norm_online_data);

    void *args[] = {
		&d_input, &readahead_norm_online_data_last_values
	};

    check_error(cuLaunchKernel(*set_matrix_with_matrix, 
				1, 1, 1,          //blocks
				readahead_online_data_cols, 1, 1,   //threads per block
				0,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);
    //set_matrix_with_matrix<<<1, readahead_online_data_cols>>>(readahead_online_data, readahead_norm_online_data_last_values);

    cuMemFree(diff);
    cuMemFree(local_average);
    cuMemFree(local_std_dev);
    cuMemFree(local_variance);
}

void get_normalized_readahead_data(CUdeviceptr d_readahead_norm_online_data, CUfunction* matrix_mult_constant, CUfunction* matrix_add, CUfunction* matrix_div_constant,
                                      CUfunction* set_matrix_with_matrix,CUfunction* matrix_sub,
                                      CUfunction* matrix_elementwise_mult, CUfunction* matrix_map,
                                      CUfunction* matrix_elementwise_div) {
    int readahead_online_data_cols = 5, readahead_std_dev_rows = 1, 
        readahead_std_dev_cols = 5, readahead_avg_rows = 1, readahead_avg_cols = 5, readahead_variance_rows = 1, readahead_variance_cols = 5;
    
    // cudaMalloc((void**)&d_readahead_norm_online_data, sizeof(double) *readahead_online_data_cols);
    
    readahead_normalized_online_data(readahead_online_data_cols, readahead_std_dev_rows, readahead_std_dev_cols,
        readahead_avg_rows, readahead_avg_cols, readahead_variance_rows, readahead_variance_cols, d_readahead_norm_online_data, 
                                    matrix_mult_constant, matrix_add, matrix_div_constant,
                                      set_matrix_with_matrix, matrix_sub, matrix_elementwise_mult, 
                                      matrix_map, matrix_elementwise_div);
}

int predict_readahead_class(CUfunction* matrix_mult_constant, CUfunction* matrix_add, CUfunction* matrix_div_constant,
                                      CUfunction* set_matrix_with_matrix,CUfunction* matrix_sub,
                                      CUfunction* matrix_elementwise_mult, CUfunction* matrix_map,
                                      CUfunction* matrix_elementwise_div, CUfunction* matrix_transpose,
                                      CUfunction* matrix_repmat, CUfunction* matrix_mult) {
    CUdeviceptr d_readahead_norm_online_data;
    int readahead_online_data_cols = 5;
    check_error(cuMemAlloc((CUdeviceptr*) &d_readahead_norm_online_data, 
    sizeof(double) *readahead_online_data_cols), "cuMemAlloc ", __LINE__);
    get_normalized_readahead_data(d_readahead_norm_online_data, matrix_mult_constant, matrix_add, matrix_div_constant,
                                      set_matrix_with_matrix, matrix_sub, matrix_elementwise_mult, 
                                      matrix_map, matrix_elementwise_div);
    int cls = 0;
    double *indv_result = readahead_class_net_inference(d_readahead_norm_online_data, matrix_mult_constant, matrix_add, matrix_div_constant, set_matrix_with_matrix, 
    matrix_sub, matrix_elementwise_mult, matrix_elementwise_div, matrix_map, matrix_transpose, matrix_repmat,
    matrix_mult);
    cls = matrix_argmax(indv_result, out2_rows, out2_cols);

    return cls;
}

static int run_gpu(void) {
    int i, j;
    int RUNS;
    const int n = 1024;
    
    int batch_size;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;
  
    CUcontext cuContext;
    gpu_init(0, &cuContext);

    CUfunction matrix_mult_constant, matrix_add, matrix_div_constant, set_matrix_with_matrix, 
    matrix_sub, matrix_elementwise_mult, matrix_elementwise_div, matrix_map, matrix_transpose, matrix_repmat,
    matrix_mult;
    int n_batches = 1;

    gpu_get_cufunc(cubin_path, "_Z20matrix_mult_constantPddS_", &matrix_mult_constant);
    gpu_get_cufunc(cubin_path, "_Z10matrix_addPdS_S_", &matrix_add);
    gpu_get_cufunc(cubin_path, "_Z19matrix_div_constantPddS_", &matrix_div_constant);
    gpu_get_cufunc(cubin_path, "_Z22set_matrix_with_matrixPdS_", &set_matrix_with_matrix);
    gpu_get_cufunc(cubin_path, "_Z10matrix_subPdS_S_", &matrix_sub);
    gpu_get_cufunc(cubin_path, "_Z23matrix_elementwise_multPdS_S_", &matrix_elementwise_mult);
    gpu_get_cufunc(cubin_path, "_Z22matrix_elementwise_divPdS_S_", &matrix_elementwise_div);
    gpu_get_cufunc(cubin_path, "_Z10matrix_mapPdPFddES_", &matrix_map);
    gpu_get_cufunc(cubin_path, "_Z16matrix_transposePdS_", &matrix_transpose);
    gpu_get_cufunc(cubin_path, "_Z13matrix_repmatPdiiiiS_", &matrix_repmat);
    gpu_get_cufunc(cubin_path, "_Z11matrix_multPdS_S_iii", &matrix_mult);
    RUNS = 10;
    comp_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
    total_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);

    //flatten n inputs, which is enough for all batches
    //parallel_input = (long*) kmalloc(n*31*sizeof(long), GFP_KERNEL);
    //flatten_input(n, input);
    //res = (bool*) kmalloc(n*sizeof(bool), GFP_KERNEL);

    for (i = 0 ; i < n_batches ; i++) {
        batch_size = 1;//batch_sizes[i];
        // setup is only run once per batch size (cuda mallocs)
        setup_gpu(batch_size);    
        // copy inputs to GPU each time we run
        //copy_batch_inputs(batch_size);
        predict_readahead_class(&matrix_mult_constant, &matrix_add, &matrix_div_constant,
                                      &set_matrix_with_matrix, &matrix_sub,
                                      &matrix_elementwise_mult, &matrix_map,
                                      &matrix_elementwise_div, &matrix_transpose,
                                      &matrix_repmat, &matrix_mult);

        //warmup
        //gpu_inference(&batch_linnos_mid_layer_kernel, &batch_linnos_final_layer_kernel, batch_size);
        usleep_range(1000, 2000);
        cuCtxSynchronize();
    
        /*for (j = 0 ; j < RUNS ; j++) {
            //PRINT(V_INFO, "Runing batch %d/%d for batch size %d\n", k+1, n/batch_size, batch_size);
            t_start = ktime_get_ns();
            //copy_batch_inputs(batch_size);
            c_start = ktime_get_ns();
            gpu_inference(&batch_linnos_mid_layer_kernel, &batch_linnos_final_layer_kernel, batch_size);
            c_stop = ktime_get_ns();
            get_result_batch(batch_size);
            t_stop = ktime_get_ns();
            comp_run_times[j] = (c_stop - c_start);
            total_run_times[j] = (t_stop - t_start);
	    }*/

	    avg = 0; avg_total = 0;
        best = 0; best_total = 0;
        for (j = 0 ; j < RUNS ; j++) {
            avg += comp_run_times[j];
            avg_total += total_run_times[j];
            if (best == 0 || comp_run_times[j] < best) best = comp_run_times[j];
            if (best_total == 0 || total_run_times[j] < best_total) best_total = total_run_times[j];
        }
        avg = avg / (1000*RUNS); avg_total = avg_total / (1000*RUNS);
        best = best / 1000; best_total = best_total / 1000;

        PRINT(V_INFO, "GPU batch_%d, %lld, %lld, %lld, %lld\n", batch_size, avg, avg_total, best, best_total);
        clean_batch();
	}

    //cleanup();
    return 0;
}


/**
 * Program main
 */
static int __init kml_init(void)
{
	return run_gpu();
}

static void __exit kml_fini(void)
{

}

module_init(kml_init);
module_exit(kml_fini);

MODULE_AUTHOR("Isha Tarte");
MODULE_DESCRIPTION("Kernel module of a KML program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
