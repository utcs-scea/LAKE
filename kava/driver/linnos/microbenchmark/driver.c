#include "weights.h"
#include <linux/delay.h>
#include <linux/ktime.h>
#include "helpers.h"
//#include <asm/fpu/api.h>
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2


static char *cubin_path = "linnos.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to linnos.cubin, default ./linnos.cubin");

static int run_cpu(void) {
    return 0;
}

CUdeviceptr d_weight_0_T_ent, d_weight_1_T_ent, d_bias_0_ent, d_bias_1_ent, d_input_vec_i, d_mid_res_i, d_final_res_i;
static long *final_res_i;

static void setup_batch(int batch_size, long* input_vec_i) {
    static long *weight_0_T_ent, * bias_0_ent, *weight_1_T_ent, * bias_1_ent; 
    static long *parallel_input;
    final_res_i = (long*) kmalloc(batch_size*64*sizeof(long), GFP_KERNEL);
    parallel_input = (long*) kmalloc(batch_size*31*sizeof(long), GFP_KERNEL);

    int b, j;
	for(b = 0 ; b < batch_size; b++) {
		for(j = 0; j < 31; j++)
			parallel_input[ b*31 + j ] = input_vec_i[j];
	}

	weight_0_T_ent = &weight_i_0_T[0][0];
    weight_1_T_ent = &weight_i_1[0][0];
    	bias_0_ent = bias_i_0;
    	bias_1_ent = bias_i_1;
	//PRINT(V_INFO, "starting cuMalloc!!  size %d\n", sizeof(long) * 256*31);
	check_error(cuMemAlloc((CUdeviceptr*) &d_weight_0_T_ent, sizeof(long) * 256*31), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_weight_1_T_ent, sizeof(long) * 256*2), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_bias_0_ent, sizeof(long) * 256), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_bias_1_ent, sizeof(long) * 2), "cuMemAlloc ", __LINE__);
    
    check_error(cuMemAlloc((CUdeviceptr*) &d_input_vec_i, sizeof(long) * 31 * batch_size), "cuMemAlloc ", __LINE__);

    check_error(cuMemAlloc((CUdeviceptr*) &d_mid_res_i, sizeof(long) *LEN_LAYER_0 * batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_final_res_i, sizeof(long) *LEN_LAYER_1 * batch_size *32), "cuMemAlloc ", __LINE__);

    check_error(cuMemcpyHtoD(d_weight_0_T_ent, weight_0_T_ent, sizeof(long) * 256*31), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD(d_weight_1_T_ent, weight_1_T_ent, sizeof(long) * 256*2), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD(d_bias_0_ent, bias_0_ent, sizeof(long) * 256), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD(d_bias_1_ent, bias_1_ent, sizeof(long) * 2), "cuMemcpyHtoD", __LINE__);
   
    check_error(cuMemcpyHtoD(d_input_vec_i, parallel_input, sizeof(long) * 31 * batch_size), "cuMemcpyHtoD", __LINE__);
    kfree(parallel_input);
}

int gpu_inference(CUfunction* cufunc1, CUfunction* cufunc2, int batch_size) {
    void *args[] = {
		&d_weight_0_T_ent, &d_bias_0_ent, &d_input_vec_i, &d_mid_res_i
	};

    check_error(cuLaunchKernel(*cufunc1, 
				batch_size, 1, 1,          //blocks
				256, 1, 1,   //threads per block
				NULL,   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    cuCtxSynchronize();

    void *args1[] = {
		&d_weight_1_T_ent, &d_bias_1_ent, &d_mid_res_i, &d_final_res_i
	};

    check_error(cuLaunchKernel(*cufunc2, 
				batch_size, 1, 1,          //blocks
				64, 1, 1,   //threads per block
				NULL,   //shared mem
                NULL, args1, NULL),
			"cuLaunchKernel", __LINE__);


    return 0;
}

void get_result_batch(int batch_size) {
    check_error(cuMemcpyDtoH(final_res_i, d_final_res_i, sizeof(long) * 64 * batch_size), "cuMemcpyDtoH", __LINE__);
	
	bool res[batch_size];
    int i;
	for(i = 0; i < batch_size; i++) {
		res[i] = final_res_i[i*64]>=(final_res_i[i *64 + 32])? false: true;
	}
}

void clean_batch(void) {
	cuMemFree(d_input_vec_i);
	cuMemFree(d_weight_0_T_ent);
	cuMemFree(d_weight_1_T_ent);
	cuMemFree(d_bias_0_ent);
	cuMemFree(d_bias_1_ent);
	cuMemFree(d_mid_res_i);
	cuMemFree(d_final_res_i);
	kfree(final_res_i);
}

static int run_gpu(void) {
    int i, j;
    int RUNS;
    int batch_sizes[] = {8, 16, 32, 64, 128, 256, 512};
    int n_batches = 7;
    const int n = 1024;
    
    int batch_size;
    long input[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;
  
    CUcontext cuContext;
    CUdeviceptr d_inputs, d_w1, d_b1, d_w2, d_results;
    gpu_init(0, &cuContext);

    CUfunction batch_linnos_final_layer_kernel, batch_linnos_mid_layer_kernel;

    gpu_get_cufunc(cubin_path, "_Z28prediction_final_layer_batchPlS_S_S_", &batch_linnos_final_layer_kernel);
    gpu_get_cufunc(cubin_path, "_Z26prediction_mid_layer_batchPlS_S_S_", &batch_linnos_mid_layer_kernel);
    RUNS = 10;
    comp_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
    total_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);

    for (i = 0 ; i < n_batches ; i++) {
         batch_size = batch_sizes[i];
        setup_batch(batch_size, input);
        gpu_inference(&batch_linnos_mid_layer_kernel, &batch_linnos_final_layer_kernel, batch_size);
        cuCtxSynchronize();
    
        for (j = 0 ; j < RUNS ; j++) {
            int k;
            //PRINT(V_INFO, "Runing batch %d/%d for batch size %d\n", k+1, n/batch_size, batch_size);
            t_start = ktime_get_ns();
            setup_batch(batch_size, input);
            c_start = ktime_get_ns();
            gpu_inference(&batch_linnos_mid_layer_kernel, &batch_linnos_final_layer_kernel, batch_size);
            c_stop = ktime_get_ns();
            get_result_batch(batch_size);
            t_stop = ktime_get_ns();
            comp_run_times[j] = (c_stop - c_start);
            total_run_times[j] = (t_stop - t_start);
	    }
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

    return 0;
}


/**
 * Program main
 */
static int __init linnos_init(void)
{
	return run_gpu();
}

static void __exit linnos_fini(void)
{
    //cleanup
}

module_init(linnos_init);
module_exit(linnos_fini);

MODULE_AUTHOR("Isha Tarte");
MODULE_DESCRIPTION("Kernel module of a linnos program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
