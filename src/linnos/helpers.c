#include "variables.h"
#include "helpers.h"
#include "predictors.h"

static void gpu_cuda_init(int dev) {
    CUdevice cuDevice;
    CUresult res;

    cuInit(0);
    res = cuDeviceGet(&cuDevice, dev);
    if (res != CUDA_SUCCESS){
        PRINT("cannot acquire device 0\n");
    }

    res = cuCtxCreate(&cuctx, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        PRINT("cannot create context\n");
    }
}

static void gpu_get_cufunc(const char* cubin, char* kname, CUfunction *func) {
    CUmodule cuModule;
    CUresult res;
    res = cuModuleLoad(&cuModule, cubin);
    if (res != CUDA_SUCCESS) {
        PRINT("cannot load module: %d\n", res);
    }

    res = cuModuleGetFunction(func, cuModule, kname);
    if (res != CUDA_SUCCESS){
        PRINT("cannot acquire kernel handle\n");
    }
}

//this function gets the CUfuncs, allocates and copies weights
//and allocates memory for max_batch_size inputs
void initialize_gpu(const char* cubin_path, long **weights, int max_batch_size) {
	long *kbuf_weight_0_T_ent;
    long *kbuf_weight_1_T_ent;
    long *kbuf_bias_0_ent;
    long *kbuf_bias_1_ent;
    //intialize kernels
    gpu_cuda_init(0);
    gpu_get_cufunc(cubin_path, "_Z28prediction_final_layer_batchPlS_S_S_", &batch_linnos_final_layer_kernel);
    gpu_get_cufunc(cubin_path, "_Z26prediction_mid_layer_batchPlS_S_S_", &batch_linnos_mid_layer_kernel);

	PRINT("cubin load done\n");
	//initialize variables
	kbuf_weight_0_T_ent = (long*) kava_alloc(256*31*sizeof(long));
    memcpy(kbuf_weight_0_T_ent, weights[0], 256*31*sizeof(long));
    kbuf_weight_1_T_ent = (long*) kava_alloc(256*2*sizeof(long));
    memcpy(kbuf_weight_1_T_ent, weights[1], 256*2*sizeof(long));
    kbuf_bias_0_ent = (long*) kava_alloc(256*sizeof(long));
    memcpy(kbuf_bias_0_ent, weights[2], 256*sizeof(long));
    kbuf_bias_1_ent = (long*) kava_alloc(2*sizeof(long));
    memcpy(kbuf_bias_1_ent, weights[3], 2*sizeof(long));
	PRINT("weights kbuf done\n");
	
	check_error(cuMemAlloc((CUdeviceptr*) &d_weight_0_T_ent, sizeof(long) * 256*31), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_weight_1_T_ent, sizeof(long) * 256*2), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_bias_0_ent, sizeof(long) * 256), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_bias_1_ent, sizeof(long) * 2), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_input_vec_i, sizeof(long) * LEN_INPUT * max_batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_mid_res_i, sizeof(long) *LEN_LAYER_0 * max_batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_final_res_i, sizeof(long) *LEN_LAYER_1 * max_batch_size *32), "cuMemAlloc ", __LINE__);

	PRINT("malloc done\n");
    check_error(cuMemcpyHtoD(d_weight_0_T_ent, kbuf_weight_0_T_ent, sizeof(long) * 256*31), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD(d_weight_1_T_ent, kbuf_weight_1_T_ent, sizeof(long) * 256*2), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD(d_bias_0_ent, kbuf_bias_0_ent, sizeof(long) * 256), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD(d_bias_1_ent, kbuf_bias_1_ent, sizeof(long) * 2), "cuMemcpyHtoD", __LINE__);
	PRINT("memcpy done\n");
    kava_free(kbuf_weight_0_T_ent);
    kava_free(kbuf_weight_1_T_ent);
    kava_free(kbuf_bias_0_ent);
    kava_free(kbuf_bias_1_ent);
}

void gpu_cuda_cleanup(void) {
	cuMemFree(d_input_vec_i);
	cuMemFree(d_weight_0_T_ent);
	cuMemFree(d_weight_1_T_ent);
	cuMemFree(d_bias_0_ent);
	cuMemFree(d_bias_1_ent);
	cuMemFree(d_mid_res_i);
	cuMemFree(d_final_res_i);
}

void check_malloc(void *p, const char* error_str, int line) {
	if (p == NULL) PRINT("ERROR: Failed to allocate %s (line %d)\n", error_str, line);
}

