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

//this is multi ssd ready
void copy_weights(long **weights, struct GPU_weights *state) {
    long *kbuf_weight_0_T_ent;
    long *kbuf_weight_1_T_ent;
    long *kbuf_bias_0_ent;
    long *kbuf_bias_1_ent;

    //initialize variables
	kbuf_weight_0_T_ent = (long*) kava_alloc(256*31*sizeof(long));
    memcpy(kbuf_weight_0_T_ent, weights[0], 256*31*sizeof(long));
    kbuf_weight_1_T_ent = (long*) kava_alloc(256*2*sizeof(long));
    memcpy(kbuf_weight_1_T_ent, weights[1], 256*2*sizeof(long));
    kbuf_bias_0_ent = (long*) kava_alloc(256*sizeof(long));
    memcpy(kbuf_bias_0_ent, weights[2], 256*sizeof(long));
    kbuf_bias_1_ent = (long*) kava_alloc(2*sizeof(long));
    memcpy(kbuf_bias_1_ent, weights[3], 2*sizeof(long));

	//check_error(cuMemAlloc((CUdeviceptr*) &state->d_weight_0_T_ent, sizeof(long) * 256*31), "cuMemAlloc ", __LINE__);
    //check_error(cuMemAlloc((CUdeviceptr*) &state->d_weight_1_T_ent, sizeof(long) * 256*2), "cuMemAlloc ", __LINE__);
    //check_error(cuMemAlloc((CUdeviceptr*) &state->d_bias_0_ent, sizeof(long) * 256), "cuMemAlloc ", __LINE__);
    //check_error(cuMemAlloc((CUdeviceptr*) &state->d_bias_1_ent, sizeof(long) * 2), "cuMemAlloc ", __LINE__);
	check_error(cuMemAlloc((CUdeviceptr*) &state->weights[0], sizeof(long) * 256*31), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[1], sizeof(long) * 256*2), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[2], sizeof(long) * 256), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &state->weights[3], sizeof(long) * 2), "cuMemAlloc ", __LINE__);

    check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[0], kbuf_weight_0_T_ent, sizeof(long) * 256*31), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[1], kbuf_weight_1_T_ent, sizeof(long) * 256*2), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[2], kbuf_bias_0_ent, sizeof(long) * 256), "cuMemcpyHtoD", __LINE__);
	check_error(cuMemcpyHtoD((CUdeviceptr )state->weights[3], kbuf_bias_1_ent, sizeof(long) * 2), "cuMemcpyHtoD", __LINE__);
    kava_free(kbuf_weight_0_T_ent);
    kava_free(kbuf_weight_1_T_ent);
    kava_free(kbuf_bias_0_ent);
    kava_free(kbuf_bias_1_ent);
}

void copy_results_from_gpu(u64 n_inputs) {
    cuMemcpyDtoH(gpu_outputs, d_final_res_i, sizeof(long) * 64 * n_inputs);
}

//this function gets the CUfuncs and allocates memory for max_batch_size inputs
void initialize_gpu(const char* cubin_path, int max_batch_size) {
    //intialize kernels
    if (cuctx) {
        return;
    }
    gpu_cuda_init(0);
    gpu_get_cufunc(cubin_path, "_Z28prediction_final_layer_batchPlS_S_S_", &batch_linnos_final_layer_kernel);
    gpu_get_cufunc(cubin_path, "_Z26prediction_mid_layer_batchPlS_S_S_", &batch_linnos_mid_layer_kernel);
    
    check_error(cuMemAlloc((CUdeviceptr*) &d_input_vec_i, sizeof(long) * LEN_INPUT * max_batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_mid_res_i, sizeof(long) *LEN_LAYER_0 * max_batch_size), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) &d_final_res_i, sizeof(long) * LEN_LAYER_1 * max_batch_size *32), "cuMemAlloc ", __LINE__);

    inputs_to_gpu = kava_alloc(LEN_INPUT * max_batch_size * sizeof(long));
    if (!inputs_to_gpu) {
        pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
    }
    gpu_outputs = kava_alloc(64 * max_batch_size * sizeof(long));
    if (!gpu_outputs) {
        pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
    }
}

void gpu_cuda_cleanup(struct GPU_weights *state) {
    int i;
    pr_warn("Cleaning up GPU state\n");
    for(i = 0; i <4 ; i++) {
        cuMemFree((CUdeviceptr)state->weights[i]);
    }
    cuMemFree(d_input_vec_i);
    cuMemFree(d_mid_res_i);
    cuMemFree(d_final_res_i);

    if (!inputs_to_gpu) {
        kava_free(inputs_to_gpu);
        inputs_to_gpu = 0;
    }
    if (!gpu_outputs) {
        kava_free(gpu_outputs);
        gpu_outputs = 0;
    }
}

void check_malloc(void *p, const char* error_str, int line) {
	if (p == NULL) PRINT("ERROR: Failed to allocate %s (line %d)\n", error_str, line);
}

//this takes one input array (with LEN_INPUT bytes) and
//expands it to N inputs, while converting to longs
void expand_input_n_times(char* input, int n) {
    int b, j;
	for(b = 0 ; b < n; b++) 
		for(j = 0; j < LEN_INPUT; j++)
			inputs_to_gpu[b*31 + j] =  (long) input[j];
}

void copy_input_to_shm(char* input, int n) {
    int j;
    for(j = 0; j < n * LEN_INPUT; j++)
	    inputs_to_gpu[j] =  (long) input[j];
}

//pass number of inputs, not bytes
void copy_inputs_to_gpu(u64 n_inputs) {
    cuMemcpyHtoDAsync(d_input_vec_i, inputs_to_gpu, sizeof(long) * LEN_INPUT * n_inputs, 0);
}

/*
 * Multi GPU, multi batch functions
*/

void multi_gpu_cuda_cleanup_dev(struct GPU_weights *state, int dev) {
    int i, batch;
    pr_warn("Cleaning up GPU %d state\n", dev);
    for(i = 0; i < 4 ; i++) {
        cuMemFree((CUdeviceptr)state->weights[i]);
    }

    for(batch = 0 ; batch < MAX_DEV_BATCHES ; batch++){
        pr_warn("Freeing for %d/%d\n", dev, batch);
        cuMemFree(multi_d_input_vec_i[dev][batch]);
        cuMemFree(multi_d_mid_res_i[dev][batch]);
        cuMemFree(multi_d_final_res_i[dev][batch]);

        // pr_warn("kava free %p\n", multi_inputs_to_gpu[dev][batch]);
        // kava_free(multi_inputs_to_gpu[dev][batch]);
        // pr_warn("kava free %p\n", multi_gpu_outputs[dev][batch]);
        // kava_free(multi_gpu_outputs[dev][batch]);
        // pr_warn("done\n");
    }
}

void multi_initialize_gpu(const char* cubin_path, int max_batch_size, int ndev) {
    int dev, batch;
    //intialize kernels
    if (cuctx) 
        return;

    gpu_cuda_init(0);
    gpu_get_cufunc(cubin_path, "_Z28prediction_final_layer_batchPlS_S_S_", &batch_linnos_final_layer_kernel);
    gpu_get_cufunc(cubin_path, "_Z26prediction_mid_layer_batchPlS_S_S_", &batch_linnos_mid_layer_kernel);
    
    for(dev = 0 ; dev < ndev ; dev++){
        for(batch = 0 ; batch < MAX_DEV_BATCHES ; batch++){
            pr_warn("Allocating for %d/%d\n", dev, batch);
            check_error(cuMemAlloc((CUdeviceptr*) &multi_d_input_vec_i[dev][batch], sizeof(long) * LEN_INPUT * max_batch_size), "cuMemAlloc ", __LINE__);
            check_error(cuMemAlloc((CUdeviceptr*) &multi_d_mid_res_i[dev][batch]  , sizeof(long) * LEN_LAYER_0 * max_batch_size), "cuMemAlloc ", __LINE__);
            check_error(cuMemAlloc((CUdeviceptr*) &multi_d_final_res_i[dev][batch], sizeof(long) * LEN_LAYER_1 * max_batch_size *32), "cuMemAlloc ", __LINE__);

            multi_inputs_to_gpu[dev][batch] = kava_alloc(LEN_INPUT * max_batch_size * sizeof(long));
            pr_warn("kava alloc %p\n", multi_inputs_to_gpu[dev][batch]);
            if (!multi_inputs_to_gpu[dev][batch]) 
                pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
            
            multi_gpu_outputs[dev][batch] = kava_alloc(64 * max_batch_size * sizeof(long));
            if (!multi_gpu_outputs[dev][batch]) 
                pr_warn("error allocating inputs_to_gpu:  %lu\n", LEN_INPUT * max_batch_size * sizeof(long));
            pr_warn("kava alloc %p\n", multi_gpu_outputs[dev][batch]);
        }
    }
}

void multi_copy_inputs_to_gpu(u64 n_inputs, int dev, int batch_id) {
    cuMemcpyHtoDAsync(multi_d_input_vec_i[dev][batch_id], multi_inputs_to_gpu[dev][batch_id], sizeof(long) * LEN_INPUT * n_inputs, 0);
}

void multi_copy_results_from_gpu(u64 n_inputs, int dev, int batch_id) {
    cuMemcpyDtoH(multi_gpu_outputs[dev][batch_id], multi_d_final_res_i[dev][batch_id], sizeof(long) * 64 * n_inputs);
}