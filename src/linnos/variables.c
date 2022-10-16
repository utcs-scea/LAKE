#include "cuda.h"
#include "lake_shm.h"

CUdeviceptr d_input_vec_i;
CUdeviceptr d_mid_res_i;
CUdeviceptr d_final_res_i;

CUfunction batch_linnos_final_layer_kernel = 0;
CUfunction batch_linnos_mid_layer_kernel = 0;
CUcontext cuctx = 0;

long *inputs_to_gpu = 0;
long *gpu_outputs = 0;
