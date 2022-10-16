#include "cuda.h"
#include "lake_shm.h"


struct GPU_state {
    CUdeviceptr d_weight_0_T_ent;
    CUdeviceptr d_weight_1_T_ent;
    CUdeviceptr d_bias_0_ent;
    CUdeviceptr d_bias_1_ent;
    CUdeviceptr d_input_vec_i;
    CUdeviceptr d_mid_res_i;
    CUdeviceptr d_final_res_i;
};


struct GPU_state default_state;

CUfunction batch_linnos_final_layer_kernel = 0;
CUfunction batch_linnos_mid_layer_kernel = 0;
CUcontext cuctx = 0;

long *inputs_to_gpu = 0;
long *gpu_outputs = 0;
