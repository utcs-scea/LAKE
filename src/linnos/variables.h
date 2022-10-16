#ifdef __KERNEL__
#include "cuda.h"
#else
#include <cuda.h>
#include <stdio.h>
#endif

struct GPU_state {
    //CUdeviceptr d_weight_0_T_ent;
    //CUdeviceptr d_weight_1_T_ent;
    //CUdeviceptr d_bias_0_ent;
    //CUdeviceptr d_bias_1_ent;
    CUdeviceptr weights[4];
    CUdeviceptr d_input_vec_i;
    CUdeviceptr d_mid_res_i;
    CUdeviceptr d_final_res_i;
};

extern struct GPU_state default_state;

extern CUfunction batch_linnos_final_layer_kernel;
extern CUfunction batch_linnos_mid_layer_kernel;
extern CUcontext cuctx;
extern long *inputs_to_gpu;
extern long *gpu_outputs;