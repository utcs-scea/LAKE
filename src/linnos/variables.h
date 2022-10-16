#ifndef __LINNOS_VARS_H
#define __LINNOS_VARS_H

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
    long *cast_weights[4];
};

extern CUdeviceptr d_input_vec_i;
extern CUdeviceptr d_mid_res_i;
extern CUdeviceptr d_final_res_i;

extern CUfunction batch_linnos_final_layer_kernel;
extern CUfunction batch_linnos_mid_layer_kernel;
extern CUcontext cuctx;
extern long *inputs_to_gpu;
extern long *gpu_outputs;


#endif