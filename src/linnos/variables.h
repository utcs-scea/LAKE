#ifndef __LINNOS_VARS_H
#define __LINNOS_VARS_H

#ifdef __KERNEL__
#include "cuda.h"
#else
#include <cuda.h>
#include <stdio.h>
#endif

#define NUMBER_DEVICES 3
#define MAX_DEV_BATCHES 8

struct GPU_weights {
    //CUdeviceptr d_weight_0_T_ent;
    //CUdeviceptr d_weight_1_T_ent;
    //CUdeviceptr d_bias_0_ent;
    //CUdeviceptr d_bias_1_ent;
    long *weights[4];
};

extern CUdeviceptr d_input_vec_i;
extern CUdeviceptr d_mid_res_i;
extern CUdeviceptr d_final_res_i;

extern CUfunction batch_linnos_final_layer_kernel;
extern CUfunction batch_linnos_mid_layer_kernel;
extern CUcontext cuctx;
extern long *inputs_to_gpu;
extern long *gpu_outputs;


extern long *multi_inputs_to_gpu[NUMBER_DEVICES][MAX_DEV_BATCHES];
extern long *multi_gpu_outputs[NUMBER_DEVICES][MAX_DEV_BATCHES];

extern CUdeviceptr multi_d_input_vec_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
extern CUdeviceptr multi_d_mid_res_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
extern CUdeviceptr multi_d_final_res_i[NUMBER_DEVICES][MAX_DEV_BATCHES];

extern long *first_weight_ptr_to_dev[NUMBER_DEVICES];

extern CUstream cu_streams[NUMBER_DEVICES][MAX_DEV_BATCHES];

#endif