#ifdef __KERNEL__
#include "cuda.h"
#include "lake_shm.h"
#else
#include <cuda.h>
#include <stdio.h>
#endif

CUdeviceptr d_weight_0_T_ent;
CUdeviceptr d_weight_1_T_ent;
CUdeviceptr d_bias_0_ent;
CUdeviceptr d_bias_1_ent;
CUdeviceptr d_input_vec_i;
CUdeviceptr d_mid_res_i;
CUdeviceptr d_final_res_i;
CUfunction batch_linnos_final_layer_kernel;
CUfunction batch_linnos_mid_layer_kernel;
CUcontext cuctx;