#ifdef __KERNEL__
#include "cuda.h"
#include "lake_shm.h"
#else
#include <cuda.h>
#include <stdio.h>
#endif

extern CUdeviceptr d_weight_0_T_ent;
extern CUdeviceptr d_weight_1_T_ent;
extern CUdeviceptr d_bias_0_ent;
extern CUdeviceptr d_bias_1_ent;
extern CUdeviceptr d_input_vec_i;
extern CUdeviceptr d_mid_res_i;
extern CUdeviceptr d_final_res_i;
extern CUfunction batch_linnos_final_layer_kernel;
extern CUfunction batch_linnos_mid_layer_kernel;

