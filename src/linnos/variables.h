/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

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
    //then 2 for +1 wt,bias
    //then 2 for +2 wt,bias
    long *weights[8];
};

extern CUdeviceptr d_input_vec_i;
extern CUdeviceptr d_mid_res_i;
extern CUdeviceptr d_mid_res_1_i;
extern CUdeviceptr d_mid_res_2_i;
extern CUdeviceptr d_final_res_i;

extern CUfunction batch_linnos_final_layer_kernel;
extern CUfunction batch_linnos_mid_layer_kernel;
extern CUfunction batch_linnos_mid_layer_1_kernel;
extern CUfunction batch_linnos_mid_layer_2_kernel;
extern CUcontext cuctx;
extern long *inputs_to_gpu;
extern long *gpu_outputs;

extern s64 window_size_ns;
extern u32 max_batch_size; 
extern u32 cpu_gpu_threshold;
extern u64 inter_arrival_threshold ;

extern long *multi_inputs_to_gpu[NUMBER_DEVICES][MAX_DEV_BATCHES];
extern long *multi_gpu_outputs[NUMBER_DEVICES][MAX_DEV_BATCHES];

extern CUdeviceptr multi_d_input_vec_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
extern CUdeviceptr multi_d_mid_res_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
extern CUdeviceptr multi_d_mid_res_1_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
extern CUdeviceptr multi_d_mid_res_2_i[NUMBER_DEVICES][MAX_DEV_BATCHES];
extern CUdeviceptr multi_d_final_res_i[NUMBER_DEVICES][MAX_DEV_BATCHES];

extern long *first_weight_ptr_to_dev[NUMBER_DEVICES];

extern CUstream cu_streams[NUMBER_DEVICES][MAX_DEV_BATCHES];

#endif