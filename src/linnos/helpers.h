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

#ifndef __MLLB_HELPERS_H
#define __MLLB_HELPERS_H

#ifdef __KERNEL__
// System includes
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/slab.h>

#include "variables.h"

// CUDA driver
#include "cuda.h"
#include "lake_shm.h"
#else
#include <cuda.h>
#include <stdio.h>
#include<stdint.h>
#define u64 uint64_t
#endif


#ifdef __KERNEL__
#define PRINT(...) do { if (1) printk(KERN_INFO __VA_ARGS__); } while (0)
#else
#define PRINT(...) do { if (1) printf(__VA_ARGS__); } while (0)
#define kava_alloc(...) malloc(__VA_ARGS__)
#define kava_free(...) free(__VA_ARGS__)
#endif

static inline CUresult check_error(CUresult error, const char* error_str, int line)
{
    if (error != CUDA_SUCCESS) {
        PRINT("ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
	}
	return error;
}

void copy_weights(long **weights, struct GPU_weights *state);
void initialize_gpu(const char* cubin_path, int max_batch_size);
void gpu_cuda_cleanup(struct GPU_weights *state);

void check_malloc(void *p, const char* error_str, int line);
void expand_input_n_times(char* input, int n);
void copy_input_to_shm(char* input, int n);
void copy_inputs_to_gpu(u64 n_inputs);
void copy_results_from_gpu(u64 n_inputs);


void multi_gpu_cuda_cleanup(struct GPU_weights *state, int dev);
void multi_initialize_gpu(const char* cubin_path, int max_batch_size, int ndev);
void multi_copy_inputs_to_gpu(u64 n_inputs, int dev, int batch_id);
void multi_copy_results_from_gpu(u64 n_inputs, int dev, int batch_id);
void multi_gpu_cuda_cleanup_dev(struct GPU_weights *state, int dev);

#endif