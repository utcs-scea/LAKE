#ifndef __MLLB_HELPERS_H
#define __MLLB_HELPERS_H

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

// CUDA driver
#include "cuda.h"
#include "shared_memory.h"

#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)

static inline CUresult check_error(CUresult error, const char* error_str, int line)
{
	if (error != CUDA_SUCCESS) {
        const char *error_str;
        cuGetErrorString(error, &error_str);
		printk(KERN_ERR "ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
        if (error_str)
            kfree(error_str);
	}
	return error;
}

void gpu_init(int dev, CUcontext* cuctx);
void gpu_get_cufunc(char* cubin, char* kname, CUfunction *func);
void gpu_setup(int n_inputs, CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, CUdeviceptr d_results);
void gpu_clean(CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, CUdeviceptr d_results);
void gpu_setup_inputs(CUdeviceptr d_inputs, int* inputs, int n);
//float gpu_inference();
int gpu_inference_many(CUfunction* cufunc, int n_inputs,
        CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, float b2, CUdeviceptr d_results);
int gpu_get_result(int n_inputs);

#endif