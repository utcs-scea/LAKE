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
// CUDA driver
#include "cuda.h"
#include "lake_shm.h"

#define LLU "%llu"
#define LLD "%lld"

#else
#include <stdint.h>
#include <sys/time.h>
#define u64 uint64_t
#define usleep_range(X,Y) sleep(X/1000000)
#define LLU "%lu"
#define LLD "%ld"
#define vmalloc(X) malloc(X)
#define vfree(X) free((void*)X)

static inline u64 get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, 0);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}
#define ktime_get_ns() get_tsns()
#define kernel_fpu_begin() (void)0
#define kernel_fpu_end() (void)0

#include <cuda.h>
#include <stdio.h>
#include <string.h>
#endif

#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO

#ifdef __KERNEL__
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)
#else
#define PRINT(verbosity, ...) do { if (1) printf(__VA_ARGS__); } while (0)
#define kava_alloc(...) malloc(__VA_ARGS__)
#define kava_free(...) free(__VA_ARGS__)
#endif

static inline CUresult check_error(CUresult error, const char* error_str, int line)
{
	if (error != CUDA_SUCCESS) {
        #ifdef __KERNEL__
        printk(KERN_ERR "ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
        #else
        printf("ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
        #endif
	}
	return error;
}

void gpu_init(int dev, CUcontext* cuctx);
void gpu_get_cufunc(char* cubin, char* kname, CUfunction *func);
void gpu_setup(int n_inputs, CUdeviceptr *d_inputs, CUdeviceptr *d_w1, CUdeviceptr *d_b1, CUdeviceptr *d_w2, CUdeviceptr *d_results);
void gpu_clean(CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, CUdeviceptr d_results);
void gpu_setup_inputs(CUdeviceptr d_inputs, int* inputs, int n);
//float gpu_inference();
int gpu_inference_many(CUfunction* cufunc, int n_inputs,
        CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, float b2, CUdeviceptr d_results, int sync);
int gpu_get_result(int n_inputs, CUdeviceptr d_results, float* outs);

#endif