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
#else
#include <cuda.h>
#include <stdio.h>
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

void initialize_gpu(const char* cubin_path, long **weights, int n_vecs);
void gpu_cuda_cleanup(void);
void check_malloc(void *p, const char* error_str, int line);

#endif