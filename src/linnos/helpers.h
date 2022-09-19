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

#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)

#else
#include <cuda.h>
#include <stdio.h>
#endif

static inline CUresult check_error(CUresult error, const char* error_str, int line)
{
	if (error != CUDA_SUCCESS) {
        #ifdef __KERNEL__
        const char *error_str;
        cuGetErrorString(error, &error_str);
		printk("ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
        if (error_str)
            kfree(error_str);
        #else
        printf("ERROR: %d returned error (line %d)\n", error, line);
        #endif
	}
	return error;
}

void gpu_init(int dev, CUcontext* cuctx);
void gpu_get_cufunc(char* cubin, char* kname, CUfunction *func);

#endif