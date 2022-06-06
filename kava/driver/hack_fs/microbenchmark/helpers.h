#ifndef __MLLB_HELPERS_H
#define __MLLB_HELPERS_H

#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/slab.h>

#include "cuda.h"
#include "shared_memory.h"

#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)

static inline CUresult check_error(CUresult error, const char* error_str)
{
	if (error != CUDA_SUCCESS) {
        const char *error_str;
        cuGetErrorString(error, &error_str);
		printk(KERN_ERR "ERROR: %s returned error: %s\n", error_str, error_str);
        if (error_str)
            kfree(error_str);
	}
	return error;
}


#endif