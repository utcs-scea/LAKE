#ifndef HELLO_H
#define HELLO_H

// System includes
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

// CUDA driver
#include "cuda.h"
#include "lake_shm.h"

#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)


static inline CUresult check_error(CUresult error, const char* error_str, int line)
{
	if (error != CUDA_SUCCESS) {
		printk(KERN_ERR "ERROR: returned error (line %d): %s\n", line, error_str);
	}
	return error;
}

static inline void check_malloc(void *p, const char* error_str, int line)
{
	if (p == NULL) {
		printk(KERN_ERR "ERROR: Failed to allocate %s (line %d)\n", error_str, line);
	}
}


#endif
