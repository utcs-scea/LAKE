#ifndef ROUTER_H
#define ROUTER_H

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
#include "shared_memory.h"

// Packet collector
#include "packet-collector.h"

#define SWAP(a,b,type) { type tmp=(a); (a)=(b); (b)=tmp; }

#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)


// Change this define to determine which processing function is used
// (e.g., firewall, longest prefix match, etc.)
#define FIREWALL

//#define PINNED_PACKET_MEMORY  // packet arrays are pinned and mapped
//#define PINNED_MEMORY  // results arrays are pinned and mapped

extern CUdeviceptr d_rules;
extern int h_num_rules;

void setup_gpu(void);
void teardown(void);
void process_packets_sequential(packet *p, int *results, int num_packets);
void setup_sequential(void);

#if defined FIREWALL
int set_num_rules(int s);
int get_num_rules(void);
#endif /* FIREWALL */

/**
 * Checks the supplied cuda error for failure
 */
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

/**
 * Checks that the supplied pointer is not NULL
 */
static inline void check_malloc(void *p, const char* error_str, int line)
{
	if (p == NULL) {
		printk(KERN_ERR "ERROR: Failed to allocate %s (line %d)\n", error_str, line);
	}
}

// A super naive double to int conversion function that only works on small positive integers.
// x is double (8 bytes)
static inline int64_t naive_dtoi(int64_t x) {
    // uint64_t u = (union { double d; uint64_t u; }) { x }.u;
    uint64_t u;
    memcpy(&u, &x, sizeof(uint64_t));
    return ((u & 0xfffffffffffff) | (1ULL << 52)) >> (1075 - ((u >> 52) & 0x7ff));
}

#endif /* ROUTER_H */
