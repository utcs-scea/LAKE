#ifndef ROUTER_H
#define ROUTER_H

// System includes
#include <stdio.h>
#include <assert.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netinet/tcp.h>

// CUDA driver
#include <cuda.h>

// Helper functions and utilities to work with CUDA
#include "helper_string.h"

// Packet collector
#include "packet-collector.h"

#define SWAP(a,b,type) { type tmp=(a); (a)=(b); (b)=tmp; }

#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) fprintf(stdout, __VA_ARGS__); } while (0)


// Change this define to determine which processing function is used
// (e.g., firewall, longest prefix match, etc.)
#define FIREWALL

//#define PINNED_PACKET_MEMORY  // packet arrays are pinned and mapped
//#define PINNED_MEMORY  // results arrays are pinned and mapped

#ifdef __cplusplus
extern "C" {
#endif

extern CUdeviceptr d_rules;
extern int h_num_rules;

void setup_gpu();
void teardown();
void process_packets_sequential(packet *p, int *results, int num_packets);
void setup_sequential();

#if defined FIREWALL
int set_num_rules(int s);
int get_num_rules();
#endif /* FIREWALL */

#ifdef __cplusplus
}
#endif

__global__ void process_packets(packet *p, int *results, int num_packets, int block_size, void *rules, int num_rules);

/**
 * Checks the supplied cuda error for failure
 */
inline CUresult check_error(CUresult error, const char* error_str, int line)
{
	if (error != CUDA_SUCCESS) {
        const char *error_str;
        cuGetErrorString(error, &error_str);
		fprintf(stderr, "ERROR: %s returned error (line %d): %s\n", error_str, line, error_str);
		exit(EXIT_FAILURE);
	}
	return error;
}

/**
 * Checks that the supplied pointer is not NULL
 */
inline void check_malloc(void *p, const char* error_str, int line)
{
	if (p == NULL) {
		fprintf(stderr, "ERROR: Failed to allocate %s (line %d)\n", error_str, line);
		exit(EXIT_FAILURE);
	}
}

#endif /* ROUTER_H */
