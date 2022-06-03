/*
 * Implementing Breadth first search on CUDA using algorithm given in HiPC'07
 * paper "Accelerating Large Graph Algorithms on the GPU using CUDA"
 *
 * Copyright (c) 2008 
 * International Institute of Information Technology - Hyderabad. 
 * All rights reserved.
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for educational purpose is hereby granted without fee, 
 * provided that the above copyright notice and this permission notice 
 * appear in all copies of this software and that you do not sell the software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, 
 * IMPLIED OR OTHERWISE.
 *
 * Created by Pawan Harish.
 *
 * Modified by Shinpei Kato.
 *
 * Ported to kernel by Hangchen Yu.
 */
#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "cuda_kava.h"

#include "../util/util.h"
#include "bfs.h"

static char *input_p = "";
static char *input_f = "";
static char *cubin_p = "";
module_param(input_p, charp, 0000);
MODULE_PARM_DESC(input_p, "Input graph path");
module_param(input_f, charp, 0000);
MODULE_PARM_DESC(input_f, "Input graph filename");
module_param(cubin_p, charp, 0000);
MODULE_PARM_DESC(cubin_p, "CUDA binary path");

int no_of_nodes;
int edge_list_size;
struct file *fp;

struct timestamp ts_init, ts_total, ts_memalloc, ts_h2d, ts_d2h, ts_kernel, ts_close;
long init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0, d2h_phy_time = 0, h2d_phy_time = 0;

int bfs_launch
(CUmodule mod, int nr_blocks, int nr_threads_per_block, int nr_nodes,
 CUdeviceptr d_over, CUdeviceptr d_graph_nodes, CUdeviceptr d_graph_edges, 
 CUdeviceptr d_graph_mask, CUdeviceptr d_updating_graph_mask, 
 CUdeviceptr d_graph_visited, CUdeviceptr d_cost)
{
	int bdx, bdy, gdx, gdy;
	int k = 0;
	int stop;
	CUfunction f1, f2;
	CUresult res;

	bdx = nr_threads_per_block;
	bdy = 1;
	gdx = nr_blocks;
	gdy = 1;

    probe_time_start(&ts_kernel);

	/* get functions. */
	res = cuModuleGetFunction(&f1, mod, "_Z6KernelP4NodePiS1_S1_S1_S1_i");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(f1) failed: res = %u\n", res);
		return -1;
	}
	res = cuModuleGetFunction(&f2, mod, "_Z7Kernel2PiS_S_S_i");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(f2) failed: res = %u\n", res);
		return -1;
	}

    kernel_time = probe_time_end(&ts_kernel);

	/* Call the Kernel untill all the elements of Frontier are not false */
	do {
		/* if no thread changes this value then the loop stops */
        probe_time_start(&ts_h2d);
		stop = false;
		res = cuMemcpyHtoD(d_over, &stop, sizeof(int));
		if (res != CUDA_SUCCESS) {
			pr_err("cuMemcpyHtoD(d_over) failed\n");
			return -1;
		}
        h2d_time += probe_time_end(&ts_h2d);

        probe_time_start(&ts_kernel);
	    /* f1 */
        {
            void *param1[] = {&d_graph_nodes, &d_graph_edges, &d_graph_mask,
                              &d_updating_graph_mask, &d_graph_visited, &d_cost,
                              &nr_nodes, NULL};
            res = cuLaunchKernel(f1, gdx, gdy, 1, bdx, bdy, 1, 0, 0, 
                                 (void**)param1, NULL);
            if (res != CUDA_SUCCESS) {
                pr_err("cuLaunchKernel(f1) failed: res = %u\n", res);
                return -1;
            }
            // cuCtxSynchronize();
            /* check if kernel execution generated and error */
        }

		/* f2 */
        {
            void *param2[] = {&d_graph_mask, &d_updating_graph_mask, 
                              &d_graph_visited, &d_over,  &nr_nodes, NULL};
            res = cuLaunchKernel(f2, gdx, gdy, 1, bdx, bdy, 1, 0, 0, 
                                 (void**)param2, NULL);
            if (res != CUDA_SUCCESS) {
                pr_err("cuLaunchKernel(f2) failed: res = %u\n", res);
                return -1;
            }
            cuCtxSynchronize();
            /* check if kernel execution generated and error */
        }
        kernel_time += probe_time_end(&ts_kernel);

        probe_time_start(&ts_d2h);
		res = cuMemcpyDtoH(&stop, d_over, sizeof(int));
		if (res != CUDA_SUCCESS) {
			pr_err("cuMemcpyDtoH(stop) failed: res = %u\n", res);
			return -1;
		}
        d2h_time += probe_time_end(&ts_d2h);

		// cuCtxSynchronize();
		k++;
	} while (stop);

	return 0;
}

void Usage(void)
{
	pr_err("Usage: insmod bfs.ko input_p=<path> input_f=<file.bin> cubin_p=<path>\n");
}

int BFSGraph(void)
{
    char input_fn[128];
    char cubin_fn[128];
    loff_t pos = 0;
	int source = 0;
	int num_of_blocks = 1;
	int num_of_threads_per_block = no_of_nodes;
	struct Node *h_graph_nodes;
	int *h_graph_mask;
	int *h_updating_graph_mask;
	int *h_graph_visited;
	int *h_graph_edges;
	int *h_cost;
	int start, edgeno;
	int id, cost;
	int i;
	CUdeviceptr d_graph_nodes;
	CUdeviceptr d_graph_edges;
	CUdeviceptr d_graph_mask;
	CUdeviceptr d_updating_graph_mask;
	CUdeviceptr d_graph_visited;
	CUdeviceptr d_cost;
	CUdeviceptr d_over;
	CUcontext ctx;
	CUmodule mod;
	CUresult res;

	if (strlen(input_f) <= 0 || strlen(input_p) <= 0 || strlen(cubin_p) <= 0) {
		Usage();
		return 0;
	}

    strcpy(input_fn, input_p);
    strcat(input_fn, "/");
    strcat(input_fn, input_f);
    pr_info("input_fn = %s\n", input_fn);
	fp = filp_open(input_fn, O_RDONLY, 0);
	if (!fp) {
		pr_err("Error Reading graph file %s\n", input_fn);
		return -1;
	}

    kernel_read(fp, (char *)&no_of_nodes, sizeof(int), &pos);

	/* Make execution Parameters according to the number of nodes and 
	   distribute threads across multiple Blocks if necessary */
	if (no_of_nodes > MAX_THREADS_PER_BLOCK) {
		num_of_blocks = (int)DIV_ROUND_UP(no_of_nodes, MAX_THREADS_PER_BLOCK);
		num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
	}

	/* allocate host memory */
	h_graph_nodes = (struct Node*) vmalloc(sizeof(struct Node) * no_of_nodes);
	h_graph_mask = (int*) vmalloc(sizeof(int) * no_of_nodes);
	h_updating_graph_mask = (int*) vmalloc(sizeof(int) * no_of_nodes);
	h_graph_visited = (int*) vmalloc(sizeof(int) * no_of_nodes);

	/* initalize the memory */
	for(i = 0; i < no_of_nodes; i++)  {
        kernel_read(fp, (char *)&start, sizeof(int), &pos);
        kernel_read(fp, (char *)&edgeno, sizeof(int), &pos);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_updating_graph_mask[i]=false;
		h_graph_visited[i]=false;
	}

	/* read the source node from the file */
    kernel_read(fp, (char *)&source, sizeof(int), &pos);
	source = 0;

	/* set the source node as true in the mask */
	h_graph_mask[source] = true;
	h_graph_visited[source] = true;

    kernel_read(fp, (char *)&edge_list_size, sizeof(int), &pos);

	h_graph_edges = (int*) vmalloc(sizeof(int) * edge_list_size);
	for(i = 0; i < edge_list_size ; i++) {
        kernel_read(fp, (char *)&id, sizeof(int), &pos);
        kernel_read(fp, (char *)&cost, sizeof(int), &pos);
		h_graph_edges[i] = id;
	}

	if (fp) {
        filp_close(fp, NULL);
    }

	/* allocate mem for the result on host side */
	h_cost = (int*) vmalloc(sizeof(int) * no_of_nodes);
	for(i = 0; i < no_of_nodes; i++)
		h_cost[i] = -1;
	h_cost[source] = 0;

	/*
	 * call our common CUDA initialization utility function.
	 */
    probe_time_start(&ts_total);
    probe_time_start(&ts_init);

    strcpy(cubin_fn, cubin_p);
    strcat(cubin_fn, "/bfs.cubin");
    pr_info("cubin_fn = %s\n", cubin_fn);
	res = cuda_driver_api_init(&ctx, &mod, cubin_fn);
	if (res != CUDA_SUCCESS) {
		pr_err("cuda_driver_api_init failed: res = %u\n", res);
		return -1;
	}

    init_time = probe_time_end(&ts_init);

	/*
	 * allocate device memory space
	 */
    probe_time_start(&ts_memalloc);

	res = cuMemAlloc(&d_graph_nodes, sizeof(struct Node) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_graph_edges, sizeof(int) * edge_list_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_graph_mask, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_updating_graph_mask, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_graph_visited, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	res = cuMemAlloc(&d_cost, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}
	/* make a int to check if the execution is over */
	res = cuMemAlloc(&d_over, sizeof(int));
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed: res = %u\n", res);
		return -1;
	}

    mem_alloc_time = probe_time_end(&ts_memalloc);
    probe_time_start(&ts_h2d);

	/* copy the node list to device memory */
	res = cuMemcpyHtoD(d_graph_nodes, h_graph_nodes, sizeof(struct Node) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	/* copy the edge List to device memory */
	res = cuMemcpyHtoD(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	/* copy the mask to device memory */
	res = cuMemcpyHtoD(d_graph_mask, h_graph_mask, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	res = cuMemcpyHtoD(d_updating_graph_mask, h_updating_graph_mask, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	/* copy the visited nodes array to device memory */
	res = cuMemcpyHtoD(d_graph_visited, h_graph_visited, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}
	/* device memory for result */
	res = cuMemcpyHtoD(d_cost, h_cost, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD failed: res = %u\n", res);
		return -1;
	}

    h2d_time = probe_time_end(&ts_h2d);

	/* we cannot use maximum thread number because of the virtualization
 	 * limitation in a3.h */
	bfs_launch(mod, num_of_blocks, num_of_threads_per_block / 4, no_of_nodes,
			   d_over, d_graph_nodes, d_graph_edges, d_graph_mask, 
			   d_updating_graph_mask, d_graph_visited, d_cost);

	/* copy result from device to host */
    probe_time_start(&ts_d2h);

	res = cuMemcpyDtoH(h_cost, d_cost, sizeof(int) * no_of_nodes);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyDtoH failed: res = %u\n", res);
		return -1;
	}

    d2h_time += probe_time_end(&ts_d2h);
    probe_time_start(&ts_close);

	/* cleanup memory */
	cuMemFree(d_graph_nodes);
	cuMemFree(d_graph_edges);
	cuMemFree(d_graph_mask);
	cuMemFree(d_updating_graph_mask);
	cuMemFree(d_graph_visited);
	cuMemFree(d_cost);
	cuMemFree(d_over);

	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		pr_err("cuda_driver_api_exit faild: res = %u\n", res);
		return -1;
	}

    close_time = probe_time_end(&ts_close);
	total_time = probe_time_end(&ts_total);

	/* Store the result into a file */
#if 0
	{
		FILE *fpo = fopen("result.txt", "w");
		for(i = 0; i < no_of_nodes; i++)
			fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
		fclose(fpo);
		pr_info("/* Result stored in result.txt */\n");
	}
#endif

    pr_info("Init: %ld\n", init_time);
	pr_info("MemAlloc: %ld\n", mem_alloc_time);
	pr_info("HtoD: %ld\n", h2d_time);
	pr_info("Exec: %ld\n", kernel_time);
	pr_info("DtoH: %ld\n", d2h_time);
	pr_info("Close: %ld\n", close_time);
	pr_info("API: %ld\n", init_time+mem_alloc_time+h2d_time+kernel_time+d2h_time+close_time);
	pr_info("Total: %ld (ns)\n", total_time);

	vfree(h_graph_nodes);
	vfree(h_graph_edges);
	vfree(h_graph_mask);
	vfree(h_updating_graph_mask);
	vfree(h_graph_visited);
	vfree(h_cost);

	return 0;
}

static int __init drv_init(void)
{
    pr_info("load cuda bfs module\n");

	no_of_nodes = 0;
	edge_list_size = 0;
	BFSGraph();

	return 0;
}

static void __exit drv_fini(void)
{
    pr_info("unload cuda bfs module\n");
}

module_init(drv_init);
module_exit(drv_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Driver module for Rodinia bfs");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
