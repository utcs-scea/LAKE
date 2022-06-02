#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/mman.h>

#include <linux/xxhash.h>
#include <cuda_kava.h>
#include <linux/delay.h>
#include <linux/ktime.h>

#include "helpers.h"

const char *xxhash_function_name = "_Z5XXH32PvPj";

static char *xxhash_cubin_path = "mllb.cubin";
module_param(xxhash_cubin_path, charp, 0444);
MODULE_PARM_DESC(xxhash_cubin_path, "The path to mllb.cubin, default ./mllb.cubin");

static const uint32_t PRIME32_1 = 0x9E3779B1U;   /* 0b10011110001101110111100110110001 */
static const uint32_t PRIME32_2 = 0x85EBCA77U;   /* 0b10000101111010111100101001110111 */

static inline void check_malloc(void *p, const char* error_str, int line) {
	if (p == NULL) printk(KERN_ERR "ERROR: Failed to allocate %s (line %d)\n", error_str, line);
}

struct kava_ksm_ctx {
	CUdevice device;
	CUcontext context;
	CUmodule module;
	CUfunction checksum_fn;
	CUstream stream;
};
static struct kava_ksm_ctx ctx;

// static uint32_t calc_checksum(struct page *page, int use_gpu) {
// 	// CUDA variables
// 	CUdeviceptr d_addr = d_single_page;
// 	CUdeviceptr d_checksum = d_single_checksum;
// 	uint32_t h_checksum;
// 	uint32_t checksum;
// 	int res;
// 	int block_x = 1;
// 	int grid_x = 1;
// 	void *args[] = { &d_addr, &d_checksum };
// 	void *addr = kmap_atomic(page);
// 	// Only run cuda functions if kava has been properly initialized
// 	if (use_gpu) {
// 		// Copy to device
// 		res = cuMemcpyHtoD(d_addr, addr, PAGE_SIZE);
// 		// Compute hash on device
// 		cuLaunchKernel(ctx.checksum_fn, grid_x, 1, 1, block_x, 1, 1, 0, NULL, args, NULL);
// 		// Copy to host
// 		cuMemcpyDtoH(&h_checksum, d_checksum, sizeof(uint32_t));
// 		// Set checksum to GPU checksum
// 		checksum = h_checksum;
// 	} else {
// 		// Calc checksum on CPU
// 		checksum = xxh32(addr, PAGE_SIZE / 4, 17);
// 	}
// 	kunmap_atomic(addr);
// 	return checksum;
// }

CUdeviceptr d_page_buf;
CUdeviceptr d_checksum_buf;
CUdeviceptr d_seeds;
CUdeviceptr d_workspace;

int ksm_gpu_init(void) {
	uint32_t seeds[4];
	printk("[kava_ksm] Initializing kava_ksm\n");
      
	cuInit(0);
	// Setup get device
	check_error(cuDeviceGet(&ctx.device, 0), "cuDeviceGet");
	check_error(cuCtxCreate(&ctx.context, 0, ctx.device), "cuCtxCreate");
	check_error(cuModuleLoad(&ctx.module, xxhash_cubin_path), "cuModuleLoad");
	printk("[kava_ksm] Loaded xxhash cubin\n");
	check_error(cuModuleGetFunction(&ctx.checksum_fn, ctx.module, xxhash_function_name), "cuModuleGetFunction");
	printk("[kava_ksm] loaded checksum function\n");

	check_error(cuMemAlloc(&d_seeds, sizeof(uint32_t) * 4), "cuMemAlloc d_seeds");
	seeds[0] = 17 + PRIME32_1 + PRIME32_2;
	seeds[1] = 17 + PRIME32_2;
	seeds[2] = 17 ;
	seeds[3] = 17 - PRIME32_1;
	check_error(cuMemcpyHtoD(d_seeds, seeds, 4*sizeof(uint32_t)), "cuMemcpyHtoD d_seeds");

	return 0;
}

void ksm_gpu_alloc(uint32_t npages) {
	check_error(cuMemAlloc(&d_page_buf, PAGE_SIZE * npages), "cuMemAlloc 1");
	printk("[kava_ksm] Allocated device page buffer at: %llx\n", d_page_buf);

	check_error(cuMemAlloc(&d_checksum_buf, sizeof(uint32_t) * npages), "cuMemAlloc 2");
	printk("[kava_ksm] Allocated device hash buffer at: %llx\n", d_checksum_buf);

	check_error(cuMemAlloc(&d_workspace, sizeof(uint32_t) * 4 * npages), "cuMemAlloc 3");
	printk("[kava_ksm] Allocated device hash buffer at: %llx\n", d_checksum_buf);
}

void ksm_gpu_clean(void) {
	cuMemFree(d_page_buf);
	cuMemFree(d_checksum_buf);
	cuMemFree(d_workspace);
}

void ksm_gpu_setup_inputs(char* kpages, uint32_t npages) {
	check_error(cuMemcpyHtoD(d_page_buf, kpages, npages*PAGE_SIZE), "cuMemAlloc 1");
}

void ksm_gpu_run(uint32_t batch_size) {
	int threads = 128;
	uint32_t seed = 17;
	int blocks = batch_size*4 / 128;
	if (blocks == 0) blocks = 1;

	void *args[] = { &d_page_buf, &d_checksum_buf, &d_workspace, &seed, &d_seeds };

	cuLaunchKernel(ctx.checksum_fn, blocks, 1, 1, threads, 1, 1, 0, NULL, args, NULL);
}

static int run_gpu(void) {
	int i, j;
    //these are changeable
    int batch_sizes[] = {128};
    //int batch_sizes[] = {1,2,4,8,16,32,64, 128, 256, 512,1024};
    int n_batches = 1;
    const int max_batch = 128;
	int RUNS = 2;

    int batch_size;
	u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;

	char* h_page_buf;
	char* h_checksum_buf;

	ksm_gpu_init();

	//alloc on kernel
	h_page_buf     = (char*) kava_alloc(PAGE_SIZE * max_batch);
	h_checksum_buf = (char*) kava_alloc(sizeof(uint32_t) * max_batch);
	comp_run_times  = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
    total_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);

	for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];
		ksm_gpu_alloc(batch_size);

		//warmup
        ksm_gpu_setup_inputs(h_page_buf, batch_size);
		ksm_gpu_run(max_batch);
        usleep_range(1000, 2000);
        cuCtxSynchronize();

		// for (j = 0 ; j < RUNS ; j++) {
        //     t_start = ktime_get_ns();
        //     ksm_gpu_setup_inputs(h_page_buf, batch_size);
        //     c_start = ktime_get_ns();
        //     ksm_gpu_run(batch_size);
        //     c_stop = ktime_get_ns();
        //     //gpu_get_result(batch_size);
        //     t_stop = ktime_get_ns();

        //     comp_run_times[j] = (c_stop - c_start);
        //     total_run_times[j] = (t_stop - t_start);
        //     usleep_range(250, 1000);
        // }

		avg = 0; avg_total = 0;
        for (j = 0 ; j < RUNS ; j++) {
            avg += comp_run_times[j];
            avg_total += total_run_times[j];
        }
        avg = avg / (1000*RUNS); avg_total = avg_total / (1000*RUNS);
	
		printk(V_INFO, "GPU batch_%d, %lld, %lld\n", batch_size, avg, avg_total);
        ksm_gpu_clean();
	}

	kava_free(h_page_buf);
	kava_free(h_checksum_buf);
	kfree(comp_run_times);
    kfree(total_run_times);
	cuMemFree(d_seeds);

	return 0;
}

static int run_cpu(void) {
    return 0;
}

/**
 * Program main
 */
static int __init ksm_init(void)
{
	return run_gpu();
}

static void __exit ksm_fini(void)
{
    //cleanup
}

module_init(ksm_init);
module_exit(ksm_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Kernel module of a ksm program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");



