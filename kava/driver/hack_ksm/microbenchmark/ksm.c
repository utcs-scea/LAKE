#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/mman.h>
#include <linux/xxhash.h>
#include <cuda_kava.h>
#include <linux/delay.h>
#include <linux/ktime.h>

#include "helpers.h"
#include "xxhash.h"

const char *xxhash_function_name_v2 = "_Z7XXH32v2PviPjS0_jS0_";
const char *xxhash_function_name_v1 = "_Z5XXH32PvPj";

static char *xxhash_cubin_path = "xxhash.cubin";
module_param(xxhash_cubin_path, charp, 0444);
MODULE_PARM_DESC(xxhash_cubin_path, "The path to mllb.cubin, default ./xxhash.cubin");

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

CUdeviceptr d_page_buf;
CUdeviceptr d_checksum_buf;
CUdeviceptr d_seeds;
CUdeviceptr d_workspace;

const int kmax_batch = 256;

int ksm_gpu_init(void) {
	uint32_t seeds[4];
	printk("[kava_ksm] Initializing kava_ksm\n");
      
	cuInit(0);
	// Setup get device
	check_error(cuDeviceGet(&ctx.device, 0), "cuDeviceGet");
	check_error(cuCtxCreate(&ctx.context, 0, ctx.device), "cuCtxCreate");
	check_error(cuModuleLoad(&ctx.module, xxhash_cubin_path), "cuModuleLoad");
	printk("[kava_ksm] Loaded xxhash cubin\n");
	check_error(cuModuleGetFunction(&ctx.checksum_fn, ctx.module, xxhash_function_name_v2), "cuModuleGetFunction");
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
	check_error(cuMemAlloc(&d_checksum_buf, sizeof(uint32_t) * npages), "cuMemAlloc 2");
	check_error(cuMemAlloc(&d_workspace, sizeof(uint32_t) * 4 * npages), "cuMemAlloc 3");
}

void ksm_gpu_clean(void) {
	cuMemFree(d_page_buf);
	cuMemFree(d_checksum_buf);
	cuMemFree(d_workspace);
}

void ksm_gpu_setup_inputs(char* kpages, uint32_t npages) {
	check_error(cuMemcpyHtoD(d_page_buf, kpages, npages*PAGE_SIZE), "cuMemAlloc 1");
}

void ksm_gpu_run(uint32_t npages) {
	int threads = 128;
	uint32_t seed = 17;
	int blocks = npages*4 / 128;
	if (blocks == 0) blocks = 1;
	void *args[] = { &d_page_buf, &npages, &d_checksum_buf, &d_workspace, &seed, &d_seeds };
	cuLaunchKernel(ctx.checksum_fn, blocks, 1, 1, threads, 1, 1, 0, NULL, args, NULL);
}

#define USE_KSHM 1

static int run_gpu(void) {
	int i, j;
    //these are changeable
	int batch_sizes[] = {1,2,4,8,16,32,64,128,256,512, 1024};
    int n_batches = 9;
    const int max_batch = 256;
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
	if(USE_KSHM)
		h_page_buf  = (char*) kava_alloc(PAGE_SIZE * max_batch);
	else
		h_page_buf  = (char*) kmalloc(PAGE_SIZE * max_batch, GFP_KERNEL);
	if(h_page_buf == 0) {
		printk("h_page_buf alloc failed\n");
		return 0;
	}

	if(USE_KSHM)
		h_checksum_buf = (char*) kava_alloc(sizeof(uint32_t) * max_batch);
	else
		h_checksum_buf = (char*) kmalloc(sizeof(uint32_t) * max_batch, GFP_KERNEL);
	if(h_checksum_buf == 0) {
		printk("h_checksum_buf alloc failed\n");
		return 0;
	}

	comp_run_times  = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
	if(comp_run_times == 0) {
		printk("comp_run_times alloc failed\n");
		return 0;
	}

    total_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
	if(total_run_times == 0) {
		printk("total_run_times alloc failed\n");
		return 0;
	}

	for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];
		ksm_gpu_alloc(batch_size);

		//warmup
        ksm_gpu_setup_inputs(h_page_buf, batch_size);
		ksm_gpu_run(batch_size);
        usleep_range(1000, 2000);
        cuCtxSynchronize();

		for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();
            ksm_gpu_setup_inputs(h_page_buf, batch_size);
            c_start = ktime_get_ns();
            ksm_gpu_run(batch_size);
            c_stop = ktime_get_ns();
            //gpu_get_result(batch_size);
            t_stop = ktime_get_ns();

            comp_run_times[j] = (c_stop - c_start);
            total_run_times[j] = (t_stop - t_start);
            usleep_range(250, 1000);
        }

		avg = 0; 
		avg_total = 0;
		for (j = 0 ; j < RUNS ; j++) {
			avg_total += total_run_times[j];
			avg       += comp_run_times[j];
		}
		printk("GPU_%d, %lld, %lld\n", batch_size, avg/(RUNS*1000), avg_total/(RUNS*1000));
		ksm_gpu_clean();
	}

	if(USE_KSHM) {
		kava_free(h_page_buf);
		kava_free(h_checksum_buf);
	} else {
		kfree(h_page_buf);
		kfree(h_checksum_buf);
	}
	
	kfree(comp_run_times);
    kfree(total_run_times);
	cuMemFree(d_seeds);
	return 0;
}

static void ksm_cpu_run(char* buf, int npages) {
	int i;
	u32 checksum;
	for (i = 0 ; i < npages ; i++) {
		checksum = xxh32(buf+(i*PAGE_SIZE), PAGE_SIZE / 4, 17);
	}
}

static int run_cpu(void) {
    int i, j;
    //these are changeable
	int batch_sizes[] = {1,2,4,8,16,32,64,128,256,512, 1024};
    int n_batches = 9;
    const int max_batch = 256;
	int RUNS = 2;

    int batch_size;
	u64 t_start, t_stop;
    u64* total_run_times;
    u64 avg_total;

	char* h_page_buf;
	char* h_checksum_buf;

	//alloc on kernel
	h_page_buf  = (char*) kmalloc(PAGE_SIZE * max_batch, GFP_KERNEL);
	if(h_page_buf == 0) {
		printk("h_page_buf alloc failed\n");
		return 0;
	}

	h_checksum_buf = (char*) kmalloc(sizeof(uint32_t) * max_batch, GFP_KERNEL);
	if(h_checksum_buf == 0) {
		printk("h_checksum_buf alloc failed\n");
		return 0;
	}

    total_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
	if(total_run_times == 0) {
		printk("total_run_times alloc failed\n");
		return 0;
	}

	for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];

		//warmup
		ksm_cpu_run(h_page_buf, batch_size);
        usleep_range(1000, 2000);

		for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();
            ksm_cpu_run(h_page_buf, batch_size);
            t_stop = ktime_get_ns();

            total_run_times[j] = (t_stop - t_start);
            usleep_range(250, 1000);
        }

		avg_total = 0;
		for (j = 0 ; j < RUNS ; j++) {
			avg_total += total_run_times[j];
		}
		printk("CPU_%d, %lld\n", batch_size, avg_total/(RUNS*1000));
	}

	kfree(h_page_buf);
	kfree(h_checksum_buf);
    kfree(total_run_times);
	return 0;
}

/**
 * Program main
 */
static int __init ksm_init(void)
{
	run_gpu();
	run_cpu();
	return 0;
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



