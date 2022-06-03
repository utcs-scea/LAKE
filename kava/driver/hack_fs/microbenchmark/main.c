#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/mman.h>
#include <linux/xxhash.h>
#include <cuda_kava.h>
#include <linux/delay.h>
#include <linux/ktime.h>

#include <crypto/skcipher.h>
#include <linux/scatterlist.h>

#include "helpers.h"


const char *ecb_enc_function_name = "_Z7XXH32v2PviPjS0_jS0_";
const char *ecb_dec_function_name = "_Z7XXH32v2PviPjS0_jS0_";

static char *cubin_path = ".cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to .cubin");

static inline void check_malloc(void *p, const char* error_str, int line) {
	if (p == NULL) printk(KERN_ERR "ERROR: Failed to allocate %s (line %d)\n", error_str, line);
}

CUdevice device;
CUcontext context;
CUmodule module;
CUfunction enc_fn;
CUfunction dec_fn;
CUstream stream;

CUdeviceptr d_page_buf;
CUdeviceptr d_page_out;
CUdeviceptr d_rk;

int gpu_init(void) {
	// cuInit(0);
	// // Setup get device
	// check_error(cuDeviceGet(&device, 0), "cuDeviceGet");
	// check_error(cuCtxCreate(&context, 0, device), "cuCtxCreate");
	// check_error(cuModuleLoad(&module, cubin_path), "cuModuleLoad");
	// check_error(cuModuleGetFunction(&enc_fn, module, ecb_enc_function_name), "cuModuleGetFunction");
	// check_error(cuModuleGetFunction(&dec_fn, module, ecb_dec_function_name), "cuModuleGetFunction");

	return 0;
}

void gpu_alloc(uint32_t npages) {
	//check_error(cuMemAlloc(&d_page_buf, PAGE_SIZE * npages), "cuMemAlloc 1");
	//check_error(cuMemAlloc(&d_checksum_buf, sizeof(uint32_t) * npages), "cuMemAlloc 2");
	//check_error(cuMemAlloc(&d_workspace, sizeof(uint32_t) * 4 * npages), "cuMemAlloc 3");
}

void gpu_clean(void) {
	//cuMemFree(d_page_buf);
	//cuMemFree(d_checksum_buf);
	//cuMemFree(d_workspace);
}

void gpu_setup_inputs(char* kpages, uint32_t npages) {
	//check_error(cuMemcpyHtoD(d_page_buf, kpages, npages*PAGE_SIZE), "cuMemcpyHtoD");
}

void gpu_output(char* out, uint32_t npages) {
	//check_error(cuMemcpyDtoH(out, d_checksum_buf, npages*sizeof(uint32_t)), "cuMemcpyHtoD");
}

void gpu_run(uint32_t npages) {	
	// int threads, blocks;
	// if (npages <= 128/4) {
	// 	threads = npages*4;
	// 	blocks = 1;
	// } else {
	// 	threads = 128;
	// 	blocks = npages*4 / 128;
	// }

	// uint32_t seed = 17;
	// if (blocks == 0) blocks = 1;
	// void *args[] = { &d_page_buf, &npages, &d_checksum_buf, &d_workspace, &seed, &d_seeds };
	// cuLaunchKernel(ctx.checksum_fn, blocks, 1, 1, threads, 1, 1, 0, NULL, args, NULL);
}

struct crypto_skcipher *tfm;
struct skcipher_request *req;
struct scatterlist sg;
char *iv;

char* data;
char* data2;

int cpu_alloc(void) {
	//char plaintext[16] = {0};
	//char ciphertext[16] = {0};
	/* We're going to use a zerod 128 bits key */
	char key[16] = {55};
	unsigned int bsize;
	int err;
	size_t ivsize;

	tfm = crypto_alloc_skcipher("cbc(aes)", 0, 0);
	if (IS_ERR(tfm)) {
        pr_err("impossible to allocate skcipher\n");
        return -1;
    }
	crypto_skcipher_setkey(tfm, key, sizeof(key));
	
	bsize = crypto_skcipher_blocksize(tfm);
	printk("Block size of CBC: %u\n", bsize);


	//init vector
	ivsize = crypto_skcipher_ivsize(tfm);

	iv = kmalloc(ivsize, GFP_KERNEL);
	if (!iv) {
        pr_err("could not allocate iv vector\n");
        return -1;
    }

	req = skcipher_request_alloc(tfm, GFP_KERNEL);
	if (!req) {
        pr_err("impossible to allocate skcipher request\n");
		return -1;
    }

	data  = kmalloc(PAGE_SIZE, GFP_KERNEL);
	data2 = kmalloc(PAGE_SIZE, GFP_KERNEL);
	if (!data) {
			err = -ENOMEM;
			return err;
	}

	sg_init_one(&sg, data, PAGE_SIZE);

	skcipher_request_set_crypt(req, &sg, &sg, 16, iv);

	err = crypto_skcipher_encrypt(req);
    if (err) {
        pr_err("could not encrypt data\n");
        return -1;
    }

	sg_copy_to_buffer(&sg, 1, data2, 16);

	printk("done\n");
	return 0;
}

void cpu_run(char* buf, int npages) {




}

#define USE_KSHM 1

// static int run(int use_gpu) {
// 	int i, j;
//     //these are changeable
// 	//int batch_sizes[] = {1,2,4,8,16,32,64,128,256,512, 1024};
// 	int batch_sizes[] = {64};
//     int n_batches = 1;
//     const int max_batch = 256;
// 	int RUNS = 3;
// 	int WARMS = 1;

//     int batch_size;
// 	u64 t_start, t_stop, c_start, c_stop;
//     u64* comp_run_times;
//     u64* total_run_times;
//     u64 avg, avg_total;

// 	char* h_page_buf;

// 	if (use_gpu)
// 		gpu_init();

// 	//alloc on kernel
// 	if(USE_KSHM)
// 		h_page_buf  = (char*) kava_alloc(PAGE_SIZE * max_batch);
// 	else
// 		h_page_buf  = (char*) kmalloc(PAGE_SIZE * max_batch, GFP_KERNEL);
// 	if(h_page_buf == 0) {
// 		printk("h_page_buf alloc failed\n");
// 		return 0;
// 	}

// 	comp_run_times  = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
// 	if(comp_run_times == 0) {
// 		printk("comp_run_times alloc failed\n");
// 		return 0;
// 	}

//     total_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
// 	if(total_run_times == 0) {
// 		printk("total_run_times alloc failed\n");
// 		return 0;
// 	}

// 	for (i = 0 ; i < n_batches ; i++) {
//         batch_size = batch_sizes[i];
		
// 		if (use_gpu) gpu_alloc(batch_size);

// 		//warmup
// 		for (j = 0 ; j < WARMS ; j++) {
// 			if (use_gpu) {
// 				gpu_setup_inputs(h_page_buf, batch_size);
// 				gpu_run(batch_size);
// 				cuCtxSynchronize();
// 			} else {

// 			}
// 			usleep_range(250, 1000);
// 		}

// 		for (j = 0 ; j < RUNS ; j++) {
//             t_start = ktime_get_ns();
//             if (use_gpu) gpu_setup_inputs(h_page_buf, batch_size);
//             c_start = ktime_get_ns();
            
// 			if (use_gpu) {
// 				gpu_run(batch_size);
// 			} else {
// 				cpu_run(batch_size);
// 			}
//             c_stop = ktime_get_ns();
//             if (use_gpu) ksm_gpu_output(h_checksum_buf, batch_size);
//             t_stop = ktime_get_ns();

//             comp_run_times[j] = (c_stop - c_start);
//             total_run_times[j] = (t_stop - t_start);
//             usleep_range(250, 1000);
//         }

// 		avg = 0; 
// 		avg_total = 0;
// 		for (j = 0 ; j < RUNS ; j++) {
// 			avg_total += total_run_times[j];
// 			avg       += comp_run_times[j];
// 		}
// 		printk("%s_%d, %lld, %lld\n",  use_gpu ? "GPU" : "CPU", batch_size, avg/(RUNS*1000), avg_total/(RUNS*1000));
// 		if (use_gpu) gpu_clean();
// 	}

// 	if(USE_KSHM) {
// 		//kava_free(h_page_buf);
// 		//kava_free(h_checksum_buf);
// 	} else {
// 		//kfree(h_page_buf);
// 		//kfree(h_checksum_buf);
// 	}
	
// 	kfree(comp_run_times);
//     kfree(total_run_times);
// 	return 0;
// }


/**
 * Program main
 */
static int __init hackcbc_init(void)
{
	cpu_alloc();
	return 0;
}

static void __exit hackcbc_fini(void)
{
    //cleanup
}

module_init(hackcbc_init);
module_exit(hackcbc_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Kernel module of a hackcbc program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");



