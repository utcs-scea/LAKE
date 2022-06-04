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
#include <asm/atomic.h>

#include "helpers.h"

const char *ecb_enc_function_name = "_Z15aes_encrypt_bptPjiPh";
const char *ecb_dec_function_name = "_Z15aes_decrypt_bptPjiPh";

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

CUdeviceptr g_key_enc;
CUdeviceptr g_key_dec;
CUdeviceptr d_page_buf;
CUdeviceptr d_rk;
int key_round;

int gpu_init(void) {
	char* enc_key;
	int i;

	cuInit(0);
	// Setup get device
	check_error(cuDeviceGet(&device, 0), "cuDeviceGet");
	check_error(cuCtxCreate(&context, 0, device), "cuCtxCreate");
	check_error(cuModuleLoad(&module, cubin_path), "cuModuleLoad");
	check_error(cuModuleGetFunction(&enc_fn, module, ecb_enc_function_name), "cuModuleGetFunction");
	check_error(cuModuleGetFunction(&dec_fn, module, ecb_dec_function_name), "cuModuleGetFunction");

	cuFuncSetCacheConfig(enc_fn, CU_FUNC_CACHE_PREFER_L1);

	//use fixed key, same as cpu
	enc_key = kmalloc(56, GFP_KERNEL);
	for (i = 0 ; i < 56 ; i++) enc_key[i] = 55;

	cuMemAlloc(&g_key_enc, 56);
    cuMemcpyHtoD(g_key_enc, &enc_key, 56);
	kfree(enc_key);
    key_round = 16 / 4 + 6;
	
	return 0;
}

void gpu_alloc(uint32_t npages) {
	check_error(cuMemAlloc(&d_page_buf, PAGE_SIZE * npages), "cuMemAlloc 1");
}

void gpu_clean(void) {
	check_error(cuMemFree(d_page_buf), "cuMemFree");
}

void gpu_setup_inputs(char* kpages, uint32_t npages) {
	check_error(cuMemcpyHtoD(d_page_buf, kpages, npages*PAGE_SIZE), "cuMemcpyHtoD");	
}

void gpu_output(char* out, uint32_t npages) {
	check_error(cuMemcpyDtoH(out, d_page_buf, npages*PAGE_SIZE), "cuMemcpyDtoH");
}

void gpu_run(uint32_t npages) {	
	int block_x, grid_x, res;
	void *args[] = { &g_key_enc, &key_round, &d_page_buf };
	block_x = PAGE_SIZE / 16;   //each thread handles 16 bytes
    grid_x = npages;
	check_error(cuLaunchKernel(enc_fn, grid_x, 1, 1, block_x, 1, 1, 0, NULL, args, NULL), "cuLaunchKernel");
}

struct crypto_skcipher *tfm;
struct skcipher_request *req;
struct scatterlist sg;
char *iv;
size_t ivsize;

int cpu_alloc(void) {
	/* We're going to use a zerod 128 bits key */
	char key[16] = {55};
	unsigned int bsize;
	
	tfm = crypto_alloc_skcipher("cbc(aes)", 0, 0);
	if (IS_ERR(tfm)) {
        pr_err("impossible to allocate skcipher\n");
        return -1;
    }
	crypto_skcipher_setkey(tfm, key, sizeof(key));

	//bsize = crypto_skcipher_blocksize(tfm);
	//printk("Block size of CBC: %u\n", bsize);
	ivsize = crypto_skcipher_ivsize(tfm);
	//printk("ivsize of CBC: %lu\n", ivsize);

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

	return 0;
}

void cpu_clean(void) {
	crypto_free_skcipher(tfm);
	skcipher_request_free(req);
	kfree(iv);
}

atomic_t atm_done_reqs;

void req_done(struct crypto_async_request *req, int error) {
	if (error){
		printk("ERROR ON REQ CALLBACK\n");
	} else {
		atomic_inc(&atm_done_reqs);
	}
}

int cpu_run_async(char* buf, int npages) {
	int i, err, done;
	struct crypto_wait *waits = 0;
	struct skcipher_request **reqs = 0;
	struct scatterlist *sgs = 0;
	char** ivs = 0;

	waits = kmalloc(npages * sizeof(struct crypto_wait), GFP_KERNEL);
	reqs = kmalloc(npages * sizeof(struct skcipher_request*), GFP_KERNEL);
	sgs = kmalloc(npages * sizeof(struct scatterlist), GFP_KERNEL);
	ivs = kmalloc(npages * sizeof(char*), GFP_KERNEL);
	for (i = 0 ; i < npages ; i++) {
		ivs[i] = kmalloc(ivsize, GFP_KERNEL);
		if (ivs == 0) {
			printk("ivs allocation failed, something bad is gonna happen\n");
			return -1;
		}
		get_random_bytes(ivs[i], ivsize);
	}
	atm_done_reqs = ((atomic_t) { (0) });

	for (i = 0 ; i < npages ; i++) {
		char* data = buf + (i*PAGE_SIZE);
		reqs[i] = skcipher_request_alloc(tfm, GFP_KERNEL);
		if (!reqs[i]) {
			pr_err("impossible to allocate skcipher request\n");
			return -1;
		}
		sg_init_one(&sgs[i], data, PAGE_SIZE);
		skcipher_request_set_crypt(reqs[i], &sgs[i], &sgs[i], 16, ivs[i]);
		crypto_init_wait(&waits[i]);
		skcipher_request_set_callback(reqs[i], CRYPTO_TFM_REQ_MAY_BACKLOG,
                      req_done, &waits[i]);

		err = crypto_skcipher_encrypt(reqs[i]);
		if (err) {
			pr_err("could not encrypt data\n");
			return -1;
		}
	}

	for (i = 0 ; i < npages ; i++) {
		done = atomic_read(&atm_done_reqs);
		if (done == npages) break;
	}

	if (waits) kfree(waits);
	if (reqs) {
		for (i = 0 ; i < npages ; i++) skcipher_request_free(reqs[i]);
		kfree(reqs);
	}
	if (sgs) kfree(sgs);
	if (ivs) {
		for (i = 0 ; i < npages ; i++) kfree(ivs[i]);
		kfree(ivs);
	} 

	return 0;
}

int cpu_run_sync(char* buf, int npages) {
	int i, err;
	char* data;

	for (i = 0 ; i < npages ; i++) {
		get_random_bytes(iv, ivsize);
		data = buf + (i*PAGE_SIZE);
		sg_init_one(&sg, data, PAGE_SIZE);
		skcipher_request_set_crypt(req, &sg, &sg, 16, iv);
		err = crypto_skcipher_encrypt(req);
		if (err) {
			pr_err("could not encrypt data\n");
			return -1;
		}
	}
	return 0;
}

#define USE_KSHM 1

static int run(int use_gpu, int async) {
	int i, j;
    //these are changeable
	//int batch_sizes[] = {512};
	int batch_sizes[] = {1,2,4,8,16,32,64,128,256,512, 1024};
    int n_batches = 11;
    const int max_batch = 1024;
	int RUNS = 3;
	int WARMS = 1;

    int batch_size;
	u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;

	char* h_page_buf;

	if (use_gpu)
		gpu_init();

	//alloc on kernel
	if(USE_KSHM)
		h_page_buf  = (char*) kava_alloc(PAGE_SIZE * max_batch);
	else
		h_page_buf  = (char*) kmalloc(PAGE_SIZE * max_batch, GFP_KERNEL);
	if(h_page_buf == 0) {
		printk("h_page_buf alloc failed\n");
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

		if (use_gpu) gpu_alloc(batch_size);
		else cpu_alloc();
		
		//warmup
		for (j = 0 ; j < WARMS ; j++) {
			if (use_gpu) {
				gpu_alloc(batch_size);
				gpu_setup_inputs(h_page_buf, batch_size);
				gpu_run(batch_size);
				cuCtxSynchronize();
			} else {
				if (async)
					cpu_run_async(h_page_buf, batch_size);
				else
					cpu_run_sync(h_page_buf, batch_size);
			}
			usleep_range(250, 1000);
		}

		for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();
            if (use_gpu) 
				gpu_setup_inputs(h_page_buf, batch_size);
            c_start = ktime_get_ns();
            
			if (use_gpu) {
				gpu_run(batch_size);
			} else {
				if (async)
					cpu_run_async(h_page_buf, batch_size);
				else
					cpu_run_sync(h_page_buf, batch_size);
			}
            c_stop = ktime_get_ns();
            if (use_gpu) gpu_output(h_page_buf, batch_size);
            //cuCtxSynchronize();
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

		if (RUNS != 0) {
			if (use_gpu) {
				printk("GPU_%d, %lld, %lld\n", batch_size, avg/(RUNS*1000), avg_total/(RUNS*1000));
			} else {
				printk("CPU_%s_%d, %lld\n", async? "async" : "",  batch_size, avg_total/(RUNS*1000));
			}
		}
		if (use_gpu) gpu_clean();
		else cpu_clean();
	}

	if(USE_KSHM) {
		kava_free(h_page_buf);
		//kava_free(h_checksum_buf);
	} else {
		kfree(h_page_buf);
		//kfree(h_checksum_buf);
	}
	
	kfree(comp_run_times);
    kfree(total_run_times);
	return 0;
}


/**
 * Program main
 */

#define USE_CPU 0
#define USE_GPU 1
#define CPU_SYNC 0
#define CPU_ASYNC 1

static int __init hackcbc_init(void)
{
	//run(USE_CPU, CPU_SYNC);
	//run(USE_CPU, CPU_ASYNC);
	run(USE_GPU, 0);
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



