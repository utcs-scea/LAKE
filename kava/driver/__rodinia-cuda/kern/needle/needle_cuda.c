/**
 * Ported to kernel by Hangchen Yu.
 */
#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "cuda_kava.h"
#define LIMIT -999
#include "needle_cuda.h"
#include "../util/util.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

struct timestamp ts_init, ts_total, ts_memalloc, ts_h2d, ts_d2h, ts_kernel, ts_close;
long init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0, d2h_phy_time = 0, h2d_phy_time = 0;

static int max_size = 0;
static int penalty = 0;
static char *cubin_p = "";
module_param(max_size, int, S_IRUSR);
MODULE_PARM_DESC(max_size, "Dimension (max rows or columns)");
module_param(penalty, int, S_IRUSR);
MODULE_PARM_DESC(penalty, "Penalty (positive integer)");
module_param(cubin_p, charp, 0000);
MODULE_PARM_DESC(cubin_p, "CUDA binary path");

void usage(void)
{
	pr_err("Usage: insmod needle.ko max_size=<dimension> penalty=<penalty> cubin_p=<path>\n");
	pr_err("\t<dimension> - x and y dimensions\n");
	pr_err("\t<penalty> - penalty\n");
	pr_err("\t<path> - cubin directory\n");
}

CUresult needle_launch(CUmodule mod, int gdx, int gdy, int bdx, int bdy,
        CUdeviceptr referrence_cuda, CUdeviceptr matrix_cuda, CUdeviceptr matrix_cuda_out,
        int max_cols, int penalty, int i, int block_width)
{
    void* param[] = {&referrence_cuda, &matrix_cuda, &matrix_cuda_out, &max_cols, &penalty, &i, &block_width, NULL};
    CUfunction f;
    CUresult res;

    res = cuModuleGetFunction(&f, mod, "_Z20needle_cuda_shared_1PiS_S_iiii");
    if (res != CUDA_SUCCESS) {
        pr_err("cuModuleGetFunction failed: res = %u\n", res);
        return res;
    }

    /* shared memory size is known in the kernel image. */
    res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**) param, NULL);
    if (res != CUDA_SUCCESS) {
        pr_err("cuLaunchKernel(euclid) failed: res = %u\n", res);
        return res;
    }

    return CUDA_SUCCESS;
}

CUresult needle_launch2(CUmodule mod, int gdx, int gdy, int bdx, int bdy,
        CUdeviceptr referrence_cuda, CUdeviceptr matrix_cuda, CUdeviceptr matrix_cuda_out,
        int max_cols, int penalty, int i, int block_width)
{
    void* param[] = {&referrence_cuda, &matrix_cuda, &matrix_cuda_out, &max_cols, &penalty, &i, &block_width, NULL};
    CUfunction f;
    CUresult res;

    res = cuModuleGetFunction(&f, mod, "_Z20needle_cuda_shared_2PiS_S_iiii");
    if (res != CUDA_SUCCESS) {
        pr_err("cuModuleGetFunction failed: res = %u\n", res);
        return res;
    }

    /* shared memory size is known in the kernel image. */
    res = cuLaunchKernel(f, gdx, gdy, 1, bdx, bdy, 1, 0, 0, (void**) param, NULL);
    if (res != CUDA_SUCCESS) {
        pr_err("cuLaunchKernel(euclid) failed: res = %u\n", res);
        return res;
    }

    return CUDA_SUCCESS;
}

int runTest(void)
{
    int max_rows, max_cols;
    int rand_num;
    int *input_itemsets, *output_itemsets, *referrence;
	int size;
    int block_width;
    int i, j;
    char cubin_fn[128];

    CUcontext ctx;
    CUmodule mod;
    CUresult res;
    CUdeviceptr referrence_cuda, matrix_cuda, matrix_cuda_out;

    // the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
    if (max_size <= 0 || penalty <= 0 || strlen(cubin_p) <= 0) {
	    usage();
        return 1;
    }

	if (max_size % 16 != 0) {
        pr_err("The dimension values must be a multiple of 16\n");
        return 1;
	}

	max_rows = max_size + 1;
	max_cols = max_size + 1;
	referrence = (int *)vmalloc( max_rows * max_cols * sizeof(int) );
    input_itemsets = (int *)vmalloc( max_rows * max_cols * sizeof(int) );
	output_itemsets = (int *)vmalloc( max_rows * max_cols * sizeof(int) );

	if (!input_itemsets)
		pr_err("error: can not allocate memory");

    for (i = 0 ; i < max_cols; i++){
		for (j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}

	pr_info("Start Needleman-Wunsch\n");

	for(i=1; i< max_rows ; i++){    //please define your own sequence.
        get_random_bytes(&rand_num, sizeof(rand_num));
        input_itemsets[i*max_cols] = rand_num % 10 + 1;
	}
    for(j=1; j< max_cols ; j++){    //please define your own sequence.
        get_random_bytes(&rand_num, sizeof(rand_num));
        input_itemsets[j] = rand_num % 10 + 1;
	}

	for (i = 1 ; i < max_cols; i++){
		for (j = 1 ; j < max_rows; j++){
		referrence[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

    for(i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
	for(j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;

    /* call our common CUDA initialization utility function. */
    strcpy(cubin_fn, cubin_p);
    strcat(cubin_fn, "/needle.cubin");

    probe_time_start(&ts_total);
    probe_time_start(&ts_init);
    res = cuda_driver_api_init(&ctx, &mod, cubin_fn);
    if (res != CUDA_SUCCESS) {
        pr_err("cuda_driver_api_init failed: res = %u\n", res);
        return -1;
    }

    size = max_cols * max_rows;

    init_time = probe_time_end(&ts_init);
    probe_time_start(&ts_memalloc);

    /* Allocate device memory */
    res = cuMemAlloc(&referrence_cuda, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemAlloc(&matrix_cuda, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemAlloc(&matrix_cuda_out, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    mem_alloc_time = probe_time_end(&ts_memalloc);
    probe_time_start(&ts_h2d);

    /* Copy data from main memory to device memory */
    res = cuMemcpyHtoD(referrence_cuda, referrence, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    res = cuMemcpyHtoD(matrix_cuda, input_itemsets, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    h2d_time = probe_time_end(&ts_h2d);

	block_width = ( max_cols - 1 )/NEEDLE_BLOCK_SIZE;

	pr_info("Processing top-left matrix\n");

    probe_time_start(&ts_kernel);

	//process top-left matrix
	for(i = 1 ; i <= block_width ; i++){
        needle_launch(mod, i, 1, NEEDLE_BLOCK_SIZE, 1, referrence_cuda, matrix_cuda,
                matrix_cuda_out, max_cols, penalty, i, block_width);
	}
	pr_info("Processing bottom-right matrix\n");
    //process bottom-right matrix
	for(i = block_width - 1  ; i >= 1 ; i--){
        needle_launch2(mod, i, 1, NEEDLE_BLOCK_SIZE, 1, referrence_cuda, matrix_cuda,
                matrix_cuda_out, max_cols, penalty, i, block_width);
	}

    cuCtxSynchronize();
    kernel_time = probe_time_end(&ts_kernel);
    probe_time_start(&ts_d2h);

    /* Copy data from device memory to main memory */
    res = cuMemcpyDtoH(output_itemsets, matrix_cuda, sizeof(int) * size);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    d2h_time += probe_time_end(&ts_d2h);
    probe_time_start(&ts_close);

	cuMemFree(referrence_cuda);
	cuMemFree(matrix_cuda);
	cuMemFree(matrix_cuda_out);

	res = cuda_driver_api_exit(ctx, mod);
	if (res != CUDA_SUCCESS) {
		pr_err("cuda_driver_api_exit faild: res = %u\n", res);
		return -1;
	}

    close_time = probe_time_end(&ts_close);
	total_time = probe_time_end(&ts_total);

    pr_info("Init: %ld\n", init_time);
	pr_info("MemAlloc: %ld\n", mem_alloc_time);
	pr_info("HtoD: %ld\n", h2d_time);
	pr_info("Exec: %ld\n", kernel_time);
	pr_info("DtoH: %ld\n", d2h_time);
	pr_info("Close: %ld\n", close_time);
	pr_info("API: %ld\n", init_time+mem_alloc_time+h2d_time+kernel_time+d2h_time+close_time);
	pr_info("Total: %ld (ns)\n", total_time);

	vfree(referrence);
	vfree(input_itemsets);
	vfree(output_itemsets);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

static int __init drv_init(void)
{
    pr_info("load cuda needle module\n");

    return runTest();
}

static void __exit drv_fini(void)
{
    pr_info("unload cuda needle module\n");
}

module_init(drv_init);
module_exit(drv_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Driver module for Rodinia needle");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
