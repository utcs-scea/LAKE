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
#include "../../util/util.h"
#include "../common/common.h"
#include "lud.h"

static int matrix_dim = 0;
static int do_verify = 0;
static char *cubin_p = NULL;
static char *input_f = NULL;
module_param(matrix_dim, int, S_IRUSR);
MODULE_PARM_DESC(matrix_dim, "Matrix dimension");
module_param(do_verify, int, S_IRUSR);
MODULE_PARM_DESC(do_verify, "Whether to verify the result");
module_param(cubin_p, charp, 0000);
MODULE_PARM_DESC(cubin_p, "CUDA binary directory path");
module_param(input_f, charp, 0000);
MODULE_PARM_DESC(input_f, "Input file path");

struct timestamp ts_init, ts_total, ts_memalloc, ts_h2d, ts_d2h, ts_kernel, ts_close;
long init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0, d2h_phy_time = 0, h2d_phy_time = 0;

int lud_launch(CUmodule mod, CUdeviceptr m, int matrix_dim)
{
	int i = 0;
	int bdx, bdy, gdx, gdy;
	int shared_size;
	int *m_debug = (int *)vmalloc(matrix_dim * matrix_dim * sizeof(int));
	CUfunction f_diagonal, f_perimeter, f_internal;
	CUresult res;

	/* get functions. */
	res = cuModuleGetFunction(&f_diagonal, mod, "_Z12lud_diagonalPfii");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(f_diagonal) failed\n");
		return 0;
	}
	res = cuModuleGetFunction(&f_perimeter, mod, "_Z13lud_perimeterPfii");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(f_perimeter) failed\n");
		return 0;
	}
	res = cuModuleGetFunction(&f_internal, mod, "_Z12lud_internalPfii");
	if (res != CUDA_SUCCESS) {
		pr_err("cuModuleGetFunction(f_internal) failed\n");
		return 0;
	}
	
	for (i = 0; i < matrix_dim - LUD_BLOCK_SIZE; i += LUD_BLOCK_SIZE) {
		void* param[] = {(void*) &m, (void*) &matrix_dim, (void*) &i, NULL};
		/* diagonal */
		gdx = 1;
		gdy = 1;
		bdx = LUD_BLOCK_SIZE;
		bdy = 1;
		shared_size = LUD_BLOCK_SIZE * LUD_BLOCK_SIZE * sizeof(int);
		res = cuLaunchKernel(f_diagonal, gdx, gdy, 1, bdx, bdy, 1, shared_size,
							 0, (void**) param, NULL);
        if (res != CUDA_SUCCESS) {
            pr_err("cuLaunchKernel(f_diagonal) failed: res = %u\n", res);
            return 0;
        }

		/* perimeter */
		gdx = (matrix_dim - i) / LUD_BLOCK_SIZE - 1;
		gdy = 1;
		bdx = LUD_BLOCK_SIZE * 2;
		bdy = 1;
		shared_size = LUD_BLOCK_SIZE * LUD_BLOCK_SIZE * sizeof(int) * 3;
		res = cuLaunchKernel(f_perimeter, gdx, gdy, 1, bdx, bdy, 1, shared_size,
							 0, (void**) param, NULL);
        if (res != CUDA_SUCCESS) {
            pr_err("cuLaunchKernel(f_perimeter) failed: res = %u\n", res);
            return 0;
        }

		/* internal */
		gdx = (matrix_dim - i) / LUD_BLOCK_SIZE - 1;
		gdy = (matrix_dim - i) / LUD_BLOCK_SIZE - 1;
		bdx = LUD_BLOCK_SIZE;
		bdy = LUD_BLOCK_SIZE;
		shared_size = LUD_BLOCK_SIZE * LUD_BLOCK_SIZE * sizeof(int) * 2;
		res = cuLaunchKernel(f_internal, gdx, gdy, 1, bdx, bdy, 1, shared_size,
							 0, (void**) param, NULL);
        if (res != CUDA_SUCCESS) {
            pr_err("cuLaunchKernel(internal) failed: res = %u\n", res);
            return 0;
        }
	}

    {
        void* param[] = {(void*) &m, (void*) &matrix_dim, (void*) &i};
        /* diagonal */
        gdx = 1;
        gdy = 1;
        res = cuLaunchKernel(f_diagonal, gdx, gdy, 1, bdx, bdy, 1, shared_size,
                             0, (void**) param, NULL);
        if (res != CUDA_SUCCESS) {
            pr_err("cuLaunchKernel(f_diagonal) failed: res = %u\n", res);
            return 0;
        }
    }

	vfree(m_debug);

	return 0;
}

static int __init drv_init(void)
{
	func_ret_t ret;
	int *m, *mm;
	CUdeviceptr d_m;
	CUcontext ctx;
	CUmodule mod;
	CUresult res;
    char cubin_fn[128];

    pr_info("load cuda lud module\n");

    if (!cubin_p || (matrix_dim == 0 && !input_f)) {
        pr_err("Usage: insmod lud.ko [do_verify=<bool>] [matrix_dim=<int>|input_f=<path>] cubin_p=<path>\n");
        return 1;
    }

	if (input_f) {
		pr_info("Reading matrix from file %s\n", input_f);
		ret = create_matrix_from_file(&m, input_f, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			pr_err("error create matrix from file %s\n", input_f);
            return 0;
		}
	}

    if (input_f == NULL &&  matrix_dim != 0) {
	    create_matrix(&m, matrix_dim);
    }

	if (do_verify){
		print_matrix(m, matrix_dim);
		matrix_duplicate(m, &mm, matrix_dim);
	}

	/*
	 * call our common CUDA initialization utility function.
	 */
    strcpy(cubin_fn, cubin_p);
    strcat(cubin_fn, "/lud.cubin");

    probe_time_start(&ts_total);
    probe_time_start(&ts_init);
	res = cuda_driver_api_init(&ctx, &mod, cubin_fn);
	if (res != CUDA_SUCCESS) {
		pr_err("cuda_driver_api_init failed: res = %u\n", res);
		return -1;
	}
    init_time = probe_time_end(&ts_init);

    probe_time_start(&ts_memalloc);
	res = cuMemAlloc(&d_m, matrix_dim * matrix_dim * sizeof(int));
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemAlloc failed\n");
		return -1;
	}
    mem_alloc_time = probe_time_end(&ts_memalloc);

    probe_time_start(&ts_h2d);
    res = cuMemcpyHtoD(d_m, m, matrix_dim * matrix_dim * sizeof(int));
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyHtoD (a) failed: res = %u\n", res);
		return -1;
	}
    h2d_time = probe_time_end(&ts_h2d);

    probe_time_start(&ts_kernel);
	lud_launch(mod, d_m, matrix_dim);

	cuCtxSynchronize();
    kernel_time = probe_time_end(&ts_kernel);

    probe_time_start(&ts_d2h);
	res = cuMemcpyDtoH(m, d_m, matrix_dim * matrix_dim * sizeof(int));
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemcpyDtoH failed: res = %u\n", res);
		return -1;
	}
    d2h_time += probe_time_end(&ts_d2h);

    probe_time_start(&ts_close);
	res = cuMemFree(d_m);
	if (res != CUDA_SUCCESS) {
		pr_err("cuMemFree failed: res = %u\n", res);
		return -1;
	}

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

	if (do_verify){
		print_matrix(m, matrix_dim);
		pr_info(">>>Verify<<<<\n");
		lud_verify(mm, m, matrix_dim);
		vfree(mm);
	}

	vfree(m);

	return 0;
}				/* ----------  end of function main  ---------- */

static void __exit drv_fini(void)
{
    pr_info("unload lud needle module\n");
}

module_init(drv_init);
module_exit(drv_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Driver module for Rodinia lud");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
