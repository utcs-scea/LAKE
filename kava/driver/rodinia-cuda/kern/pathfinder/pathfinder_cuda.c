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
#include "pathfinder_cuda.h"
#include "../util/util.h"

#define PATHFINDER_BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0

//#define BENCH_PRINT

struct timestamp ts_init, ts_total, ts_memalloc, ts_h2d, ts_d2h, ts_kernel, ts_close;
long init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0, d2h_phy_time = 0, h2d_phy_time = 0;

static int rows = 0;
static int cols = 0;
static int pyramid_height = 0;
static char *cubin_p = "";
module_param(rows, int, S_IRUSR);
MODULE_PARM_DESC(rows, "Number of rows)");
module_param(cols, int, S_IRUSR);
MODULE_PARM_DESC(cols, "Number of columns");
module_param(pyramid_height, int, S_IRUSR);
MODULE_PARM_DESC(pyramid_height, "Height of pyramid");
module_param(cubin_p, charp, 0000);
MODULE_PARM_DESC(cubin_p, "CUDA binary path");

int* data;
int** wall;
int* result;
#define M_SEED 9

int init(void)
{
    int n;
    int i, j;
    int rand_num;

    if (rows == 0 || cols == 0 || pyramid_height == 0 || strlen(cubin_p) <= 0) {
        pr_info("Usage: dynproc row_len col_len pyramid_height\n");
        return 1;
    }

    data = (int*)vmalloc(sizeof(int) * rows * cols);
    wall = (int**)vmalloc(sizeof(int*) * rows);
    for (n = 0; n < rows; n++)
        wall[n] = data + cols * n;
    result = (int *)vmalloc(sizeof(int) * cols);

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            get_random_bytes(&rand_num, sizeof(rand_num));
            wall[i][j] = rand_num % 10;
        }
    }
#ifdef BENCH_PRINT
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            pr_info("%d ",wall[i][j]) ;
        }
        pr_info("\n") ;
    }
#endif

    return 0;
}

void
fatal(char *s)
{
    pr_err("error: %s\n", s);
}

CUresult pathfinder_launch(CUmodule mod, int gdx, int bdx, int iteration,
                CUdeviceptr gpuWall, CUdeviceptr gpuSrc, CUdeviceptr gpuResults,
                int cols, int rows, int startStep, int border)
{
    void* param[] = {&iteration, &gpuWall, &gpuSrc, &gpuResults, &cols, &rows,
        &startStep, &border, NULL};
    CUfunction f;
    CUresult res;

    res = cuModuleGetFunction(&f, mod, "_Z14dynproc_kerneliPiS_S_iiii");
    if (res != CUDA_SUCCESS) {
        pr_err("cuModuleGetFunction failed: res = %u\n", res);
        return res;
    }

    /* shared memory size is known in the kernel image. */
    res = cuLaunchKernel(f, gdx, 1, 1, bdx, 1, 1, 0, 0, (void**) param, NULL);
    if (res != CUDA_SUCCESS) {
        pr_err("cuLaunchKernel(euclid) failed: res = %u\n", res);
        return res;
    }

    return CUDA_SUCCESS;
}

/*
   compute N time steps
*/
int calc_path(CUmodule mod, CUdeviceptr gpuWall, CUdeviceptr gpuResult[2], int rows, int cols, \
     int pyramid_height, int blockCols, int borderCols)
{
    int src = 1, dst = 0;
    int t;
    for (t = 0; t < rows-1; t+=pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;
        pathfinder_launch(mod, blockCols, PATHFINDER_BLOCK_SIZE,
                MIN(pyramid_height, rows-t-1), gpuWall, gpuResult[src],
                gpuResult[dst], cols,rows, t, borderCols);
    }
    return dst;
}

int run(void)
{
    int ret;
    int borderCols, smallBlockCol, blockCols;
    int size;
    int final_ret;
    char cubin_fn[128];

    CUcontext ctx;
    CUmodule mod;
    CUresult res;
    CUdeviceptr gpuWall, gpuResult[2];

    ret = init();
    if (ret != 0)
        return ret;

    /* --------------- pyramid parameters --------------- */
    borderCols = (pyramid_height)*HALO;
    smallBlockCol = PATHFINDER_BLOCK_SIZE-(pyramid_height)*HALO*2;
    blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    pr_info("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
    pyramid_height, cols, borderCols, PATHFINDER_BLOCK_SIZE, blockCols, smallBlockCol);

    size = rows*cols;

    /* call our common CUDA initialization utility function. */
    strcpy(cubin_fn, cubin_p);
    strcat(cubin_fn, "/pathfinder.cubin");

    probe_time_start(&ts_total);
    probe_time_start(&ts_init);
    res = cuda_driver_api_init(&ctx, &mod, cubin_fn);
    if (res != CUDA_SUCCESS) {
        pr_err("cuda_driver_api_init failed: res = %u\n", res);
        return -1;
    }
    init_time = probe_time_end(&ts_init);

    probe_time_start(&ts_memalloc);
    res = cuMemAlloc(&gpuResult[0], sizeof(int) * cols);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemAlloc(&gpuResult[1], sizeof(int) * cols);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }
    res = cuMemAlloc(&gpuWall, sizeof(int) * (size - cols));
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }
    mem_alloc_time = probe_time_end(&ts_memalloc);

    probe_time_start(&ts_h2d);
    res = cuMemcpyHtoD(gpuResult[0], data, sizeof(int) * cols);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    res = cuMemcpyHtoD(gpuWall, data+cols, sizeof(int) * (size - cols));
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }
    h2d_time = probe_time_end(&ts_h2d);

    probe_time_start(&ts_kernel);
    final_ret = calc_path(mod, gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);
    cuCtxSynchronize();
    kernel_time = probe_time_end(&ts_kernel);

    /* Copy data from device memory to main memory */
    probe_time_start(&ts_d2h);
    res = cuMemcpyDtoH(result, gpuResult[final_ret], sizeof(float) * cols);
    if (res != CUDA_SUCCESS) {
        pr_err("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }
    d2h_time += probe_time_end(&ts_d2h);

    probe_time_start(&ts_close);
    cuMemFree(gpuWall);
    cuMemFree(gpuResult[0]);
    cuMemFree(gpuResult[1]);

    res = cuda_driver_api_exit(ctx, mod);
    if (res != CUDA_SUCCESS) {
        pr_err("cuda_driver_api_exit failed: res = %u\n", res);
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

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
        pr_info("%d ",data[i]);
    pr_info("\n") ;
    for (int i = 0; i < cols; i++)
        pr_info("%d ",result[i]);
    pr_info("\n") ;
#endif

    vfree(data);
    vfree(wall);
    vfree(result);

    return 0;
}

static int __init drv_init(void)
{
    pr_info("load cuda pathfinder module\n");
    return run();
}

static void __exit drv_fini(void)
{
    pr_info("unload cuda pathfinder module\n");
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
