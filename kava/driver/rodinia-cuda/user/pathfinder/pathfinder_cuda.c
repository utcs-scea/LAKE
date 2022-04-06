#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <cuda.h>
#include "util.h"

#include "pathfinder_cuda.h"

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0

//#define BENCH_PRINT

struct timestamp ts_init, ts_total, ts_memalloc, ts_h2d, ts_d2h, ts_kernel, ts_close;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0, d2h_phy_time = 0, h2d_phy_time = 0;

void init(int argc, char** argv);
int run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

int main(int argc, char** argv)
{
    int rt;
    rt = run(argc,argv);
    if (rt < 0) return rt;

    return EXIT_SUCCESS;
}

void
init(int argc, char** argv)
{
    if(argc==4){
        cols = atoi(argv[1]);
        rows = atoi(argv[2]);
                pyramid_height=atoi(argv[3]);
    }else{
                printf("Usage: dynproc row_len col_len pyramid_height\n");
                exit(0);
        }
    data = (int*)malloc(sizeof(int) * rows * cols);
    wall = (int**)malloc(sizeof(int*) * rows);
    int n;
    for(n=0; n<rows; n++)
        wall[n]=data+cols*n;
    result = (int*)malloc(sizeof(int) * cols);

    int seed = M_SEED;
    srand(seed);

    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
#ifdef BENCH_PRINT
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif
}

void
fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);
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
        printf("cuModuleGetFunction failed: res = %u\n", res);
        return res;
    }

    /* shared memory size is known in the kernel image. */
    res = cuLaunchKernel(f, gdx, 1, 1, bdx, 1, 1, 0, 0, (void**) param, NULL);
    if (res != CUDA_SUCCESS) {
        printf("cuLaunchKernel(euclid) failed: res = %u\n", res);
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
        pathfinder_launch(mod, blockCols, BLOCK_SIZE,
                MIN(pyramid_height, rows-t-1), gpuWall, gpuResult[src],
                gpuResult[dst], cols,rows, t, borderCols);
    }
    return dst;
}

int run(int argc, char** argv)
{
    init(argc, argv);

    CUcontext ctx;
    CUmodule mod;
    CUresult res;

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
    pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

    CUdeviceptr gpuWall, gpuResult[2];
    int size = rows*cols;

    /* call our common CUDA initialization utility function. */
    probe_time_start(&ts_total);
    probe_time_start(&ts_init);
    res = cuda_driver_api_init(&ctx, &mod, "./pathfinder.cubin");
    if (res != CUDA_SUCCESS) {
        printf("cuda_driver_api_init failed: res = %u\n", res);
        return -1;
    }
    init_time = probe_time_end(&ts_init);

    probe_time_start(&ts_memalloc);
    res = cuMemAlloc(&gpuResult[0], sizeof(int) * cols);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }

    res = cuMemAlloc(&gpuResult[1], sizeof(int) * cols);
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }
    res = cuMemAlloc(&gpuWall, sizeof(int) * (size - cols));
    if (res != CUDA_SUCCESS) {
        printf("cuMemAlloc failed: res = %u\n", res);
        return -1;
    }
    mem_alloc_time = probe_time_end(&ts_memalloc);

    probe_time_start(&ts_h2d);
    res = cuMemcpyHtoD(gpuResult[0], data, sizeof(int) * cols);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }

    res = cuMemcpyHtoD(gpuWall, data+cols, sizeof(int) * (size - cols));
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }
    h2d_time = probe_time_end(&ts_h2d);

    int final_ret;
    probe_time_start(&ts_kernel);
    final_ret = calc_path(mod, gpuWall, gpuResult, rows, cols, pyramid_height, blockCols, borderCols);
    cuCtxSynchronize();
    kernel_time = probe_time_end(&ts_kernel);

    /* Copy data from device memory to main memory */
    probe_time_start(&ts_d2h);
    res = cuMemcpyDtoH(result, gpuResult[final_ret], sizeof(float) * cols);
    if (res != CUDA_SUCCESS) {
        printf("cuMemcpyHtoD failed: res = %u\n", res);
        return -1;
    }
    d2h_time += probe_time_end(&ts_d2h);

    probe_time_start(&ts_close);
    cuMemFree(gpuWall);
    cuMemFree(gpuResult[0]);
    cuMemFree(gpuResult[1]);

    res = cuda_driver_api_exit(ctx, mod);
    if (res != CUDA_SUCCESS) {
        printf("cuda_driver_api_exit failed: res = %u\n", res);
        return -1;
    }

    close_time = probe_time_end(&ts_close);
	total_time = probe_time_end(&ts_total);

    printf("Init: %f\n", init_time);
	printf("MemAlloc: %f\n", mem_alloc_time);
	printf("HtoD: %f\n", h2d_time);
	printf("Exec: %f\n", kernel_time);
	printf("DtoH: %f\n", d2h_time);
	printf("Close: %f\n", close_time);
	printf("API: %f\n", init_time+mem_alloc_time+h2d_time+kernel_time+d2h_time+close_time);
	printf("Total: %f\n", total_time);

#ifdef BENCH_PRINT
    for (int i = 0; i < cols; i++)
            printf("%d ",data[i]) ;
    printf("\n") ;
    for (int i = 0; i < cols; i++)
            printf("%d ",result[i]) ;
    printf("\n") ;
#endif

    free(data);
    free(wall);
    free(result);

    return 0;
}
