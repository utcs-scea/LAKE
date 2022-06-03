/*******************************************************************************

  Demo linux driver module for kernel-space CUDA API.

*******************************************************************************/

#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "cuda_kava.h"
#include "shared_memory.h"

#define block_size 16

#define SIMPLE_TEST(fn)    do { \
    CUresult ret;               \
    pr_info("Before simple test: execute " #fn "\n");          \
    ret = fn;                   \
    pr_info("After simple test: execute " #fn " = %d\n", ret); \
} while (0)

static int __init drv_init(void)
{
    pr_info("load cuda demo driver\n");

#if defined(TEST_CUDA_HELLOWORLD) || 1
#define REPORT_ERROR(func)             \
    if (ret != CUDA_SUCCESS) {         \
        pr_err(#func " error\n"); \
        return 0;                      \
    }

    {
        CUresult ret;
        CUdevice dev;
        CUcontext ctx;
        CUmodule mod;
        CUfunction func;
        int blocks_per_grid = 4, threads_per_block = 5;
        void *args = NULL;
        CUdeviceptr mem;
        const int count = 10;
        int val[count];
        int *shm_val1, *shm_val2;
        int i;

        ret = cuInit(0);
        REPORT_ERROR(cuInit);

        ret = cuDeviceGet(&dev, 0);
        REPORT_ERROR(cuDeviceGet);

        ret = cuCtxCreate(&ctx, 0, dev);
        REPORT_ERROR(cuCtxCreate);

        ret = cuModuleLoad(&mod, "/home/hyu/kava/driver/cuda/hello_world.cubin");
        REPORT_ERROR(cuModuleLoad);

        ret = cuModuleGetFunction(&func, mod, "_Z10HelloWorldv");
        REPORT_ERROR(cuModuleGetFunction);

        ret = cuCtxSetCurrent(ctx);
        REPORT_ERROR(cuCtxSetCurrent);

        ret = cuLaunchKernel(func, blocks_per_grid, 1, 1, threads_per_block, 1, 1,
                0, 0, args, 0);
        REPORT_ERROR(cuLaunchKernel);

        /* Synchronous memory copy */
        ret = cuMemAlloc(&mem, count * sizeof(int));
        REPORT_ERROR(cuMemAlloc);

        for (i = 0; i < count; i++)
            val[i] = i;
        ret = cuMemcpyHtoD(mem, val, count * sizeof(int));
        REPORT_ERROR(cuMemcpyHtoD);
        memset(val, 0, sizeof(val));
        ret = cuMemcpyDtoH(val, mem, count * sizeof(int));
        REPORT_ERROR(cuMemcpyDtoH);
        for (i = 0; i < count; i++)
            pr_info("val[%d] = %d\n", i, val[i]);

        /* Asynchronous memory copy */
        shm_val1 = (int *)kava_alloc(sizeof(int) * count);
        shm_val2 = (int *)kava_alloc(sizeof(int) * count);
        for (i = 0; i < count; i++) {
            shm_val1[i] = i;
            shm_val2[i] = 0;
        }
        ret = cuMemcpyHtoDAsync(mem, (void *)shm_val1, sizeof(int) * count, NULL);
        REPORT_ERROR(cuMemcpyHtoDAsync);
        ret = cuMemcpyDtoHAsync((void *)shm_val2, mem, sizeof(int) * count, NULL);
        REPORT_ERROR(cuMemcpyDtoHAsync);
        ret = cuCtxSynchronize();
        REPORT_ERROR(cuCtxSynchronize);
        for (i = 0; i < count; i++)
            pr_info("shm_val2[%d] = %d\n", i, shm_val2[i]);
        kava_free(shm_val1);
        kava_free(shm_val2);

        ret = cuMemFree(mem);
        REPORT_ERROR(cuMemFree);

        ret = cuCtxDestroy(ctx);
        REPORT_ERROR(cuCtxDestroy);
    }
#endif

    /* Call CUDA APIs */
#if defined(TEST_CUDA_API)
    {
        SIMPLE_TEST(cuInit(0));
    }
    {
        CUdevice device;
        SIMPLE_TEST(cuDeviceGet(&device, 0));
    }
    {
        CUcontext ctx;
        CUdevice device = (CUdevice)0x100;
        SIMPLE_TEST(cuCtxCreate(&ctx, 0, device));
    }
    {
        CUmodule module;
        SIMPLE_TEST(cuModuleLoad(&module, "fatbin_name"));
    }
    {
        CUmodule module = (CUmodule)0x100;
        SIMPLE_TEST(cuModuleUnload(module));
    }
    {
        CUfunction func;
        CUmodule module = (CUmodule)0x100;
        SIMPLE_TEST(cuModuleGetFunction(&func, module, "_Z4zoomi"));
    }
    {
        CUfunction func = (CUfunction)0x100;
        CUstream stream = (CUstream)0x100;
        SIMPLE_TEST(cuLaunchKernel(func, 0, 0, 0, 0, 0, 0, 0, stream, NULL, NULL));
    }
    {
        CUcontext ctx = (CUcontext)0x100;
        SIMPLE_TEST(cuCtxDestroy(ctx));
    }
    {
        CUdeviceptr dptr;
        SIMPLE_TEST(cuMemAlloc(&dptr, 0x1000));
    }
    {
        CUdeviceptr dstDevice = 0x100;
        char srcHost[0x100];
        SIMPLE_TEST(cuMemcpyHtoD(dstDevice, srcHost, 0x100));
    }
    {
        char dstHost[0x100];
        CUdeviceptr srcDevice = 0x100;
        SIMPLE_TEST(cuMemcpyDtoH(dstHost, srcDevice, 0x100));
    }
    {
        SIMPLE_TEST(cuCtxSynchronize());
    }
    {
        int driverVersion;
        SIMPLE_TEST(cuDriverGetVersion(&driverVersion));
    }
    {
        CUdeviceptr dptr = 0x100;
        SIMPLE_TEST(cuMemFree(dptr));
    }
    {
        CUdeviceptr dptr;
        size_t bytes = 0;
        CUmodule module = (CUmodule)0x100;
        SIMPLE_TEST(cuModuleGetGlobal(&dptr, &bytes, module, "module_name"));
    }
    {
        int count;
        SIMPLE_TEST(cuDeviceGetCount(&count));
    }
#endif

#if defined(TEST_UPCALL)
    cuTestInit();
    msleep(2000); // Make sure the last API has finished
    cuTestMmul(0);
    msleep(1000); // Make sure the last API has finished
    cuTestFree();
    msleep(1000); // Make sure the last API has finished
    cuTestKtoU(block_size * 4);
#endif

    return 0;
}

static void __exit drv_fini(void)
{
    pr_info("unload cuda demo driver\n");
}

module_init(drv_init);
module_exit(drv_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Demo driver module for kernel-space CUDA API");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
