#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define kava_is_worker 1

#include "api.h"
#include "channel.h"
#include "command_handler.h"
#include "debug.h"
#include "endpoint.h"

#include <cuda.h>
#include <nvml.h>
#include <worker.h>
#include "../klib/cuda/cuda_kava.h"
#include "cuda_kava_utilities.h"


#define TBREAKDOWN 1


static struct kava_endpoint __kava_endpoint;

typedef struct {
    /* argument types */
    int func_argc;
    char func_arg_is_handle[64];
    size_t func_arg_size[64];
} Metadata;

struct cu_metadata {
    struct kava_metadata_base base;
    Metadata application;
};

static void __handle_command_cuda_init();
static void __handle_command_cuda_destroy();
void __handle_command_cuda(struct kava_chan *__chan, const struct kava_cmd_base* __cmd);
void __print_command_cuda(FILE* file, const struct kava_chan *__chan,
                        const struct kava_cmd_base* __cmd);

#define kava_metadata(p) (&((struct cu_metadata *)kava_internal_metadata(&__kava_endpoint, p))->application)

void enable_constructor(void) { /*  do nothing */ }

static nvmlDevice_t nvml_dev;

void __attribute__((constructor)) init_cuda_worker(void) {
    nvmlReturn_t ret;

    worker_common_init();
    __handle_command_cuda_init();

    ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "Fail to initialize nvml: %s\n", nvmlErrorString(ret));
        exit(-1);
    }
    ret = nvmlDeviceGetHandleByIndex(0, &nvml_dev);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "Fail to get device by index (0): %s\n", nvmlErrorString(ret));
        exit(-1);
    }
}

void __handle_command_cuda_init()
{
    kava_endpoint_init(&__kava_endpoint, sizeof(struct cu_metadata));
    kava_register_cmd_handler(KAVA_CMD_MODE_API,
                            __handle_command_cuda,
                            __print_command_cuda);
}

void __attribute__((destructor)) fini_cuda_worker(void) {
    worker_common_fini();
}

void __handle_command_cuda_destroy()
{
    kava_endpoint_destroy(&__kava_endpoint);
}

CUdevice device;
CUcontext  context;
CUmodule   module;
CUfunction function;

#define block_size 16
#define WA (4 * block_size) // Matrix A width
#define HA (4 * block_size) // Matrix A height
#define WB (4 * block_size) // Matrix B width
#define HB WA
#define WC WB
#define HC HA

int *a, *b, *c;

static int
__wrapper_cudaGetGPUUtilizationRates(void)
{
    {
        nvmlReturn_t ret;
        nvmlUtilization_t utilization;
        int gpu = 0;
#ifdef KAVA_HAS_GPU
        ret = nvmlDeviceGetUtilizationRates(nvml_dev, &utilization);
        if (ret != NVML_SUCCESS) {
            fprintf(stderr, "Fail to get device utilization rates: %s\n", nvmlErrorString(ret));
        }
        gpu = utilization.gpu;
#endif
        return gpu;
    }
}

static CUresult
__wrapper_cuInit(unsigned int Flags)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuInit(Flags);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuDeviceGet(CUdevice * device, int ordinal)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuDeviceGet(device, ordinal);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuCtxCreate_v2(CUcontext * pctx, unsigned int flags, CUdevice dev)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuCtxCreate_v2(pctx, flags, dev);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuModuleLoad(CUmodule * module, const char *fname)
{
    int res;
#ifdef KAVA_HAS_GPU
    res = cuModuleLoad(module, fname);
#endif
    return res;
}

static CUresult
__wrapper_cuModuleUnload(CUmodule hmod)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuModuleUnload(hmod);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuModuleGetFunction(hfunc, hmod, name);
        kava_parse_function_args(name, &kava_metadata(*hfunc)->func_argc,
                                kava_metadata(*hfunc)->func_arg_is_handle,
                                kava_metadata(*hfunc)->func_arg_size);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **extra, void **kernelParams)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
    #if TBREAKDOWN
        struct timespec start, stop;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    #endif
        ret = cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                             blockDimY, blockDimZ, sharedMemBytes, hStream,
                             kernelParams, extra);
    #if TBREAKDOWN
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        double result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;  
        printf("real_cuLaunchKernel: %f\n", result);
    #endif
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuCtxDestroy_v2(CUcontext ctx)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuCtxDestroy_v2(ctx);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuMemAlloc_v2(CUdeviceptr * dptr, size_t bytesize)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuMemAlloc_v2(dptr, bytesize);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuMemcpyHtoD_v2(CUdeviceptr dstDevice, size_t ByteCount, const void *srcHost)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuMemcpyDtoH_v2(CUdeviceptr srcDevice, size_t ByteCount, void *dstHost)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuCtxSynchronize()
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
    #if TBREAKDOWN
        struct timespec start, stop;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
    #endif
        ret = cuCtxSynchronize();
    #if TBREAKDOWN
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        double result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;  
        printf("__wrapper_cuCtxSynchronize: %f\n", result);
    #endif
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuCtxSetCurrent(CUcontext ctx)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuCtxSetCurrent(ctx);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuDriverGetVersion(int *driverVersion)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuDriverGetVersion(driverVersion);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuMemFree_v2(CUdeviceptr dptr)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuMemFree_v2(dptr);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuModuleGetGlobal_v2(CUdeviceptr * dptr, size_t * bytes, CUmodule hmod, const char *name)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuDeviceGetCount(int *count)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuDeviceGetCount(count);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuFuncSetCacheConfig(hfunc, config);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuStreamCreate(CUstream * phStream, unsigned int Flags)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuStreamCreate(phStream, Flags);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuStreamSynchronize(CUstream hStream)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuStreamSynchronize(hStream);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuStreamDestroy_v2(CUstream hStream)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuStreamDestroy_v2(hStream);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, size_t ByteCount, CUstream hStream, const void *srcHost)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuMemcpyDtoHAsync_v2(CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream, void *dstHost)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
#warning The memory synchronization should be fixed by mmap.
        cuCtxSynchronize();
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuGetErrorString(CUresult error, const char **pStr)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuGetErrorString(error, pStr);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuDeviceGetAttribute(pi, attrib, dev);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuDeviceGetName(int len, CUdevice dev, char *name)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuDeviceGetName(name, len, dev);
#endif
        return ret;
    }
}

static CUresult
__wrapper_cuMemAllocPitch(CUdeviceptr * dptr, size_t *pPitch,
                        size_t WidthInBytes, size_t Height,
                        unsigned int ElementSizeBytes)
{
    {
        CUresult ret;
#ifdef KAVA_HAS_GPU
        ret = cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
#endif
        return ret;
    }
}

void __handle_command_cuda(struct kava_chan* __chan,
                        const struct kava_cmd_base* __cmd)
{
    //__chan->cmd_print(__chan, __cmd);

    switch (__cmd->command_id) {
        case CALL_CUDA___CUDA_TEST_INIT:
        {
            char name[100];

            checkCudaErrors(cuInit(0));
            checkCudaErrors(cuDeviceGet(&device, 0));
            checkCudaErrors(cuDeviceGetName(name, 100, device));
            pr_info("Using device 0: %s\n", name);
            checkCudaErrors(cuCtxCreate(&context, 0, device));
            checkCudaErrors(cuModuleLoad(&module, "./files/matMulKernel.ptx"));
            checkCudaErrors(cuModuleGetFunction(&function, module, "matrixMul"));

            /* Prepare matrix data */
            a = malloc(sizeof(int) * WA * HA);
            b = malloc(sizeof(int) * WB * HB);
            c = malloc(sizeof(int) * WC * HC);
            srand(2019);
            for (int i = 0; i < WA * HA; i++)
                a[i] = rand() % 100;
            for (int i = 0; i < WB * HB; i++)
                b[i] = rand() % 100;
            memset(c, 0, sizeof(int) * WC * HC);

            pr_info("cuTestInit is done\n");
            break;
        }

        case CALL_CUDA___CUDA_TEST_FREE:
            free(a);
            free(b);
            free(c);
            cuCtxDestroy(context);
            break;

        case CALL_CUDA___CUDA_TEST_MMUL:
        {
            struct timeval tv_invoke, tv_h2d_start, tv_h2d_end, tv_d2h_start, tv_d2h_end;
            gettimeofday(&tv_invoke, NULL);

            CUdeviceptr d_a, d_b, d_c;
            cuMemAlloc(&d_a, sizeof(int) * WA * HA);
            cuMemAlloc(&d_b, sizeof(int) * WB * HB);
            cuMemAlloc(&d_c, sizeof(int) * WC * HC);

            gettimeofday(&tv_h2d_start, NULL);

            cuMemcpyHtoD(d_a, a, sizeof(int) * WA * HA);
            cuMemcpyHtoD(d_b, b, sizeof(int) * WB * HB);

            gettimeofday(&tv_h2d_end, NULL);

            size_t mat_width_a = (size_t)WA;
            size_t mat_width_b = (size_t)WB;
            void *args[5] = { &d_c, &d_a, &d_b, &mat_width_a, &mat_width_b };
            checkCudaErrors(cuLaunchKernel(function,
                        WC / block_size, HC / block_size, 1, // Nx1x1 blocks
                        block_size, block_size, 1,           // 1x1x1 threads
                        2 * block_size * block_size * sizeof(int),
                        0, args, 0));
            cuCtxSynchronize();

            gettimeofday(&tv_d2h_start, NULL);

            cuMemcpyDtoH(c, d_c, sizeof(int) * WC * HC);

            gettimeofday(&tv_d2h_end, NULL);

            cuMemFree(d_a);
            cuMemFree(d_b);
            cuMemFree(d_c);

            pr_info("cuMmul is invoked: sec=%lu, usec=%lu\n", tv_invoke.tv_sec, tv_invoke.tv_usec);
            pr_info("cuMmul worker:tot: usec=%lu\n",
                    (tv_d2h_end.tv_sec - tv_invoke.tv_sec) * 1000000 + tv_d2h_end.tv_usec - tv_invoke.tv_usec);
            pr_info("cuMmul worker:memcpy: usec=%lu\n",
                    (tv_d2h_end.tv_sec - tv_d2h_start.tv_sec) * 1000000 + tv_d2h_end.tv_usec - tv_d2h_start.tv_usec +
                    (tv_h2d_end.tv_sec - tv_h2d_start.tv_sec) * 1000000 + tv_h2d_end.tv_usec - tv_h2d_start.tv_usec);
            pr_info("cuMmul worker:kernel: usec=%lu\n",
                    (tv_d2h_start.tv_sec - tv_h2d_end.tv_sec) * 1000000 + tv_d2h_start.tv_usec - tv_h2d_end.tv_usec);
            break;
        }

        case CALL_CUDA___CUDA_TEST_K_TO_U:
        {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            pr_info("enter testKtoU: sec=%lu, usec=%lu\n", tv.tv_sec, tv.tv_usec);
            break;
        }

        case CALL_CUDA___CUDA_TEST_CHANNEL:
        {
            struct cu_cu_test_channel_call *__call = (struct cu_cu_test_channel_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_test_channel_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            size_t __total_buffer_size =
                __call->base.command_size + __call->base.region_size - sizeof(struct cu_cu_test_channel_ret);

            struct cu_cu_test_channel_ret *__ret =
                (struct cu_cu_test_channel_ret *)__chan->cmd_new(__chan, sizeof(struct cu_cu_test_channel_ret),
                __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CUDA_TEST_CHANNEL;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);

            break;
        }

        case CALL_CUDA___CUDA_GET_GPU_UTILIZATION_RATES:
        {
            struct cu_cu_get_gpu_utilization_rates_call *__call = (struct cu_cu_get_gpu_utilization_rates_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_get_gpu_utilization_rates_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Perform Call */
            int ret;
            ret = __wrapper_cudaGetGPUUtilizationRates();

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_get_gpu_utilization_rates_ret *__ret =
                (struct cu_cu_get_gpu_utilization_rates_ret *)__chan->cmd_new(__chan, sizeof(struct cu_cu_get_gpu_utilization_rates_ret),
                __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CUDA_GET_GPU_UTILIZATION_RATES;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);

            break;
        }

        case CALL_CUDA___CU_INIT:
        {
            struct cu_cu_init_call *__call = (struct cu_cu_init_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_init_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: unsigned int Flags */
            unsigned int Flags = (unsigned int)__call->Flags;

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuInit(Flags);

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_init_ret *__ret =
                (struct cu_cu_init_ret *)__chan->cmd_new(__chan, sizeof(struct cu_cu_init_ret),
                __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_INIT;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);

            break;
        }

        case CALL_CUDA___CU_DEVICE_GET:
        {
            GPtrArray *__kava_alloc_list_cuDeviceGet =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_device_get_call *__call = (struct cu_cu_device_get_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_device_get_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUdevice * device */
            CUdevice *device;
            {
                device = ((__call->device) != (NULL)) ?
                         ((CUdevice *)__chan->chan_get_buffer(__chan, __cmd, __call->device)) :
                         ((CUdevice *) __call->device);
                if ((__call->device) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    device = (CUdevice *)malloc(__size * sizeof(CUdevice));
                    g_ptr_array_add(__kava_alloc_list_cuDeviceGet, kava_buffer_with_deallocator_new(free, device));
                }
            }

            /* Input: int ordinal */
            int ordinal; {
                ordinal = (int)__call->ordinal;
                ordinal = __call->ordinal;
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuDeviceGet(device, ordinal);

            size_t __total_buffer_size = 0;
            {
                /* Size: CUdevice * device */
                if ((device) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(CUdevice));
                }
            }
            struct cu_cu_device_get_ret *__ret =
                (struct cu_cu_device_get_ret *)__chan->cmd_new(__chan, sizeof(struct cu_cu_device_get_ret),
                __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_DEVICE_GET;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: CUdevice * device */
            {
                if ((device) != (NULL)) {
                    __ret->device =
                        (CUdevice *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, device,
                        ((size_t) (1)) * sizeof(CUdevice));
                } else {
                    __ret->device = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuDeviceGet);        /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_CTX_CREATE_V2:
        {
            GPtrArray *__kava_alloc_list_cuCtxCreate_v2 =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_ctx_create_v2_call *__call = (struct cu_cu_ctx_create_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_ctx_create_v2_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUcontext * pctx */
            CUcontext *pctx;
            {
                pctx = ((__call->pctx) != (NULL)) ?
                       ((CUcontext *)__chan->chan_get_buffer(__chan, __cmd, __call->pctx)) :
                       ((CUcontext *) __call->pctx);
                if ((__call->pctx) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    pctx = (CUcontext *)malloc(__size * sizeof(CUcontext));
                    g_ptr_array_add(__kava_alloc_list_cuCtxCreate_v2, kava_buffer_with_deallocator_new(free, pctx));
                }
            }

            /* Input: unsigned int flags */
            unsigned int flags;
            {
                flags = (unsigned int)__call->flags;
            }

            /* Input: CUdevice dev */
            CUdevice dev;
            {
                dev = (CUdevice) __call->dev;
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuCtxCreate_v2(pctx, flags, dev);

            size_t __total_buffer_size = 0;
            {
                /* Size: CUcontext * pctx */
                if ((pctx) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(CUcontext));
                }
            }
            struct cu_cu_ctx_create_v2_ret *__ret =
                (struct cu_cu_ctx_create_v2_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_ctx_create_v2_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_CTX_CREATE_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: CUcontext * pctx */
            {
                if ((pctx) != (NULL)) {
                    const size_t __size_pctx_0 = ((size_t) (1));
                    CUcontext *__tmp_pctx_0;
                    __tmp_pctx_0 = (CUcontext *) calloc(1, __size_pctx_0 * sizeof(CUcontext));
                    g_ptr_array_add(__kava_alloc_list_cuCtxCreate_v2,
                                    kava_buffer_with_deallocator_new(free, __tmp_pctx_0));
                    const size_t __pctx_size_0 = __size_pctx_0;
                    for (size_t __pctx_index_0 = 0; __pctx_index_0 < __pctx_size_0; __pctx_index_0++) {
                        CUcontext *__pctx_a_0;
                        __pctx_a_0 = (CUcontext *) (__tmp_pctx_0) + __pctx_index_0;

                        CUcontext *__pctx_b_0;
                        __pctx_b_0 = (CUcontext *) (pctx) + __pctx_index_0;

                        {
                            *__pctx_a_0 = (CUcontext)*__pctx_b_0;
                        }
                    }
                    __ret->pctx =
                        (CUcontext *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, __tmp_pctx_0,
                        ((size_t) (1)) * sizeof(CUcontext));
                } else {
                    __ret->pctx = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuCtxCreate_v2);     /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MODULE_LOAD:
        {
            GPtrArray *__kava_alloc_list_cuModuleLoad =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_module_load_call *__call = (struct cu_cu_module_load_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_module_load_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUmodule * module */
            CUmodule *module;
            {
                module = ((__call->module) != (NULL)) ?
                         ((CUmodule *)__chan->chan_get_buffer(__chan, __cmd, __call->module)) :
                         ((CUmodule *) __call->module);
                if ((__call->module) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    module = (CUmodule *) malloc(__size * sizeof(CUmodule));
                    g_ptr_array_add(__kava_alloc_list_cuModuleLoad,
                                    kava_buffer_with_deallocator_new(free, module));
                }
            }

            /* Input: const char * fname */
            const char *fname;
            {
                fname = ((__call->fname) != (NULL)) ?
                        ((const char *)__chan->chan_get_buffer(__chan, __cmd, __call->fname)) :
                        ((const char *)__call->fname);
                if ((__call->fname) != (NULL)) {
                    const char *__src_fname_0 = fname;
                    volatile size_t __buffer_size = ((size_t) (strlen(fname) + 1));
                    fname = (const char *)__chan->chan_get_buffer(__chan, __cmd, __call->fname);

                    if ((fname) != (__src_fname_0)) {
                        memcpy((void *)fname, __src_fname_0, __buffer_size * sizeof(const char));
                    }
                }
                else {
                    fname = ((__call->fname) != (NULL)) ?
                            ((const char *)__chan->chan_get_buffer(__chan, __cmd, __call->fname)) :
                            ((const char *)__call->fname);
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuModuleLoad(module, fname);

            size_t __total_buffer_size = 0;
            {
                /* Size: CUmodule * module */
                if ((module) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(CUmodule));
                }
            }
            struct cu_cu_module_load_ret *__ret =
                (struct cu_cu_module_load_ret *)__chan->cmd_new(__chan, sizeof(struct cu_cu_module_load_ret),
                __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MODULE_LOAD;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: CUmodule * module */
            {
                if ((module) != (NULL)) {
                    const size_t __size_module_0 = ((size_t) (1));
                    CUmodule *__tmp_module_0 = (CUmodule *) calloc(1, __size_module_0 * sizeof(CUmodule));
                    g_ptr_array_add(__kava_alloc_list_cuModuleLoad,
                                    kava_buffer_with_deallocator_new(free, __tmp_module_0));
                    const size_t __module_size_0 = __size_module_0;
                    for (size_t __module_index_0 = 0; __module_index_0 < __module_size_0; __module_index_0++) {
                        CUmodule *__module_a_0;
                        __module_a_0 = (CUmodule *) (__tmp_module_0) + __module_index_0;

                        CUmodule *__module_b_0;
                        __module_b_0 = (CUmodule *) (module) + __module_index_0;

                        {
                            *__module_a_0 = (CUmodule)*__module_b_0;
                        }
                    }
                    __ret->module = (CUmodule *)__chan->chan_attach_buffer(__chan,
                            (struct kava_cmd_base *)__ret, __tmp_module_0,
                            ((size_t) (1)) * sizeof(CUmodule));
                }
                else {
                    __ret->module = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuModuleLoad);       /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MODULE_UNLOAD:
        {
            GPtrArray *__kava_alloc_list_cuModuleUnload =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_module_unload_call *__call = (struct cu_cu_module_unload_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_module_unload_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUmodule hmod */
            CUmodule hmod = (CUmodule)__call->hmod;

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuModuleUnload(hmod);

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_module_unload_ret *__ret =
                (struct cu_cu_module_unload_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_module_unload_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MODULE_UNLOAD;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuModuleUnload);     /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MODULE_GET_FUNCTION:
        {
            GPtrArray *__kava_alloc_list_cuModuleGetFunction =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_module_get_function_call *__call = (struct cu_cu_module_get_function_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_module_get_function_call)
                &&
                "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUfunction * hfunc */
            CUfunction *hfunc;
            {
                hfunc = ((__call->hfunc) != (NULL)) ?
                        ((CUfunction *)__chan->chan_get_buffer(__chan, __cmd, __call->hfunc)) :
                        ((CUfunction *) __call->hfunc);
                if ((__call->hfunc) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    hfunc = (CUfunction *) malloc(__size * sizeof(CUfunction));
                    g_ptr_array_add(__kava_alloc_list_cuModuleGetFunction,
                                    kava_buffer_with_deallocator_new(free, hfunc));
                }
            }

            /* Input: CUmodule hmod */
            CUmodule hmod = (CUmodule)__call->hmod;

            /* Input: const char * name */
            const char *name;
            {
                name = ((__call->name) != (NULL)) ?
                       ((const char *)__chan->chan_get_buffer(__chan, __cmd, __call->name)) :
                       ((const char *)__call->name);
                if ((__call->name) != (NULL)) {
                    const char *__src_name_0 = name;
                    volatile size_t __buffer_size = ((size_t) (strlen(name) + 1));
                    name = (const char *)__chan->chan_get_buffer(__chan, __cmd, __call->name);

                    if ((name) != (__src_name_0)) {
                        memcpy((void *)name, __src_name_0, __buffer_size * sizeof(const char));
                    }
                }
                else {
                    name = ((__call->name) != (NULL)) ?
                           ((const char *)__chan->chan_get_buffer(__chan, __cmd, __call->name)) :
                           ((const char *)__call->name);
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuModuleGetFunction(hfunc, hmod, name);

            size_t __total_buffer_size = 0;
            {
                /* Size: CUfunction * hfunc */
                if ((hfunc) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(CUfunction));
                }
            }
            struct cu_cu_module_get_function_ret *__ret =
                (struct cu_cu_module_get_function_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_module_get_function_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MODULE_GET_FUNCTION;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: CUfunction * hfunc */
            {
                if ((hfunc) != (NULL)) {
                    const size_t __size_hfunc_0 = ((size_t) (1));
                    CUfunction *__tmp_hfunc_0;
                    __tmp_hfunc_0 = (CUfunction *) calloc(1, __size_hfunc_0 * sizeof(CUfunction));
                    g_ptr_array_add(__kava_alloc_list_cuModuleGetFunction,
                                    kava_buffer_with_deallocator_new(free, __tmp_hfunc_0));
                    const size_t __hfunc_size_0 = __size_hfunc_0;
                    for (size_t __hfunc_index_0 = 0; __hfunc_index_0 < __hfunc_size_0; __hfunc_index_0++) {
                        CUfunction *__hfunc_a_0;
                        __hfunc_a_0 = (CUfunction *) (__tmp_hfunc_0) + __hfunc_index_0;

                        CUfunction *__hfunc_b_0;
                        __hfunc_b_0 = (CUfunction *) (hfunc) + __hfunc_index_0;

                        {
                            *__hfunc_a_0 = (CUfunction)*__hfunc_b_0;
                        }
                    }
                    __ret->hfunc =
                        (CUfunction *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_hfunc_0, ((size_t) (1)) * sizeof(CUfunction));
                }
                else {
                    __ret->hfunc = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuModuleGetFunction);        /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_LAUNCH_KERNEL:
        {
            #if TBREAKDOWN
            struct timespec start, stop, p1, p2, p3;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            #endif 

            GPtrArray *__kava_alloc_list_cuLaunchKernel =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_launch_kernel_call *__call = (struct cu_cu_launch_kernel_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_launch_kernel_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUfunction f */
            CUfunction f = (CUfunction)__call->f;

            /* Input: unsigned int gridDimX */
            unsigned int gridDimX;
            {
                gridDimX = (unsigned int)__call->gridDimX;
                gridDimX = __call->gridDimX;
            }

            /* Input: unsigned int gridDimY */
            unsigned int gridDimY;
            {
                gridDimY = (unsigned int)__call->gridDimY;
                gridDimY = __call->gridDimY;
            }

            /* Input: unsigned int gridDimZ */
            unsigned int gridDimZ;
            {
                gridDimZ = (unsigned int)__call->gridDimZ;
                gridDimZ = __call->gridDimZ;
            }

            /* Input: unsigned int blockDimX */
            unsigned int blockDimX;
            {
                blockDimX = (unsigned int)__call->blockDimX;
                blockDimX = __call->blockDimX;
            }

            /* Input: unsigned int blockDimY */
            unsigned int blockDimY;
            {
                blockDimY = (unsigned int)__call->blockDimY;
                blockDimY = __call->blockDimY;
            }

            /* Input: unsigned int blockDimZ */
            unsigned int blockDimZ;
            {
                blockDimZ = (unsigned int)__call->blockDimZ;
                blockDimZ = __call->blockDimZ;
            }

            /* Input: unsigned int sharedMemBytes */
            unsigned int sharedMemBytes;
            {
                sharedMemBytes = (unsigned int)__call->sharedMemBytes;
                sharedMemBytes = __call->sharedMemBytes;
            }

            /* Input: CUstream hStream */
            CUstream hStream = (CUstream)__call->hStream;

            /* Input: void ** extra */
            void **extra;
            {
                extra = ((__call->extra) != (NULL)) ?
                        ((void **)__chan->chan_get_buffer(__chan, __cmd, __call->extra)) :
                        ((void **)__call->extra);
                if ((__call->extra) != (NULL)) {
                    void **__src_extra_0 = extra;
                    volatile size_t __buffer_size = ((size_t) (cuLaunchKernel_extra_size(extra)));
                    extra = (void **)__chan->chan_get_buffer(__chan, __cmd, __call->extra);
                    if ((__call->extra) != (NULL)) {
                        const size_t __size = ((size_t) (cuLaunchKernel_extra_size(extra)));
                        extra = (void **)malloc(__size * sizeof(void *));
                        g_ptr_array_add(__kava_alloc_list_cuLaunchKernel,
                                        kava_buffer_with_deallocator_new(free, extra));
                    }

                    const size_t __extra_size_0 = __buffer_size;
                    for (size_t __extra_index_0 = 0; __extra_index_0 < __extra_size_0; __extra_index_0++) {
                        void **__extra_a_0;
                        __extra_a_0 = (void **)(extra) + __extra_index_0;

                        void **__extra_b_0;
                        __extra_b_0 = (void **)(__src_extra_0) + __extra_index_0;

                        {
                            *__extra_a_0 = ((*__extra_b_0) != (NULL)) ?
                                           ((void *)__chan->chan_get_buffer(__chan, __cmd, *__extra_b_0)) :
                                           ((void *)*__extra_b_0);
                            if ((*__extra_b_0) != (NULL)) {
                                void *__src_extra_1 = *__extra_a_0;
                                volatile size_t __buffer_size = ((size_t) (1));
                                *__extra_a_0 = (void *)__chan->chan_get_buffer(__chan, __cmd, *__extra_b_0);

                                if ((*__extra_a_0) != (__src_extra_1)) {
                                    memcpy(*__extra_a_0, __src_extra_1, __buffer_size * sizeof(void));
                                }
                            }
                            else {
                                *__extra_a_0 = ((*__extra_b_0) != (NULL)) ?
                                               ((void *)__chan->chan_get_buffer(__chan, __cmd, *__extra_b_0)) :
                                               ((void *)*__extra_b_0);
                            }
                        }
                    }
                }
                else {
                    if ((__call->extra) != (NULL)) {
                        const size_t __size = ((size_t) (cuLaunchKernel_extra_size(extra)));
                        extra = (void **)malloc(__size * sizeof(void *));
                        g_ptr_array_add(__kava_alloc_list_cuLaunchKernel,
                                        kava_buffer_with_deallocator_new(free, extra));
                    }
                }
            }

            /* Input: void ** kernelParams */
            #if TBREAKDOWN
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &p1);
            #endif

            void **kernelParams;
            {
                kernelParams = ((__call->kernelParams) != (NULL)) ?
                               ((void **)__chan->chan_get_buffer(__chan, __cmd, __call->kernelParams)) :
                               ((void **)__call->kernelParams);
                if ((__call->kernelParams) != (NULL)) {
                    void **__src_kernelParams_0 = kernelParams;
                    volatile size_t __buffer_size = ((size_t) (kava_metadata(f)->func_argc));
                    kernelParams = (void **)__chan->chan_get_buffer(__chan, __cmd, __call->kernelParams);
                    if ((__call->kernelParams) != (NULL)) {
                        const size_t __size = ((size_t) (kava_metadata(f)->func_argc));
                        kernelParams = (void **)malloc(__size * sizeof(void *));
                        g_ptr_array_add(__kava_alloc_list_cuLaunchKernel,
                                        kava_buffer_with_deallocator_new(free, kernelParams));
                    }

                    const size_t __kernelParams_size_0 = __buffer_size;
                    for (size_t __kernelParams_index_0 = 0; __kernelParams_index_0 < __kernelParams_size_0;
                        __kernelParams_index_0++) {
                        const size_t kava_index = __kernelParams_index_0;

                        void **__kernelParams_a_0 = (void **)(kernelParams) + __kernelParams_index_0;
                        void **__kernelParams_b_0 = (void **)(__src_kernelParams_0) + __kernelParams_index_0;

                        *__kernelParams_a_0 = ((*__kernelParams_b_0) != (NULL)) ?
                                              ((void *)__chan->chan_get_buffer(__chan, __cmd, *__kernelParams_b_0)) :
                                              ((void *)*__kernelParams_b_0);
                        if (kava_metadata(f)->func_arg_is_handle[kava_index]) {
                            *__kernelParams_a_0 =
                                ((*__kernelParams_b_0) != (NULL)) ? ((CUdeviceptr *)__chan->chan_get_buffer(__chan,
                                    __cmd, *__kernelParams_b_0)) : ((CUdeviceptr *) * __kernelParams_b_0);
                            if ((*__kernelParams_b_0) != (NULL)) {
                                CUdeviceptr *__src_kernelParams_1;
                                __src_kernelParams_1 = *__kernelParams_a_0;
                                volatile size_t __buffer_size = sizeof(void);
                                *__kernelParams_a_0 =
                                    (CUdeviceptr *)__chan->chan_get_buffer(__chan, __cmd, *__kernelParams_b_0);

                                if ((*__kernelParams_a_0) != (__src_kernelParams_1)) {
                                    memcpy(*__kernelParams_a_0, __src_kernelParams_1,
                                        __buffer_size * sizeof(CUdeviceptr));
                                }
                            } else {
                                *__kernelParams_a_0 =
                                    ((*__kernelParams_b_0) !=
                                    (NULL)) ? ((CUdeviceptr *)__chan->chan_get_buffer(__chan, __cmd,
                                        *__kernelParams_b_0)) : ((CUdeviceptr *) * __kernelParams_b_0);
                            }
                        }
                        else {
                            *__kernelParams_a_0 =
                                ((*__kernelParams_b_0) != (NULL)) ? ((int *)__chan->chan_get_buffer(__chan, __cmd,
                                    *__kernelParams_b_0)) : ((int *)*__kernelParams_b_0);
                            if ((*__kernelParams_b_0) != (NULL)) {
                                int *__src_kernelParams_1;
                                __src_kernelParams_1 = *__kernelParams_a_0;
                                volatile size_t __buffer_size = sizeof(void);
                                *__kernelParams_a_0 =
                                    (int *)__chan->chan_get_buffer(__chan, __cmd, *__kernelParams_b_0);

                                if ((*__kernelParams_a_0) != (__src_kernelParams_1)) {
                                    memcpy(*__kernelParams_a_0, __src_kernelParams_1, __buffer_size * kava_metadata(f)->func_arg_size[kava_index]);
                                }
                            } else {
                                *__kernelParams_a_0 =
                                    ((*__kernelParams_b_0) != (NULL)) ? ((int *)__chan->chan_get_buffer(__chan,
                                        __cmd, *__kernelParams_b_0)) : ((int *)*__kernelParams_b_0);
                            }
                        }
                    }
                } else {
                    if ((__call->kernelParams) != (NULL)) {
                        const size_t __size = ((size_t) (kava_metadata(f)->func_argc));
                        kernelParams = (void **)malloc(__size * sizeof(void *));
                        g_ptr_array_add(__kava_alloc_list_cuLaunchKernel,
                                        kava_buffer_with_deallocator_new(free, kernelParams));
                    }
                }
            }

            #if TBREAKDOWN
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &p2);
            #endif

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                           blockDimX, blockDimY, blockDimZ,
                                           sharedMemBytes, hStream,
                                           extra, kernelParams);

            #if TBREAKDOWN
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &p3);
            #endif

            size_t __total_buffer_size = 0;

            struct cu_cu_launch_kernel_ret *__ret =
                (struct cu_cu_launch_kernel_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_launch_kernel_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_LAUNCH_KERNEL;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuLaunchKernel);     /* Deallocate all memory in the alloc list */

            #if TBREAKDOWN
                clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
                double result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;  
                double rp = (p2.tv_sec - p1.tv_sec) * 1e6 + (p2.tv_nsec - p1.tv_nsec) / 1e3;  
                double ae = (p1.tv_sec - start.tv_sec) * 1e6 + (p1.tv_nsec - start.tv_nsec) / 1e3;  
                double after = (stop.tv_sec - p3.tv_sec) * 1e6 + (stop.tv_nsec - p3.tv_nsec) / 1e3;  
                double kl = (p3.tv_sec - p2.tv_sec) * 1e6 + (p3.tv_nsec - p2.tv_nsec) / 1e3;
                printf("kernel args      : %f\n", rp);
                printf("array+extra      : %f\n", ae);
                printf("wrap_kernel      : %f\n", kl);
                printf("ret+sendmsg      : %f\n", after);
                printf("_CU_LAUNCH_KERNEL: %f\n", result);
            #endif

            break;
        }

        case CALL_CUDA___CU_CTX_DESTROY_V2:
        {
            GPtrArray *__kava_alloc_list_cuCtxDestroy_v2 =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_ctx_destroy_v2_call *__call = (struct cu_cu_ctx_destroy_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_ctx_destroy_v2_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUcontext ctx */
            CUcontext ctx = (CUcontext)__call->ctx;

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuCtxDestroy_v2(ctx);

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_ctx_destroy_v2_ret *__ret =
                (struct cu_cu_ctx_destroy_v2_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_ctx_destroy_v2_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_CTX_DESTROY_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuCtxDestroy_v2);    /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MEM_ALLOC_V2:
        {
            GPtrArray *__kava_alloc_list_cuMemAlloc_v2 =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_mem_alloc_v2_call *__call = (struct cu_cu_mem_alloc_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_mem_alloc_v2_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUdeviceptr * dptr */
            CUdeviceptr *dptr;
            {
                dptr = ((__call->dptr) != (NULL)) ?
                       ((CUdeviceptr *)__chan->chan_get_buffer(__chan, __cmd, __call->dptr)) :
                       ((CUdeviceptr *) __call->dptr);
                if ((__call->dptr) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    dptr = (CUdeviceptr *)malloc(__size * sizeof(CUdeviceptr));
                    g_ptr_array_add(__kava_alloc_list_cuMemAlloc_v2,
                                    kava_buffer_with_deallocator_new(free, dptr));
                }
            }

            /* Input: size_t bytesize */
            size_t bytesize;
            {
                bytesize = (size_t) __call->bytesize;
                bytesize = __call->bytesize;
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuMemAlloc_v2(dptr, bytesize);

            size_t __total_buffer_size = 0;
            {
                /* Size: CUdeviceptr * dptr */
                if ((dptr) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(CUdeviceptr));
                }
            }
            struct cu_cu_mem_alloc_v2_ret *__ret =
                (struct cu_cu_mem_alloc_v2_ret *)__chan->cmd_new(__chan, sizeof(struct cu_cu_mem_alloc_v2_ret),
                __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MEM_ALLOC_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: CUdeviceptr * dptr */
            {
                if ((dptr) != (NULL)) {
                    __ret->dptr = (CUdeviceptr *)__chan->chan_attach_buffer(
                            __chan, (struct kava_cmd_base *)__ret,
                            dptr, ((size_t) (1)) * sizeof(CUdeviceptr));
                } else {
                    __ret->dptr = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuMemAlloc_v2);      /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MEMCPY_HTO_D_V2:
        {
            GPtrArray *__kava_alloc_list_cuMemcpyHtoD_v2 =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_memcpy_hto_d_v2_call *__call = (struct cu_cu_memcpy_hto_d_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_memcpy_hto_d_v2_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUdeviceptr dstDevice */
            CUdeviceptr dstDevice;
            {
                dstDevice = (CUdeviceptr) __call->dstDevice;
                dstDevice = __call->dstDevice;
            }

            /* Input: size_t ByteCount */
            size_t ByteCount;
            {
                ByteCount = (size_t) __call->ByteCount;
                ByteCount = __call->ByteCount;
            }

            /* Input: const void * srcHost */
            const void *srcHost;
            {
                if ((__call->srcHost) != (NULL)) {
                    if ((__call->__shm_srcHost)) {
                        srcHost = kava_shm_address((long)__call->srcHost);
                    }
                    else {
                        srcHost = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->srcHost);
                    }
                }
                else {
                    srcHost = NULL;
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuMemcpyHtoD_v2(dstDevice, ByteCount, srcHost);

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_memcpy_hto_d_v2_ret *__ret =
                (struct cu_cu_memcpy_hto_d_v2_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_memcpy_hto_d_v2_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MEMCPY_HTO_D_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuMemcpyHtoD_v2);    /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MEMCPY_DTO_H_V2:
        {
            GPtrArray *__kava_alloc_list_cuMemcpyDtoH_v2 =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_memcpy_dto_h_v2_call *__call = (struct cu_cu_memcpy_dto_h_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_memcpy_dto_h_v2_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUdeviceptr srcDevice */
            CUdeviceptr srcDevice;
            {
                srcDevice = (CUdeviceptr) __call->srcDevice;
            }

            /* Input: size_t ByteCount */
            size_t ByteCount;
            {
                ByteCount = (size_t) __call->ByteCount;
            }

            /* Input: void * dstHost */
            void *dstHost;
            {
                if ((__call->dstHost) != (NULL)) {
                    if ((__call->__shm_dstHost)) {
                        dstHost = kava_shm_address((long)__call->dstHost);
                    }
                    else {
                        dstHost = (void *)malloc(ByteCount * sizeof(void));
                        g_ptr_array_add(__kava_alloc_list_cuMemcpyDtoH_v2,
                                        kava_buffer_with_deallocator_new(free, dstHost));
                    }
                }
                else {
                    dstHost = NULL;
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuMemcpyDtoH_v2(srcDevice, ByteCount, dstHost);

            size_t __total_buffer_size = 0;
            {
                /* Size: void * dstHost */
                if ((dstHost) != (NULL) && (ByteCount) > (0)) {
                    if (__call->__shm_dstHost) {
                    }
                    else {
                        __total_buffer_size += __chan->chan_buffer_size(__chan, ByteCount * sizeof(void));
                    }
                }
            }
            struct cu_cu_memcpy_dto_h_v2_ret *__ret =
                (struct cu_cu_memcpy_dto_h_v2_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_memcpy_dto_h_v2_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MEMCPY_DTO_H_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: void * dstHost */
            {
                if ((dstHost) != (NULL) && (ByteCount) > (0)) {
                    if (__call->__shm_dstHost) {
                        __ret->dstHost = __call->dstHost;
                    }
                    else {
                        __ret->dstHost = (void *)__chan->chan_attach_buffer(__chan,
                                (struct kava_cmd_base *)__ret, dstHost,
                                ((size_t) (ByteCount)) * sizeof(void));
                    }
                } else {
                    __ret->dstHost = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuMemcpyDtoH_v2);    /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_CTX_SYNCHRONIZE:
        {
            #if TBREAKDOWN
            struct timespec start, stop;
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
            #endif

            GPtrArray *__kava_alloc_list_cuCtxSynchronize =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_ctx_synchronize_call *__call = (struct cu_cu_ctx_synchronize_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_ctx_synchronize_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuCtxSynchronize();

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_ctx_synchronize_ret *__ret =
                (struct cu_cu_ctx_synchronize_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_ctx_synchronize_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_CTX_SYNCHRONIZE;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuCtxSynchronize);   /* Deallocate all memory in the alloc list */

            #if TBREAKDOWN
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
            double result = (stop.tv_sec - start.tv_sec) * 1e6 + (stop.tv_nsec - start.tv_nsec) / 1e3;  
            printf("CALL_CUDA___CU_CTX_SYNCHRONIZE: %f\n", result);
            #endif
            
            break;
        }

        case CALL_CUDA___CU_CTX_SET_CURRENT:
        {
            struct cu_cu_ctx_set_current_call *__call = (struct cu_cu_ctx_set_current_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_ctx_set_current_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUcontext ctx */
            CUcontext ctx = (CUcontext)__call->ctx;

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuCtxSetCurrent(ctx);

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_ctx_set_current_ret *__ret =
                (struct cu_cu_ctx_set_current_ret *)__chan->cmd_new(__chan,
                        sizeof(struct cu_cu_ctx_set_current_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_CTX_SET_CURRENT;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);

            break;
        }

        case CALL_CUDA___CU_DRIVER_GET_VERSION:
        {
            GPtrArray *__kava_alloc_list_cuDriverGetVersion =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_driver_get_version_call *__call = (struct cu_cu_driver_get_version_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_driver_get_version_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: int * driverVersion */
            int *driverVersion;
            {
                driverVersion = ((__call->driverVersion) != (NULL)) ?
                                ((int *)__chan->chan_get_buffer(__chan, __cmd, __call->driverVersion)) :
                                ((int *)__call->driverVersion);
                if ((__call->driverVersion) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    driverVersion = (int *)malloc(__size * sizeof(int));
                    g_ptr_array_add(__kava_alloc_list_cuDriverGetVersion,
                                    kava_buffer_with_deallocator_new(free, driverVersion));
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuDriverGetVersion(driverVersion);

            size_t __total_buffer_size = 0;
            {
                /* Size: int * driverVersion */
                if ((driverVersion) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(int));
                }
            }
            struct cu_cu_driver_get_version_ret *__ret =
                (struct cu_cu_driver_get_version_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_driver_get_version_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_DRIVER_GET_VERSION;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: int * driverVersion */
            {
                if ((driverVersion) != (NULL)) {
                    __ret->driverVersion = (int *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, driverVersion,
                        ((size_t) (1)) * sizeof(int));
                } else {
                    __ret->driverVersion = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuDriverGetVersion); /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MEM_FREE_V2:
        {
            GPtrArray *__kava_alloc_list_cuMemFree_v2 =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_mem_free_v2_call *__call = (struct cu_cu_mem_free_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_mem_free_v2_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUdeviceptr dptr */
            CUdeviceptr dptr;
            {
                dptr = (CUdeviceptr) __call->dptr;
                dptr = __call->dptr;
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuMemFree_v2(dptr);

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_mem_free_v2_ret *__ret =
                (struct cu_cu_mem_free_v2_ret *)__chan->cmd_new(__chan, sizeof(struct cu_cu_mem_free_v2_ret),
                __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MEM_FREE_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuMemFree_v2);       /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MODULE_GET_GLOBAL_V2:
        {
            GPtrArray *__kava_alloc_list_cuModuleGetGlobal_v2 =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_module_get_global_v2_call *__call = (struct cu_cu_module_get_global_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_module_get_global_v2_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUdeviceptr * dptr */
            CUdeviceptr *dptr;
            {
                dptr = ((__call->dptr) != (NULL)) ? ((CUdeviceptr *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->dptr)) : ((CUdeviceptr *) __call->dptr);
                if ((__call->dptr) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    dptr = (CUdeviceptr *) malloc(__size * sizeof(CUdeviceptr));
                    g_ptr_array_add(__kava_alloc_list_cuModuleGetGlobal_v2,
                                    kava_buffer_with_deallocator_new(free, dptr));
                }
            }

            /* Input: size_t * bytes */
            size_t *bytes;
            {
                bytes = ((__call->bytes) != (NULL)) ?
                        ((size_t *)__chan->chan_get_buffer(__chan, __cmd, __call->bytes)) :
                        ((size_t *)__call->bytes);
                if ((__call->bytes) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    bytes = (size_t *) malloc(__size * sizeof(size_t));
                    g_ptr_array_add(__kava_alloc_list_cuModuleGetGlobal_v2,
                                    kava_buffer_with_deallocator_new(free, bytes));
                }
            }

            /* Input: CUmodule hmod */
            CUmodule hmod = (CUmodule)__call->hmod;

            /* Input: const char * name */
            const char *name;
            {
                name = ((__call->name) != (NULL)) ?
                       ((const char *)__chan->chan_get_buffer(__chan, __cmd, __call->name)) :
                       ((const char *)__call->name);
                if ((__call->name) != (NULL)) {
                    const char *__src_name_0 = name;
                    volatile size_t __buffer_size = ((size_t) (strlen(name) + 1));
                    name = (const char *)__chan->chan_get_buffer(__chan, __cmd, __call->name);

                    if ((name) != (__src_name_0)) {
                        memcpy((void *)name, __src_name_0, __buffer_size * sizeof(const char));
                    }
                }
                else {
                    name = ((__call->name) != (NULL)) ?
                           ((const char *)__chan->chan_get_buffer(__chan, __cmd, __call->name)) :
                           ((const char *)__call->name);
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuModuleGetGlobal_v2(dptr, bytes, hmod, name);

            size_t __total_buffer_size = 0;
            {
                /* Size: CUdeviceptr * dptr */
                if ((dptr) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(CUdeviceptr));
                }

                /* Size: size_t * bytes */
                if ((bytes) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(size_t));
                }
            }
            struct cu_cu_module_get_global_v2_ret *__ret =
                (struct cu_cu_module_get_global_v2_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_module_get_global_v2_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MODULE_GET_GLOBAL_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: CUdeviceptr * dptr */
            {
                if ((dptr) != (NULL)) {
                    __ret->dptr =
                        (CUdeviceptr *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, dptr,
                        ((size_t) (1)) * sizeof(CUdeviceptr));
                } else {
                    __ret->dptr = NULL;
                }
            }
    /* Output: size_t * bytes */
            {
                if ((bytes) != (NULL)) {
                    __ret->bytes =
                        (size_t *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, bytes,
                        ((size_t) (1)) * sizeof(size_t));
                } else {
                    __ret->bytes = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuModuleGetGlobal_v2);       /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_DEVICE_GET_COUNT:
        {
            GPtrArray *__kava_alloc_list_cuDeviceGetCount =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_device_get_count_call *__call = (struct cu_cu_device_get_count_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_device_get_count_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: int * count */
            int *count;
            {
                count = ((__call->count) != (NULL)) ?
                        ((int *)__chan->chan_get_buffer(__chan, __cmd, __call->count)) :
                        ((int *)__call->count);
                if ((__call->count) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    count = (int *)malloc(__size * sizeof(int));
                    g_ptr_array_add(__kava_alloc_list_cuDeviceGetCount,
                                    kava_buffer_with_deallocator_new(free, count));
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuDeviceGetCount(count);

            size_t __total_buffer_size = 0;
            {
                /* Size: int * count */
                if ((count) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(int));
                }
            }
            struct cu_cu_device_get_count_ret *__ret =
                (struct cu_cu_device_get_count_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_device_get_count_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_DEVICE_GET_COUNT;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: int * count */
            {
                if ((count) != (NULL)) {
                    __ret->count = (int *)__chan->chan_attach_buffer(__chan,
                            (struct kava_cmd_base *)__ret, count, ((size_t) (1)) * sizeof(int));
                } else {
                    __ret->count = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuDeviceGetCount);   /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_FUNC_SET_CACHE_CONFIG:
        {
            GPtrArray *__kava_alloc_list_cuFuncSetCacheConfig =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_func_set_cache_config_call *__call = (struct cu_cu_func_set_cache_config_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_func_set_cache_config_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */
            CUfunction hfunc = (CUfunction)__call->hfunc;
            CUfunc_cache config = __call->config;

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuFuncSetCacheConfig(hfunc, config);

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_func_set_cache_config_ret *__ret =
                (struct cu_cu_func_set_cache_config_ret *)__chan->cmd_new(__chan,
                sizeof(struct cu_cu_func_set_cache_config_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_FUNC_SET_CACHE_CONFIG;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuFuncSetCacheConfig);   /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_STREAM_CREATE:
        {
            GPtrArray *__kava_alloc_list_cuStreamCreate =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_stream_create_call *__call = (struct cu_cu_stream_create_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_stream_create_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUstream * phStream */
            CUstream *phStream;
            {
                phStream = ((__call->phStream) != (NULL)) ?
                    ((CUstream *)__chan->chan_get_buffer(__chan, __cmd, __call->phStream)) :
                    ((CUstream *) __call->phStream);
                if ((__call->phStream) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    phStream = (CUstream *)malloc(__size * sizeof(CUstream));
                    g_ptr_array_add(__kava_alloc_list_cuStreamCreate,
                                    kava_buffer_with_deallocator_new(free, phStream));
                }
            }

            /* Input: unsigned int Flags */
            unsigned int Flags;
            {
                Flags = (unsigned int)__call->Flags;
                Flags = __call->Flags;
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuStreamCreate(phStream, Flags);

            size_t __total_buffer_size = 0;
            {
                /* Size: CUstream * phStream */
                if ((phStream) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(CUstream));
                }
            }
            struct cu_cu_stream_create_ret *__ret =
                (struct cu_cu_stream_create_ret *)__chan->cmd_new(__chan,
                        sizeof(struct cu_cu_stream_create_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_STREAM_CREATE;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: CUstream * phStream */
            {
                if ((phStream) != (NULL)) {
                    __ret->phStream = (CUstream *)__chan->chan_attach_buffer(
                            __chan, (struct kava_cmd_base *)__ret,
                            phStream, ((size_t) (1)) * sizeof(CUstream));
                } else {
                    __ret->phStream = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuStreamCreate);     /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_STREAM_SYNCHRONIZE:
        {
            struct cu_cu_stream_synchronize_call *__call = (struct cu_cu_stream_synchronize_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_stream_synchronize_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUstream hStream */
            CUstream hStream;
            {
                hStream = (CUstream) __call->hStream;
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuStreamSynchronize(hStream);

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_stream_synchronize_ret *__ret =
                (struct cu_cu_stream_synchronize_ret *)__chan->cmd_new(__chan,
                        sizeof(struct cu_cu_stream_synchronize_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_STREAM_SYNCHRONIZE;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);

            break;
        }

        case CALL_CUDA___CU_STREAM_DESTROY_V2:
        {
            GPtrArray *__kava_alloc_list_cuStreamDestroy_v2 =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_stream_destroy_v2_call *__call = (struct cu_cu_stream_destroy_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_stream_destroy_v2_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUstream hStream */
            CUstream hStream;
            {
                hStream = (CUstream) __call->hStream;
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuStreamDestroy_v2(hStream);

            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_stream_destroy_v2_ret *__ret =
                (struct cu_cu_stream_destroy_v2_ret *)__chan->cmd_new(__chan,
                        sizeof(struct cu_cu_stream_destroy_v2_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_STREAM_DESTROY_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuStreamDestroy_v2); /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MEMCPY_HTO_D_ASYNC_V2:
        {
            struct cu_cu_memcpy_hto_d_async_v2_call *__call = (struct cu_cu_memcpy_hto_d_async_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_memcpy_hto_d_async_v2_call)
                &&
                "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUdeviceptr dstDevice */
            CUdeviceptr dstDevice;
            {
                dstDevice = (CUdeviceptr)__call->dstDevice;
            }

            /* Input: size_t ByteCount */
            size_t ByteCount;
            {
                ByteCount = (size_t)__call->ByteCount;
            }

            /* Input: CUstream hStream */
            CUstream hStream;
            {
                hStream = (CUstream)__call->hStream;
            }

            /* Input: const void * srcHost */
            void *srcHost;
            {
                if ((__call->srcHost) != (NULL)) {
                    if ((__call->__shm_srcHost)) {
                        srcHost = kava_shm_address((long)__call->srcHost);
                    }
                    else {
                        const void *__src_srcHost_0 = (const void *)__chan->chan_get_buffer(__chan,
                                __cmd, __call->srcHost);
                        volatile size_t __buffer_size = __call->ByteCount;   /* Size is in bytes. */
                        KAVA_DEBUG_ASSERT(__buffer_size % sizeof(const void) == 0);

                        srcHost = (void *)malloc(__buffer_size);
                        g_ptr_array_add(__kava_alloc_list_async_in, kava_buffer_with_deallocator_new(free, srcHost));

                        if ((srcHost) != (__src_srcHost_0)) {
                            memcpy(srcHost, __src_srcHost_0, __buffer_size);
                        }
                    }
                } else {
                    srcHost = NULL;
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuMemcpyHtoDAsync_v2(dstDevice, ByteCount, hStream, srcHost);

#ifdef REPLY_ASYNC_API
            size_t __total_buffer_size = 0;
            {
            }
            struct cu_cu_memcpy_hto_d_async_v2_ret *__ret =
                (struct cu_cu_memcpy_hto_d_async_v2_ret *)__chan->cmd_new(__chan,
                    sizeof(struct cu_cu_memcpy_hto_d_async_v2_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MEMCPY_HTO_D_ASYNC_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
#endif

            break;
        }

        case CALL_CUDA___CU_MEMCPY_DTO_H_ASYNC_V2:
        {
            struct cu_cu_memcpy_dto_h_async_v2_call *__call = (struct cu_cu_memcpy_dto_h_async_v2_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_memcpy_dto_h_async_v2_call)
                &&
                "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUdeviceptr srcDevice */
            CUdeviceptr srcDevice;
            {
                srcDevice = (CUdeviceptr) __call->srcDevice;
            }

            /* Input: size_t ByteCount */
            size_t ByteCount;
            {
                ByteCount = (size_t) __call->ByteCount;
            }

            /* Input: CUstream hStream */
            CUstream hStream;
            {
                hStream = (CUstream) __call->hStream;
            }

            /* Input: void * dstHost */
            void *dstHost;
            {
                if ((__call->dstHost) != (NULL)) {
                    if ((__call->__shm_dstHost)) {
                        dstHost = kava_shm_address((long)__call->dstHost);
                    }
                    else {
                        volatile size_t __buffer_size = __call->ByteCount;            /* Size is in bytes. */
                        KAVA_DEBUG_ASSERT(__buffer_size % sizeof(void) == 0);
                        dstHost = (void *)malloc(__buffer_size);
                        g_hash_table_insert(__kava_alloc_list_async_out, dstHost, dstHost);
                    }
                }
                else {
                    dstHost = NULL;
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuMemcpyDtoHAsync_v2(srcDevice, ByteCount, hStream, dstHost);

#ifdef REPLY_ASYNC_API
            size_t __total_buffer_size = 0;
            {
                /* Size: void * dstHost */
                if ((dstHost) != (NULL) && (ByteCount) > (0)) {
                    if (__call->__shm_dstHost) {
                    }
                    else {
                        __total_buffer_size +=(ByteCount) * sizeof(void));
                    }
                }
            }
            struct cu_cu_memcpy_dto_h_async_v2_ret *__ret =
                (struct cu_cu_memcpy_dto_h_async_v2_ret *)__chan->cmd_new(__chan,
                        sizeof(struct cu_cu_memcpy_dto_h_async_v2_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MEMCPY_DTO_H_ASYNC_V2;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: void * dstHost */
            {
                if ((dstHost) != (NULL) && (ByteCount) > (0)) {
                    if (__call->__shm_dstHost) {
                        __ret->dstHost = __call->dstHost;
                    }
                    else {
                        __ret->dstHost = (void *)kava_chan_attach_buffer(__chan,
                                (struct kava_cmd_base *)__ret, dstHost,
                                (ByteCount) * sizeof(void));
                    }
                } else {
                    __ret->dstHost = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);

            g_hash_table_remove(__kava_alloc_list_async_out, dstHost);
#endif

            break;
        }

        case CALL_CUDA___CU_GET_ERROR_STRING:
        {
            GPtrArray *__kava_alloc_list_cuGetErrorString =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_get_error_string_call *__call = (struct cu_cu_get_error_string_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_get_error_string_call)
                &&
                "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUresult error */
            CUresult error;
            {
                error = (CUresult) __call->error;
            }

            /* Input: const char ** pStr */
            char **pStr;
            {
                if ((__call->pStr) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    pStr = (char **)malloc(__size * sizeof(const char *));
                    g_ptr_array_add(__kava_alloc_list_cuGetErrorString, kava_buffer_with_deallocator_new(free, pStr));
                }
                else {
                    pStr = NULL;
                }
            }

            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuGetErrorString(error, (const char **)pStr);

            size_t __total_buffer_size = 0;
            {
                /* Size: const char ** pStr */
                if ((pStr) != (NULL)) {
                    const size_t __pStr_size_0 = ((size_t) (1));
                    for (size_t __pStr_index_0 = 0; __pStr_index_0 < __pStr_size_0; __pStr_index_0++) {
                        char **__pStr_a_0 = (char **)(pStr) + __pStr_index_0;

                        if ((*__pStr_a_0) != (NULL) && (strlen(*__pStr_a_0) + 1) > (0)) {
                            __total_buffer_size +=
                                __chan->chan_buffer_size(__chan,
                                        ((size_t)(strlen(*__pStr_a_0) + 1)) * sizeof(const char));
                        }
                    }
                    __total_buffer_size += __chan->chan_buffer_size(__chan, __pStr_size_0 * sizeof(const char *));
                }
            }
            struct cu_cu_get_error_string_ret *__ret =
                (struct cu_cu_get_error_string_ret *)__chan->cmd_new(__chan,
                        sizeof(struct cu_cu_get_error_string_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_GET_ERROR_STRING;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: const char ** pStr */
            {
                if ((pStr) != (NULL)) {
                    const size_t __pStr_size_0 = ((size_t) (1));
                    char **__tmp_pStr_0 = (char **)calloc(1, __pStr_size_0 * sizeof(const char *));
                    g_ptr_array_add(__kava_alloc_list_cuGetErrorString, kava_buffer_with_deallocator_new(free,
                            __tmp_pStr_0));
                    for (size_t __pStr_index_0 = 0; __pStr_index_0 < __pStr_size_0; __pStr_index_0++) {
                        char **__pStr_a_0 = (char **)(__tmp_pStr_0) + __pStr_index_0;
                        char **__pStr_b_0 = (char **)(pStr) + __pStr_index_0;

                        if ((*__pStr_b_0) != (NULL) && (strlen(*__pStr_b_0) + 1) > (0)) {
                            *__pStr_a_0 = (char *)__chan->chan_attach_buffer(__chan,
                                    (struct kava_cmd_base *)__ret,
                                    *__pStr_b_0,
                                    ((size_t)(strlen(*__pStr_b_0) + 1)) * sizeof(const char));
                        } else {
                            *__pStr_a_0 = NULL;
                        }
                    }
                    __ret->pStr = (char **)__chan->chan_attach_buffer(__chan,
                            (struct kava_cmd_base *)__ret, __tmp_pStr_0, ((size_t)(1)) * sizeof(const char *));
                } else {
                    __ret->pStr = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuGetErrorString);   /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_DEVICE_GET_ATTRIBUTE:
        {
            GPtrArray *__kava_alloc_list_cuDeviceGetAttribute =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_device_get_attribute_call *__call = (struct cu_cu_device_get_attribute_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_device_get_attribute_call)
                &&
                "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: int * pi */
            int *pi;
            {
                if ((__call->pi) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    pi = (int *)malloc(__size * sizeof(int));
                    g_ptr_array_add(__kava_alloc_list_cuDeviceGetAttribute, kava_buffer_with_deallocator_new(free, pi));
                }
                else {
                    pi = NULL;
                }
            }

            /* Input: CUdevice_attribute attrib */
            CUdevice_attribute attrib;
            {
                attrib = (CUdevice_attribute) __call->attrib;
            }

            /* Input: CUdevice dev */
            CUdevice dev;
            {
                dev = (CUdevice) __call->dev;
            }

            /* Perform Call */

            CUresult ret;
            ret = __wrapper_cuDeviceGetAttribute(pi, attrib, dev);

            size_t __total_buffer_size = 0;
            {
                /* Size: int * pi */
                if ((pi) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(int));
                }
            }
            struct cu_cu_device_get_attribute_ret *__ret =
                (struct cu_cu_device_get_attribute_ret *)__chan->cmd_new(__chan,
                        sizeof(struct cu_cu_device_get_attribute_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_DEVICE_GET_ATTRIBUTE;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: int * pi */
            {
                if ((pi) != (NULL)) {
                    __ret->pi =
                        (int *)__chan->chan_attach_buffer(__chan,
                                (struct kava_cmd_base *)__ret, pi, ((size_t) (1)) * sizeof(int));
                } else {
                    __ret->pi = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuDeviceGetAttribute);       /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_DEVICE_GET_NAME:
        {
            GPtrArray *__kava_alloc_list_cuDeviceGetName =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_device_get_name_call *__call = (struct cu_cu_device_get_name_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_device_get_name_call)
                &&
                "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: int len */
            int len;
            {
                len = (int)__call->len;
            }

            /* Input: CUdevice dev */
            CUdevice dev;
            {
                dev = (CUdevice) __call->dev;
            }

            /* Input: char * name */
            char *name;
            {
                if ((__call->name) != (NULL)) {
                    const size_t __size = ((size_t) (len));
                    name = (char *)malloc(__size * sizeof(char));
                    g_ptr_array_add(__kava_alloc_list_cuDeviceGetName, kava_buffer_with_deallocator_new(free, name));
                }
                else {
                    name = NULL;
                }
            }

            /* Perform Call */

            CUresult ret;
            ret = __wrapper_cuDeviceGetName(len, dev, name);

            size_t __total_buffer_size = 0;
            {
                /* Size: char * name */
                if ((name) != (NULL) && (len) > (0)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (len)) * sizeof(char));
                }
            }
            struct cu_cu_device_get_name_ret *__ret =
                (struct cu_cu_device_get_name_ret *)__chan->cmd_new(__chan,
                        sizeof(struct cu_cu_device_get_name_ret), __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_DEVICE_GET_NAME;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: char * name */
            {
                if ((name) != (NULL) && (len) > (0)) {
                    __ret->name = (char *)__chan->chan_attach_buffer(__chan,
                            (struct kava_cmd_base *)__ret, name, ((size_t) (len)) * sizeof(char));
                } else {
                    __ret->name = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuDeviceGetName);    /* Deallocate all memory in the alloc list */

            break;
        }

        case CALL_CUDA___CU_MEM_ALLOC_PITCH:
        {
            GPtrArray *__kava_alloc_list_cuMemAllocPitch =
                g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
            struct cu_cu_mem_alloc_pitch_call *__call = (struct cu_cu_mem_alloc_pitch_call *)__cmd;
            assert(__call->base.mode == KAVA_CMD_MODE_API);
            assert(__call->base.command_size == sizeof(struct cu_cu_mem_alloc_pitch_call) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)");

            /* Unpack and translate arguments */

            /* Input: CUdeviceptr * dptr */
            CUdeviceptr *dptr = NULL;
            {
                if ((__call->dptr) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    dptr = (CUdeviceptr *)malloc(__size * sizeof(CUdeviceptr));
                    g_ptr_array_add(__kava_alloc_list_cuMemAllocPitch,
                                    kava_buffer_with_deallocator_new(free, dptr));
                }
            }

            /* Input: size_t * pPitch */
            CUdeviceptr *pPitch = NULL;
            {
                if ((__call->pPitch) != (NULL)) {
                    const size_t __size = ((size_t) (1));
                    pPitch = (size_t *)malloc(__size * sizeof(size_t));
                    g_ptr_array_add(__kava_alloc_list_cuMemAllocPitch,
                                    kava_buffer_with_deallocator_new(free, pPitch));
                }
            }

            /* Input: size_t WidthInBytes */
            size_t WidthInBytes;
            {
                WidthInBytes = (size_t) __call->WidthInBytes;
            }

            /* Input: size_t Height */
            size_t Height;
            {
                Height = (size_t) __call->Height;
            }

            /* Input: size_t ElementSizeBytes */
            size_t ElementSizeBytes;
            {
                ElementSizeBytes = (size_t) __call->ElementSizeBytes;
            }


            /* Perform Call */
            CUresult ret;
            ret = __wrapper_cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);

            size_t __total_buffer_size = 0;
            {
                /* Size: CUdeviceptr * dptr */
                if ((dptr) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(CUdeviceptr));
                }
            }
            {
                /* Size: size_t * pPitch */
                if ((pPitch) != (NULL)) {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(size_t));
                }
            }
            struct cu_cu_mem_alloc_pitch_ret *__ret =
                (struct cu_cu_mem_alloc_pitch_ret *)__chan->cmd_new(__chan, sizeof(struct cu_cu_mem_alloc_pitch_ret),
                __total_buffer_size);
            __ret->base.mode = KAVA_CMD_MODE_API;
            __ret->base.command_id = RET_CUDA___CU_MEM_ALLOC_PITCH;
            __ret->base.thread_id = __call->base.thread_id;
            __ret->__call_id = __call->__call_id;

            /* Output: CUresult ret */
            {
                __ret->ret = ret;
            }
            /* Output: CUdeviceptr * dptr */
            {
                if ((dptr) != (NULL)) {
                    __ret->dptr = (CUdeviceptr *)__chan->chan_attach_buffer(
                            __chan, (struct kava_cmd_base *)__ret,
                            dptr, ((size_t) (1)) * sizeof(CUdeviceptr));
                } else {
                    __ret->dptr = NULL;
                }
            }
            /* Output: size_t * pPitch */
            {
                if ((pPitch) != (NULL)) {
                    __ret->pPitch = (size_t *)__chan->chan_attach_buffer(
                            __chan, (struct kava_cmd_base *)__ret,
                            pPitch, ((size_t) (1)) * sizeof(size_t));
                } else {
                    __ret->pPitch = NULL;
                }
            }

            /* Send reply message */
            __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
            g_ptr_array_unref(__kava_alloc_list_cuMemAllocPitch);      /* Deallocate all memory in the alloc list */

            break;
        }

        default:
            pr_err("Unrecognized CUDA command: %lu\n", __cmd->command_id);
    }
}

void __print_command_cuda(FILE* file, const struct kava_chan *__chan,
                        const struct kava_cmd_base *__cmd)
{
    switch (__cmd->command_id) {
        case CALL_CUDA___CU_INIT:
            pr_info("cuInit print is invoked\n");
            break;

        //default:
            //pr_err("Unrecognized CUDA command: %lu\n", __cmd->command_id);
    }
}
