#ifndef __CUDA_KAVA_H__
#define __CUDA_KAVA_H__

#include "command.h"
#include "util.h"

#include "cuda.h"

#ifndef __KERNEL__

#define checkCudaErrors(err)  __checkCudaErrors(err, __FILE__, __LINE__)

static void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

#endif

enum cuda_functions {
    CALL_CUDA___CUDA_RESERVED,    RET_CUDA___CUDA_RESERVED,

    CALL_CUDA___CUDA_GET_GPU_UTILIZATION_RATES, RET_CUDA___CUDA_GET_GPU_UTILIZATION_RATES,

    CALL_CUDA___CUDA_TEST_MMUL,   RET_CUDA___CUDA_TEST_MMUL,
    CALL_CUDA___CUDA_TEST_INIT,   RET_CUDA___CUDA_TEST_INIT,
    CALL_CUDA___CUDA_TEST_FREE,   RET_CUDA___CUDA_TEST_FREE,
    CALL_CUDA___CUDA_TEST_K_TO_U, RET_CUDA___CUDA_TEST_K_TO_U,
    CALL_CUDA___CUDA_TEST_CHANNEL, RET_CUDA___CUDA_TEST_CHANNEL,

    CALL_CUDA___CU_INIT,                 RET_CUDA___CU_INIT,
    CALL_CUDA___CU_DEVICE_GET,           RET_CUDA___CU_DEVICE_GET,
    CALL_CUDA___CU_CTX_CREATE_V2,        RET_CUDA___CU_CTX_CREATE_V2,
    CALL_CUDA___CU_MODULE_LOAD,          RET_CUDA___CU_MODULE_LOAD,
    CALL_CUDA___CU_MODULE_UNLOAD,        RET_CUDA___CU_MODULE_UNLOAD,
    CALL_CUDA___CU_MODULE_GET_FUNCTION,  RET_CUDA___CU_MODULE_GET_FUNCTION,
    CALL_CUDA___CU_LAUNCH_KERNEL,        RET_CUDA___CU_LAUNCH_KERNEL,
    CALL_CUDA___CU_CTX_DESTROY_V2,       RET_CUDA___CU_CTX_DESTROY_V2,
    CALL_CUDA___CU_MEM_ALLOC_V2,         RET_CUDA___CU_MEM_ALLOC_V2,
    CALL_CUDA___CU_MEMCPY_HTO_D_V2,      RET_CUDA___CU_MEMCPY_HTO_D_V2,
    CALL_CUDA___CU_MEMCPY_DTO_H_V2,      RET_CUDA___CU_MEMCPY_DTO_H_V2,
    CALL_CUDA___CU_CTX_SYNCHRONIZE,      RET_CUDA___CU_CTX_SYNCHRONIZE,
    CALL_CUDA___CU_CTX_SET_CURRENT,      RET_CUDA___CU_CTX_SET_CURRENT,
    CALL_CUDA___CU_DRIVER_GET_VERSION,   RET_CUDA___CU_DRIVER_GET_VERSION,
    CALL_CUDA___CU_MEM_FREE_V2,          RET_CUDA___CU_MEM_FREE_V2,
    CALL_CUDA___CU_MODULE_GET_GLOBAL_V2, RET_CUDA___CU_MODULE_GET_GLOBAL_V2,
    CALL_CUDA___CU_DEVICE_GET_COUNT,     RET_CUDA___CU_DEVICE_GET_COUNT,
    CALL_CUDA___CU_FUNC_SET_CACHE_CONFIG, RET_CUDA___CU_FUNC_SET_CACHE_CONFIG,

    CALL_CUDA___CU_STREAM_CREATE,        RET_CUDA___CU_STREAM_CREATE,
    CALL_CUDA___CU_STREAM_DESTROY_V2,    RET_CUDA___CU_STREAM_DESTROY_V2,
    CALL_CUDA___CU_STREAM_SYNCHRONIZE,   RET_CUDA___CU_STREAM_SYNCHRONIZE,
    CALL_CUDA___CU_MEMCPY_HTO_D_ASYNC_V2, RET_CUDA___CU_MEMCPY_HTO_D_ASYNC_V2,
    CALL_CUDA___CU_MEMCPY_DTO_H_ASYNC_V2, RET_CUDA___CU_MEMCPY_DTO_H_ASYNC_V2,

    CALL_CUDA___CU_GET_ERROR_STRING,     RET_CUDA___CU_GET_ERROR_STRING,
    CALL_CUDA___CU_DEVICE_GET_ATTRIBUTE, RET_CUDA___CU_DEVICE_GET_ATTRIBUTE,
    CALL_CUDA___CU_DEVICE_GET_NAME,      RET_CUDA___CU_DEVICE_GET_NAME,
    CALL_CUDA___CU_MEM_HOST_ALLOC,       RET_CUDA___CU_MEM_HOST_ALLOC,
    CALL_CUDA___CU_MEM_FREE_HOST,        RET_CUDA___CU_MEM_FREE_HOST,
    CALL_CUDA___CU_MEM_HOST_GET_DEVICE_POINTER, RET_CUDA___CU_MEM_HOST_GET_DEVICE_POINTER,
    CALL_CUDA___CU_MEM_ALLOC_PITCH,      RET_CUDA___CU_MEM_ALLOC_PITCH,
    CALL_CUDA___CU_MEMCPY_2D_V2,         RET_CUDA___CU_MEMCPY_2D_V2,
};

struct kava_stop_shadow_thread_call {
    struct kava_cmd_base base;
};

struct cu_cu_get_gpu_utilization_rates_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
};

struct cu_cu_get_gpu_utilization_rates_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int ret;
};

struct cu_cu_get_gpu_utilization_rates_call_record {
    char __handler_deallocate;
    volatile char __call_complete;
    int ret;
};

struct cu_cu_test_channel_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
};

struct cu_cu_test_channel_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
};

struct cu_cu_test_channel_call_record {
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_init_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int Flags;
};

struct cu_cu_init_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_init_call_record {
    unsigned int Flags;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_device_get_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdevice *device;
    int ordinal;
};

struct cu_cu_device_get_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdevice *device;
    CUresult ret;
};

struct cu_cu_device_get_call_record {
    CUdevice *device;
    int ordinal;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_ctx_create_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUcontext *pctx;
    unsigned int flags;
    CUdevice dev;
};

struct cu_cu_ctx_create_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUcontext *pctx;
    CUresult ret;
};

struct cu_cu_ctx_create_v2_call_record {
    CUcontext *pctx;
    unsigned int flags;
    CUdevice dev;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_module_load_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUmodule *module;
    char *fname;
    size_t size;
    void *cubin;
};

struct cu_cu_module_load_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUmodule *module;
    CUresult ret;
};

struct cu_cu_module_load_call_record {
    CUmodule *module;
    char *fname;
    CUresult ret;
    int res;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_module_unload_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUmodule hmod;
};

struct cu_cu_module_unload_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_module_unload_call_record {
    CUmodule hmod;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_module_get_function_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUfunction *hfunc;
    CUmodule hmod;
    char *name;
};

struct cu_cu_module_get_function_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUfunction *hfunc;
    CUresult ret;
};

struct cu_cu_module_get_function_call_record {
    CUfunction *hfunc;
    CUmodule hmod;
    char *name;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_launch_kernel_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **extra;
    void **kernelParams;
};

struct cu_cu_launch_kernel_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_launch_kernel_call_record {
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    void **extra;
    void **kernelParams;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_ctx_destroy_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUcontext ctx;
};

struct cu_cu_ctx_destroy_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_ctx_destroy_v2_call_record {
    CUcontext ctx;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_mem_alloc_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr *dptr;
    size_t bytesize;
};

struct cu_cu_mem_alloc_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr *dptr;
    CUresult ret;
};

struct cu_cu_mem_alloc_v2_call_record {
    CUdeviceptr *dptr;
    size_t bytesize;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_memcpy_hto_d_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr dstDevice;
    size_t ByteCount;
    void *srcHost;
    char __shm_srcHost;
};

struct cu_cu_memcpy_hto_d_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_memcpy_hto_d_v2_call_record {
    CUdeviceptr dstDevice;
    size_t ByteCount;
    void *srcHost;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_memcpy_dto_h_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    void *dstHost;
    char __shm_dstHost;
};

struct cu_cu_memcpy_dto_h_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    void *dstHost;
    CUresult ret;
};

struct cu_cu_memcpy_dto_h_v2_call_record {
    CUdeviceptr srcDevice;
    size_t ByteCount;
    void *dstHost;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_ctx_synchronize_call {
    struct kava_cmd_base base;
    intptr_t __call_id;

};

struct cu_cu_ctx_synchronize_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_ctx_synchronize_call_record {
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_ctx_set_current_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUcontext ctx;
};

struct cu_cu_ctx_set_current_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_ctx_set_current_call_record {
    CUcontext ctx;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_driver_get_version_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int *driverVersion;
};

struct cu_cu_driver_get_version_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int *driverVersion;
    CUresult ret;
};

struct cu_cu_driver_get_version_call_record {
    int *driverVersion;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_mem_free_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr dptr;
};

struct cu_cu_mem_free_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_mem_free_v2_call_record {
    CUdeviceptr dptr;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_module_get_global_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr *dptr;
    size_t *bytes;
    CUmodule hmod;
    char *name;
};

struct cu_cu_module_get_global_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr *dptr;
    size_t *bytes;
    CUresult ret;
};

struct cu_cu_module_get_global_v2_call_record {
    CUdeviceptr *dptr;
    size_t *bytes;
    CUmodule hmod;
    char *name;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_device_get_count_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int *count;
};

struct cu_cu_device_get_count_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int *count;
    CUresult ret;
};

struct cu_cu_device_get_count_call_record {
    int *count;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_func_set_cache_config_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUfunction hfunc;
    CUfunc_cache config;
};

struct cu_cu_func_set_cache_config_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_func_set_cache_config_call_record {
    CUfunction hfunc;
    CUfunc_cache config;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_stream_create_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUstream *phStream;
    unsigned int Flags;
};

struct cu_cu_stream_create_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUstream *phStream;
    CUresult ret;
};

struct cu_cu_stream_create_call_record {
    CUstream *phStream;
    unsigned int Flags;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_stream_synchronize_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUstream hStream;
};

struct cu_cu_stream_synchronize_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_stream_synchronize_call_record {
    CUstream hStream;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_stream_destroy_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUstream hStream;
};

struct cu_cu_stream_destroy_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_stream_destroy_v2_call_record {
    CUstream hStream;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_memcpy_hto_d_async_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr dstDevice;
    size_t ByteCount;
    CUstream hStream;
    void *srcHost;
    char __shm_srcHost;
};

struct cu_cu_memcpy_hto_d_async_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_memcpy_hto_d_async_v2_call_record {
    CUdeviceptr dstDevice;
    size_t ByteCount;
    CUstream hStream;
    void *srcHost;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_memcpy_dto_h_async_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
    void *dstHost;
    char __shm_dstHost;
};

struct cu_cu_memcpy_dto_h_async_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    void *dstHost;
    CUresult ret;
};

struct cu_cu_memcpy_dto_h_async_v2_call_record {
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
    void *dstHost;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_get_error_string_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult error;
    char **pStr;
};

struct cu_cu_get_error_string_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    char **pStr;
    CUresult ret;
};

struct cu_cu_get_error_string_call_record {
    CUresult error;
    char **pStr;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_device_get_attribute_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int *pi;
    CUdevice_attribute attrib;
    CUdevice dev;
};

struct cu_cu_device_get_attribute_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int *pi;
    CUresult ret;
};

struct cu_cu_device_get_attribute_call_record {
    int *pi;
    CUdevice_attribute attrib;
    CUdevice dev;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_device_get_name_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int len;
    CUdevice dev;
    char *name;
};

struct cu_cu_device_get_name_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    char *name;
    CUresult ret;
};

struct cu_cu_device_get_name_call_record {
    int len;
    CUdevice dev;
    char *name;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_mem_host_alloc_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    size_t bytesize;
    unsigned int Flags;
    void **pp;
};

struct cu_cu_mem_host_alloc_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    void **pp;
    CUresult ret;
};

struct cu_cu_mem_host_alloc_call_record {
    size_t bytesize;
    unsigned int Flags;
    void **pp;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_mem_free_host_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    void *p;
};

struct cu_cu_mem_free_host_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    void *p;
    CUresult ret;
};

struct cu_cu_mem_free_host_call_record {
    void *p;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_mem_host_get_device_pointer_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr* pdptr;
    void *p;
    unsigned int Flags;
};

struct cu_cu_mem_host_get_device_pointer_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr* pdptr;
    CUresult ret;
};

struct cu_cu_mem_host_get_device_pointer_call_record {
    CUdeviceptr* pdptr;
    void *p;
    unsigned int Flags;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_mem_alloc_pitch_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr *dptr;
    size_t *pPitch;
    size_t WidthInBytes;
    size_t Height;
    unsigned int ElementSizeBytes;
};

struct cu_cu_mem_alloc_pitch_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUdeviceptr *dptr;
    size_t *pPitch;
    CUresult ret;
};

struct cu_cu_mem_alloc_pitch_call_record {
    CUdeviceptr *dptr;
    size_t *pPitch;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct cu_cu_memcpy_2d_v2_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUDA_MEMCPY2D *pCopy;
};

struct cu_cu_memcpy_2d_v2_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    CUresult ret;
};

struct cu_cu_memcpy_2d_v2_call_record {
    CUDA_MEMCPY2D *pCopy;
    CUresult ret;
    char __handler_deallocate;
    volatile char __call_complete;
};



// ======= Begin KAVA ksm struct =======
// Copied from klib's cuda.h
struct kava_cuda_fn_table {
  void (*__kava_stop_shadow_thread) (void);
  void * (*kava_alloc) (size_t size);
  int (*cudaGetGPUUtilizationRates) (void);
  CUresult (*cuInit) (unsigned int Flags);
  CUresult (*cuDeviceGet) (CUdevice *device, int ordinal);
  CUresult (*cuCtxCreate) (CUcontext *pctx, unsigned int flags, CUdevice dev);
  CUresult (*cuModuleLoad) (CUmodule *module, const char *fname);
  CUresult (*cuModuleUnload) (CUmodule hmod);
  CUresult (*cuModuleGetFunction) (CUfunction *hfunc, CUmodule hmod, const char *name);
  CUresult (*cuLaunchKernel) (CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra);
  CUresult (*cuCtxDestroy) (CUcontext ctx);
  CUresult (*cuStreamCreate) (CUstream *phstream, unsigned int flags);
  CUresult (*cuStreamSynchronize) (CUstream hStream);
  CUresult (*cuStreamDestroy) (CUstream hstream);
  CUresult (*cuMemAlloc) (CUdeviceptr *dptr, size_t bytesize);
  CUresult (*cuMemcpyHtoD) (CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
  CUresult (*cuMemcpyDtoH) (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
  CUresult (*cuMemcpyHtoDAsync) (CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hstream);
  CUresult (*cuMemcpyDtoHAsync) (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hstream);
  CUresult (*cuCtxSynchronize) (void);
  CUresult (*cuDriverGetVersion) (int *driverVersion);
  CUresult (*cuMemFree) (CUdeviceptr dptr);
  CUresult (*cuModuleGetGlobal) (CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name);
  CUresult (*cuDeviceGetCount) (int *count);
  CUresult (*cuFuncSetCacheConfig) (CUfunction hfunc, CUfunc_cache config);
  CUresult (*cuCtxSetCurrent) (CUcontext ctx);
};
// ======= End KAVA ksm struct =======

#endif
