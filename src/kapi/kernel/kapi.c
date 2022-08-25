#include <linux/types.h>
#include <linux/module.h>
#include "commands.h"
#include "lake_kapi.h"
#include "lake_shm.h"

/*
 *
 *   Functions in this file export CUDA symbols.
 *   In general they fill a struct and send it through netlink.
 *   They also choose if they are sync or async calls.
 *   Some have special handling, such as memcpys
 * 
 *   TODO: support netlink copies (not urgent)
 *   TODO: get return codes and accumulate them
 */

//if (kava_shm_offset(srcHost) >= 0) {
//    __cmd->srcHost = (void *)kava_shm_offset(srcHost);
//    __cmd->__shm_srcHost = 1;


CUresult CUDAAPI cuInit(unsigned int flags) {
    struct lake_cmd_cuInit cmd = {
        .API_ID = LAKE_API_cuInit, .flags = flags,
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuInit);

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    struct lake_cmd_cuDeviceGet cmd = {
        .API_ID = LAKE_API_cuDeviceGet, .ordinal = ordinal,
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuDeviceGet);

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    struct lake_cmd_cuCtxCreate cmd = {
        .API_ID = LAKE_API_cuCtxCreate, .flags = flags, .dev = dev
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuCtxCreate);

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
    struct lake_cmd_cuModuleLoad cmd = {
        .API_ID = LAKE_API_cuModuleLoad
    };
    strcpy(cmd.fname, fname);
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuModuleLoad);

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
    struct lake_cmd_cuModuleUnload cmd = {
        .API_ID = LAKE_API_cuModuleUnload, .hmod = hmod
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuModuleUnload);

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    struct lake_cmd_cuModuleGetFunction cmd = {
        .API_ID = LAKE_API_cuModuleGetFunction, .hmod = hmod
    };
    strcpy(cmd.name, name);
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuModuleGetFunction);

CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra) {
    struct lake_cmd_cuLaunchKernel cmd = {
        .API_ID = LAKE_API_cuLaunchKernel, .f = f, .gridDimX = gridDimX,
        .gridDimY = gridDimY, .gridDimZ = gridDimZ, .blockDimX = blockDimX,
        .blockDimY = blockDimY, .blockDimZ = blockDimZ, .sharedMemBytes = sharedMemBytes,
        .hStream = hStream, .extra = 0
    };
    //TODO: serialize kernelParams
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuLaunchKernel);

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    struct lake_cmd_cuMemAlloc cmd = {
        .API_ID = LAKE_API_cuMemAlloc, .bytesize = bytesize
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuMemAlloc);

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    struct lake_cmd_cuMemcpyHtoD cmd = {
        .API_ID = LAKE_API_cuMemcpyHtoD, .dstDevice = dstDevice, .srcHost = srcHost,
        .ByteCount = ByteCount
    };

    long offset = kava_shm_offset(srcHost);

    if (offset < 0) {
        pr_err("srcHost in cuMemcpyHtoD is NOT a kshm pointer (use kava_alloc to fix it)\n");
        return CUDA_ERROR_INVALID_VALUE;
    }
    cmd.srcHost = offset;

    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC);
}
EXPORT_SYMBOL(cuMemcpyHtoD);

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    struct lake_cmd_cuMemcpyDtoH cmd = {
        .API_ID = LAKE_API_cuMemcpyDtoH, .dstHost = dstHost, .srcDevice = srcDevice,
        .ByteCount = ByteCount
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuMemcpyDtoH);

CUresult CUDAAPI cuCtxSynchronize(void) {
    struct lake_cmd_cuCtxSynchronize cmd = {
        .API_ID = LAKE_API_cuCtxSynchronize,
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuCtxSynchronize);

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    struct lake_cmd_cuMemFree cmd = {
        .API_ID = LAKE_API_cuMemFree, .dptr = dptr
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuMemFree);

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    struct lake_cmd_cuStreamCreate cmd = {
        .API_ID = LAKE_API_cuStreamCreate, .Flags = Flags
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuStreamCreate);

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
    struct lake_cmd_cuStreamSynchronize cmd = {
        .API_ID = LAKE_API_cuStreamSynchronize, .hStream = hStream
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
}
EXPORT_SYMBOL(cuStreamSynchronize);

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    struct lake_cmd_cuMemcpyHtoDAsync cmd = {
        .API_ID = LAKE_API_cuMemcpyHtoDAsync, .dstDevice = dstDevice, .srcHost = srcHost, 
        .ByteCount = ByteCount, .hStream = hStream
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC);
}
EXPORT_SYMBOL(cuMemcpyHtoDAsync);

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    struct lake_cmd_cuMemcpyDtoHAsync cmd = {
        .API_ID = LAKE_API_cuMemcpyDtoHAsync, .dstHost = dstHost, .srcDevice = srcDevice,
        .ByteCount = ByteCount, .hStream = hStream
    };
    return lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC);
}
EXPORT_SYMBOL(cuMemcpyDtoHAsync);
