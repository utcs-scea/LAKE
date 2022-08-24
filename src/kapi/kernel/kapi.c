#include <linux/types.h>
#include <linux/module.h>
#include "commands.h"
#include "lake_kapi.h"


// CUresult CUDAAPI XXX() {
//     struct lake_cmd_XXX cmd = {
//         .API_ID = LAKE_API_XXX, .flags = flags, .dev = dev
//     };
//     lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
//     return CUDA_SUCCESS;
// }
// EXPORT_SYMBOL(XXX);

CUresult CUDAAPI cuInit(unsigned int flags) {
    struct lake_cmd_cuInit cmd = {
        .API_ID = LAKE_API_cuInit, .flags = flags,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuInit);

CUresult CUDAAPI cuDeviceGet(CUdevice *device, int ordinal) {
    struct lake_cmd_cuDeviceGet cmd = {
        .API_ID = LAKE_API_cuDeviceGet, .ordinal = ordinal,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuDeviceGet);

CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    struct lake_cmd_cuCtxCreate cmd = {
        .API_ID = LAKE_API_cuCtxCreate, .flags = flags, .dev = dev
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuCtxCreate);

CUresult CUDAAPI cuModuleLoad(CUmodule *module, const char *fname) {
    struct lake_cmd_cuModuleLoad cmd = {
        .API_ID = LAKE_API_cuModuleLoad
    };
    strcpy(cmd.fname, fname);
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuModuleLoad);

CUresult CUDAAPI cuModuleUnload(CUmodule hmod) {
    struct lake_cmd_cuModuleUnload cmd = {
        .API_ID = LAKE_API_cuModuleUnload, .hmod = hmod
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuModuleUnload);

CUresult CUDAAPI cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    struct lake_cmd_cuModuleGetFunction cmd = {
        .API_ID = LAKE_API_cuModuleGetFunction, .hmod = hmod
    };
    strcpy(cmd.name, name);
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
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
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuLaunchKernel);

CUresult CUDAAPI cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    struct lake_cmd_cuMemAlloc cmd = {
        .API_ID = LAKE_API_cuMemAlloc, .bytesize = bytesize
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuMemAlloc);

CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    struct lake_cmd_cuMemcpyHtoD cmd = {
        .API_ID = LAKE_API_cuMemcpyHtoD, .dstDevice = dstDevice, .srcHost = srcHost,
        .ByteCount = ByteCount
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuMemcpyHtoD);

CUresult CUDAAPI cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    struct lake_cmd_cuMemcpyDtoH cmd = {
        .API_ID = LAKE_API_cuMemcpyDtoH, .dstHost = dstHost, .srcDevice = srcDevice,
        .ByteCount = ByteCount
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuMemcpyDtoH);

CUresult CUDAAPI cuCtxSynchronize(void) {
    struct lake_cmd_cuCtxSynchronize cmd = {
        .API_ID = LAKE_API_cuCtxSynchronize,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuCtxSynchronize);

CUresult CUDAAPI cuMemFree(CUdeviceptr dptr) {
    struct lake_cmd_cuMemFree cmd = {
        .API_ID = LAKE_API_cuMemFree, .dptr = dptr
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuMemFree);

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    struct lake_cmd_cuStreamCreate cmd = {
        .API_ID = LAKE_API_cuStreamCreate, .Flags = Flags
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuStreamCreate);

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream) {
    struct lake_cmd_cuStreamSynchronize cmd = {
        .API_ID = LAKE_API_cuStreamSynchronize, .hStream = hStream
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuStreamSynchronize);

CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
    struct lake_cmd_cuMemcpyHtoDAsync cmd = {
        .API_ID = LAKE_API_cuMemcpyHtoDAsync, .dstDevice = dstDevice, .srcHost = srcHost, 
        .ByteCount = ByteCount, .hStream = hStream
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuMemcpyHtoDAsync);

CUresult CUDAAPI cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream) {
    struct lake_cmd_cuMemcpyDtoHAsync cmd = {
        .API_ID = LAKE_API_cuMemcpyDtoHAsync, .dstHost = dstHost, .srcDevice = srcDevice,
        .ByteCount = ByteCount, .hStream = hStream
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_ASYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuMemcpyDtoHAsync);
