/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef __KAPI_COMMANDS_H__
#define __KAPI_COMMANDS_H__

#include "cuda.h"

typedef unsigned int u32;

#define CMD_ASYNC 0
#define CMD_SYNC  1

enum lake_api_ids {
    LAKE_API_cuInit = 0,
    LAKE_API_cuDeviceGet,
    LAKE_API_cuCtxCreate,
    LAKE_API_cuModuleLoad,
    LAKE_API_cuModuleUnload,
    LAKE_API_cuModuleGetFunction,
    LAKE_API_cuLaunchKernel,
    LAKE_API_cuCtxDestroy,
    LAKE_API_cuMemAlloc,
    LAKE_API_cuMemcpyHtoD,
    LAKE_API_cuMemcpyDtoH,
    LAKE_API_cuCtxSynchronize,
    LAKE_API_cuMemFree,
    LAKE_API_cuStreamCreate,
    LAKE_API_cuStreamSynchronize,
    LAKE_API_cuStreamDestroy,
    LAKE_API_cuMemcpyHtoDAsync,
    LAKE_API_cuMemcpyDtoHAsync,
    LAKE_API_cuMemAllocPitch,
    LAKE_API_kleioLoadModel,
    LAKE_API_kleioInference,
    LAKE_API_kleioForceGC,
    LAKE_API_nvmlRunningProcs,
};

struct lake_cmd_ret {
    CUresult res;
    union {
        CUdeviceptr ptr; //u64
        CUdevice device; //int
        CUcontext pctx; //ptr
        CUmodule module; //ptr
        CUfunction func; //ptr
        CUstream stream; //ptr
    };
    size_t pPitch; //malloc pitch ruined everything
};

struct lake_cmd_cuInit {
    u32 API_ID;
    int flags;
};

struct lake_cmd_cuDeviceGet {
    u32 API_ID;
    //CUdevice *device; 
    int ordinal;
};

struct lake_cmd_cuCtxCreate {
    u32 API_ID;
    //CUcontext *pctx; 
    unsigned int flags; 
    CUdevice dev;
};

struct lake_cmd_cuModuleLoad {
    u32 API_ID;
    //CUmodule *module;
    //const char *fname;
    char fname[256];
};

struct lake_cmd_cuModuleUnload {
    u32 API_ID;
    CUmodule hmod;
};

struct lake_cmd_cuModuleGetFunction {
    u32 API_ID;
    //CUfunction *hfunc;
    CUmodule hmod; 
    //char *name;
    char name[256];
};

struct lake_cmd_cuLaunchKernel {
    u32 API_ID;
    CUfunction f;
    unsigned int gridDimX;
    unsigned int gridDimY;
    unsigned int gridDimZ;
    unsigned int blockDimX;
    unsigned int blockDimY;
    unsigned int blockDimZ;
    unsigned int sharedMemBytes;
    CUstream hStream;
    //extra is always null
    void **extra;
    unsigned int paramsSize;
};

struct lake_cmd_cuCtxDestroy {
    u32 API_ID;
    CUcontext ctx;
};

struct lake_cmd_cuMemAlloc {
    u32 API_ID;
    //CUdeviceptr *dptr; 
    size_t bytesize;
};

struct lake_cmd_cuMemcpyHtoD {
    u32 API_ID;
    CUdeviceptr dstDevice; 
    const void *srcHost;
    size_t ByteCount;
};

struct lake_cmd_cuMemcpyDtoH {
    u32 API_ID;
    void *dstHost; 
    CUdeviceptr srcDevice; 
    size_t ByteCount;
};

struct lake_cmd_cuCtxSynchronize {
    u32 API_ID;
};

struct lake_cmd_cuMemFree {
    u32 API_ID;
    CUdeviceptr dptr;
};

struct lake_cmd_cuStreamCreate {
    u32 API_ID;
    //CUstream *phStream;
    unsigned int Flags;
};

struct lake_cmd_cuStreamSynchronize {
    u32 API_ID;
    CUstream hStream;
};

struct lake_cmd_cuStreamDestroy {
    u32 API_ID;
    CUstream hStream;
};

struct lake_cmd_cuMemcpyHtoDAsync {
    u32 API_ID;
    CUdeviceptr dstDevice;
    const void *srcHost; 
    size_t ByteCount; 
    CUstream hStream;
};

struct lake_cmd_cuMemcpyDtoHAsync {
    u32 API_ID;
    void *dstHost;
    CUdeviceptr srcDevice;
    size_t ByteCount;
    CUstream hStream;
};

struct lake_cmd_cuMemAllocPitch {
    u32 API_ID;
    //CUdeviceptr* dptr;
    //size_t* pPitch;
    size_t WidthInBytes; 
    size_t Height;
    unsigned int ElementSizeBytes;
};

struct lake_cmd_kleioLoadModel {
    u32 API_ID;
    const void *srcHost;
    size_t len;
};

struct lake_cmd_kleioInference {
    u32 API_ID;
    const void *srcHost;
    size_t len;
    int use_gpu;
};

struct lake_cmd_kleioForceGC {
    u32 API_ID;
};

struct lake_cmd_nvmlRunningProcs {
    u32 API_ID;
};

#endif