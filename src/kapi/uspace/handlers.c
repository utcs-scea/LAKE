#include <cuda.h>
#include <inttypes.h>
#include <stdio.h>
#include "commands.h"
#include "lake_shm.h"
#include "lake_kapi.h"
#include "kargs.h"

#define DRY_RUN 0

/*********************
 *  cuInit    
 *********************/
static int lake_handler_cuInit(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuInit *cmd = (struct lake_cmd_cuInit *) buf;
#if DRY_RUN
    printf("Dry running cuInit\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuInit(cmd->flags);
#endif
    return 0;
}

/*********************
 *  cuDeviceGet   
 *********************/
static int lake_handler_cuDeviceGet(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuDeviceGet *cmd = (struct lake_cmd_cuDeviceGet *) buf;
#if DRY_RUN
    printf("Dry running cuDeviceGet\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuDeviceGet(&cmd_ret->device, cmd->ordinal);
#endif
    return 0;
}

/*********************
 *  cuCtxCreate   
 *********************/
static int lake_handler_cuCtxCreate(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuCtxCreate *cmd = (struct lake_cmd_cuCtxCreate *) buf;
#if DRY_RUN
    printf("Dry running cuCtxCreate\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuCtxCreate_v2(&cmd_ret->pctx, cmd->flags, cmd->dev);
#endif
    return 0;
}

/*********************
 *  cuModuleLoad   
 *********************/
static int lake_handler_cuModuleLoad(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuModuleLoad *cmd = (struct lake_cmd_cuModuleLoad *) buf;
#if DRY_RUN
    printf("Dry running cuModuleLoad\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    //printf("Running cuModuleLoad  %s\n", cmd->fname);
    cmd_ret->res = cuModuleLoad(&cmd_ret->module, cmd->fname);
    //printf("cuModuleLoad ret %d\n", cmd_ret->res);
#endif
    return 0;
}

/*********************
 *  cuModuleUnload   
 *********************/
static int lake_handler_cuModuleUnload(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuModuleUnload *cmd = (struct lake_cmd_cuModuleUnload *) buf;
#if DRY_RUN
    printf("Dry running cuModuleUnload\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuModuleUnload(cmd->hmod);
#endif
    return 0;
}

/*********************
 *  cuModuleGetFunction   
 *********************/
static int lake_handler_cuModuleGetFunction(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuModuleGetFunction *cmd = (struct lake_cmd_cuModuleGetFunction *) buf;
#if DRY_RUN
    printf("Dry running cuModuleGetFunction\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuModuleGetFunction(&cmd_ret->func, cmd->hmod, cmd->name);
    struct kernel_args_metadata* meta = get_kargs(cmd_ret->func);
    kava_parse_function_args(cmd->name, meta);
#endif
    return 0;
}

/*********************
 *  cuLaunchKernel   
 *********************/
static int lake_handler_cuLaunchKernel(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuLaunchKernel *cmd = (struct lake_cmd_cuLaunchKernel *) buf;
#if DRY_RUN
    printf("Dry running cuLaunchKernel\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    struct kernel_args_metadata* meta = get_kargs(cmd->f);
    uint8_t *serialized = ((u8*)buf) + sizeof(struct lake_cmd_cuLaunchKernel);
    void* args = malloc(meta->func_argc * sizeof(void*));
    construct_args(meta, args, serialized), 
    cmd_ret->res = cuLaunchKernel(cmd->f, cmd->gridDimX, cmd->gridDimY,
        cmd->gridDimZ, cmd->blockDimX, cmd->blockDimY, cmd->blockDimZ, cmd->sharedMemBytes,
        cmd->hStream, args, cmd->extra);
    
    //free(args);
#endif
    return 0;
}

/*********************
 *  cuCtxDestroy   
 *********************/
static int lake_handler_cuCtxDestroy(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuCtxDestroy *cmd = (struct lake_cmd_cuCtxDestroy *) buf;
#if DRY_RUN
    printf("Dry running cuCtxDestroy\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuCtxDestroy(cmd->ctx);
#endif
    return 0;
}

/*********************
 *  cuMemAlloc   
 *********************/
static int lake_handler_cuMemAlloc(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuMemAlloc *cmd = (struct lake_cmd_cuMemAlloc *) buf;
#if DRY_RUN
    printf("Dry running cuMemAlloc\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuMemAlloc(&cmd_ret->ptr, cmd->bytesize);
    //printf("cuMemAlloc ptr %lx\n", cmd_ret->ptr);
#endif
    return 0;
}

/*********************
 *  cuMemcpyHtoD   
 *********************/
static int lake_handler_cuMemcpyHtoD(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuMemcpyHtoD *cmd = (struct lake_cmd_cuMemcpyHtoD *) buf;
#if DRY_RUN
    printf("Dry running cuMemcpyHtoD\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuMemcpyHtoD(cmd->dstDevice, lake_shm_address(cmd->srcHost), cmd->ByteCount);
#endif
    return 0;
}

/*********************
 *  cuMemcpyDtoH   
 *********************/
static int lake_handler_cuMemcpyDtoH(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuMemcpyDtoH *cmd = (struct lake_cmd_cuMemcpyDtoH *) buf;
#if DRY_RUN
    printf("Dry running cuMemcpyDtoH\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuMemcpyDtoH(lake_shm_address(cmd->dstHost), cmd->srcDevice, cmd->ByteCount);
#endif
    return 0;
}

/*********************
 *  cuCtxSynchronize   
 *********************/
static int lake_handler_cuCtxSynchronize(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuCtxSynchronize *cmd = (struct lake_cmd_cuCtxSynchronize *) buf;
#if DRY_RUN
    printf("Dry running cuCtxSynchronize\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuCtxSynchronize();
#endif
    return 0;
}

/*********************
 *  cuMemFree   
 *********************/
static int lake_handler_cuMemFree(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuMemFree *cmd = (struct lake_cmd_cuMemFree *) buf;
#if DRY_RUN
    printf("Dry running cuMemFree\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    printf("cuMemFree %llx\n", cmd->dptr);
    cmd_ret->res = cuMemFree(cmd->dptr);
#endif
    return 0;
}

/*********************
 *  cuStreamCreate   
 *********************/
static int lake_handler_cuStreamCreate(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuStreamCreate *cmd = (struct lake_cmd_cuStreamCreate *) buf;
#if DRY_RUN
    printf("Dry running cuStreamCreate\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuStreamCreate(&cmd_ret->stream, cmd->Flags);
#endif
    return 0;
}

/*********************
 *  cuStreamSynchronize   
 *********************/
static int lake_handler_cuStreamSynchronize(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuStreamSynchronize *cmd = (struct lake_cmd_cuStreamSynchronize *) buf;
#if DRY_RUN
    printf("Dry running cuStreamSynchronize\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuStreamSynchronize(cmd->hStream);
#endif
    return 0;
}

/*********************
 *  cuStreamDestroy   
 *********************/
static int lake_handler_cuStreamDestroy(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuStreamDestroy *cmd = (struct lake_cmd_cuStreamDestroy *) buf;
#if DRY_RUN
    printf("Dry running cuStreamDestroy\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuStreamDestroy(cmd->hStream);
#endif
    return 0;
}

/*********************
 *  cuMemcpyHtoDAsync   
 *********************/
static int lake_handler_cuMemcpyHtoDAsync(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuMemcpyHtoDAsync *cmd = (struct lake_cmd_cuMemcpyHtoDAsync *) buf;
#if DRY_RUN
    printf("Dry running cuMemcpyHtoDAsync\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuMemcpyHtoDAsync(cmd->dstDevice, lake_shm_address(cmd->srcHost), cmd->ByteCount, cmd->hStream);
#endif
    return 0;
}

/*********************
 *  cuMemcpyDtoHAsync   
 *********************/
static int lake_handler_cuMemcpyDtoHAsync(void* buf, struct lake_cmd_ret* cmd_ret) {
    struct lake_cmd_cuMemcpyDtoHAsync *cmd = (struct lake_cmd_cuMemcpyDtoHAsync *) buf;
#if DRY_RUN
    printf("Dry running cuMemcpyDtoHAsync\n");
    cmd_ret->res = CUDA_SUCCESS;
#else
    cmd_ret->res = cuMemcpyDtoHAsync(lake_shm_address(cmd->dstHost), cmd->srcDevice, cmd->ByteCount, cmd->hStream);
#endif
    return 0;
}

/*********************
 * 
 *  END OF HANDLERS
 *    
 *********************/

//order matters, need to match enum in src/kapi/include/commands.h
static int (*kapi_handlers[])(void* buf, struct lake_cmd_ret* cmd_ret) = {
    lake_handler_cuInit,
    lake_handler_cuDeviceGet,
    lake_handler_cuCtxCreate,
    lake_handler_cuModuleLoad,
    lake_handler_cuModuleUnload,
    lake_handler_cuModuleGetFunction,
    lake_handler_cuLaunchKernel,
    lake_handler_cuCtxDestroy,
    lake_handler_cuMemAlloc,
    lake_handler_cuMemcpyHtoD,
    lake_handler_cuMemcpyDtoH,
    lake_handler_cuCtxSynchronize,
    lake_handler_cuMemFree,
    lake_handler_cuStreamCreate,
    lake_handler_cuStreamSynchronize,
    lake_handler_cuStreamDestroy,
    lake_handler_cuMemcpyHtoDAsync,
    lake_handler_cuMemcpyDtoHAsync,
};

void lake_handle_cmd(void* buf, struct lake_cmd_ret* cmd_ret) {
    uint32_t cmd_id = *((uint32_t*) buf);
    kapi_handlers[cmd_id](buf, cmd_ret);
}