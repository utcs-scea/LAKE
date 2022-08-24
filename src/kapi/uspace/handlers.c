#include <cuda.h>
#include <inttypes.h>
#include <stdio.h>
#include "commands.h"

#define DRY_RUN 1

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
    cmd_ret->res = cuDeviceGet(cmd->flags);
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
    cmd_ret->res = cuCtxCreate(cmd->flags);
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
    cmd_ret->res = cuModuleLoad(cmd->flags);
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
    cmd_ret->res = cuModuleUnload(cmd->flags);
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
    cmd_ret->res = cuModuleGetFunction(cmd->flags);
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
    cmd_ret->res = cuLaunchKernel(cmd->flags);
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
    cmd_ret->res = cuMemcpyHtoD(cmd->flags);
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
    cmd_ret->res = cuMemcpyDtoH(cmd->flags);
#endif
    return 0;
}


/*********************
 * 
 *  END OF HANDLERS
 *    
 *********************/

//order matters, need to match src/kapi/include/commands.h
static int (*kapi_handlers[])(void* buf, struct lake_cmd_ret* cmd_ret) = {
    lake_handler_cuInit,
    lake_handler_cuDeviceGet, 
    lake_handler_cuCtxCreate,
    lake_handler_cuModuleLoad,
    lake_handler_cuModuleUnload,
    lake_handler_cuModuleGetFunction,
    lake_handler_cuLaunchKernel,
    lake_handler_cuMemcpyHtoD,
    lake_handler_cuMemcpyDtoH,
};

void lake_handle_cmd(void* buf, struct lake_cmd_ret* cmd_ret) {
    uint32_t cmd_id = *((uint32_t*) buf);
    kapi_handlers[cmd_id](buf, cmd_ret);
}