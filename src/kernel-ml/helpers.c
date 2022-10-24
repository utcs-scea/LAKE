#include "helpers.h"

void gpu_init(int dev, CUcontext *cuctx) {
    CUdevice cuDevice;
    CUresult res;

    cuInit(0);
    res = cuDeviceGet(&cuDevice, dev);
    if (res != CUDA_SUCCESS){
        #ifdef __KERNEL__
        PRINT("cannot acquire device 0\n");
        #else
        printf("1cannot acquire device 0\n");
        #endif
    }

    res = cuCtxCreate(cuctx, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        #ifdef __KERNEL__
        PRINT("cannot create context\n");
        #else
        printf("2cannot acquire device 0\n");
        #endif
    }
}

void gpu_get_cufunc(char* cubin, char* kname, CUfunction *func) {
    CUmodule cuModule;
    CUresult res;
    res = cuModuleLoad(&cuModule, cubin);
    if (res != CUDA_SUCCESS) {
        #ifdef __KERNEL__
        PRINT("cannot load module: %d\n", res);
        #else
        printf("cuModuleLoad err %d\n", res);
        #endif
    }

    res = cuModuleGetFunction(func, cuModule, kname);
    if (res != CUDA_SUCCESS){
        #ifdef __KERNEL__
        PRINT("cannot acquire kernel handle\n");
        #else
        printf("cuModuleGetFunction err %d\n", res);
        #endif
    }
}
