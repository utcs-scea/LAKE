#include "helpers.h"

void gpu_init(int dev, CUcontext *cuctx) {
    CUdevice cuDevice;
    CUresult res;

    cuInit(0);
    res = cuDeviceGet(&cuDevice, dev);
    if (res != CUDA_SUCCESS){
        #ifdef __KERNEL__
        PRINT(V_INFO, "cannot acquire device 0\n");
        #else
        printf("cannot acquire device 0\n");
        #endif
    }

    res = cuCtxCreate(cuctx, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        #ifdef __KERNEL__
        PRINT(V_INFO, "cannot create context\n");
        #else
        printf("cannot acquire device 0\n");
        #endif
    }
}

void gpu_get_cufunc(char* cubin, char* kname, CUfunction *func) {
    CUmodule cuModule;
    CUresult res;
    res = cuModuleLoad(&cuModule, cubin);
    if (res != CUDA_SUCCESS) {
        #ifdef __KERNEL__
        PRINT(V_INFO, "cannot load module: %d\n", res);
        #else
        printf("cannot acquire device 0\n");
        #endif
    }

    res = cuModuleGetFunction(func, cuModule, kname);
    if (res != CUDA_SUCCESS){
        #ifdef __KERNEL__
        PRINT(V_INFO, "cannot acquire kernel handle\n");
        #else
        printf("cannot acquire device 0\n");
        #endif
    }
}
