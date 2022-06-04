#include <stdio.h>
#include <cuda.h>

int main(){
    cuInit(0);

    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot acquire device 0\n");
        exit(1);
    }

    CUcontext cuContext;
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        printf("cannot create context\n");
        exit(1);
    }

    CUmodule cuModule;
    res = cuModuleLoad(&cuModule, "/home/hyu/kava/driver/cuda/hello_world.cubin");
    if (res != CUDA_SUCCESS) {
        printf("cannot load module: %d\n", res);
        exit(1);
    }

    CUfunction helloWorld;
    res = cuModuleGetFunction(&helloWorld, cuModule, "_Z10HelloWorldv");
    if (res != CUDA_SUCCESS){
        printf("cannot acquire kernel handle\n");
        exit(1);
    }

    int blocks_per_grid = 4;
    int threads_per_block = 5;

    void* args[] = {};
    res = cuLaunchKernel(helloWorld, blocks_per_grid, 1, 1, threads_per_block, 1, 1, 0, 0, args, 0);
    if (res != CUDA_SUCCESS){
        printf("cannot run kernel\n");
        exit(1);
    }

    cuCtxDestroy(cuContext);

    return 0;
}
