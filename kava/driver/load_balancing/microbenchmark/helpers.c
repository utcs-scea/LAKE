#include "helpers.h"
#include "consts.h"

void gpu_init(int dev, CUcontext *cuctx) {
    CUdevice cuDevice;
    CUresult res;

    cuInit(0);
    res = cuDeviceGet(&cuDevice, dev);
    if (res != CUDA_SUCCESS){
        PRINT(V_INFO, "cannot acquire device 0\n");
    }

    res = cuCtxCreate(cuctx, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        PRINT(V_INFO, "cannot create context\n");
    }
}

void gpu_get_cufunc(char* cubin, char* kname, CUfunction *func) {
    CUmodule cuModule;
    CUresult res;
    res = cuModuleLoad(&cuModule, cubin);
    if (res != CUDA_SUCCESS) {
        PRINT(V_INFO, "cannot load module: %d\n", res);
    }

    res = cuModuleGetFunction(func, cuModule, kname);
    if (res != CUDA_SUCCESS){
        PRINT(V_INFO, "cannot acquire kernel handle\n");
    }
}

void gpu_setup(int n_inputs, CUdeviceptr *d_inputs, CUdeviceptr *d_w1, CUdeviceptr *d_b1, CUdeviceptr *d_w2, CUdeviceptr *d_results) {
    check_error(cuMemAlloc((CUdeviceptr*) d_inputs, n_inputs*NR_FEAT*sizeof(float)), "cuMemAlloc ", __LINE__);
    //PRINT(V_INFO, "allocated %ld bytes at %lld for input\n", n_inputs*NR_FEAT*sizeof(float), *d_inputs);
    check_error(cuMemAlloc((CUdeviceptr*) d_w1,     NR_FEAT*10*sizeof(float)), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) d_b1,     10*sizeof(float)), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) d_w2,     10*sizeof(float)), "cuMemAlloc ", __LINE__);
    check_error(cuMemAlloc((CUdeviceptr*) d_results,n_inputs*sizeof(float)), "cuMemAlloc ", __LINE__);
    //PRINT(V_INFO, "allocated\n");
    check_error(cuMemcpyHtoD(*d_w1, w1, NR_FEAT*10*sizeof(float)), "cuMemcpyHtoD", __LINE__);
    check_error(cuMemcpyHtoD(*d_b1, b1, 10*sizeof(float)), "cuMemcpyHtoD", __LINE__);
    check_error(cuMemcpyHtoD(*d_w2, w2, 10*sizeof(float)), "cuMemcpyHtoD", __LINE__);
    //PRINT(V_INFO, "copied weights\n");
}

void gpu_clean(CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, CUdeviceptr d_results) {
    cuMemFree(d_inputs);
    cuMemFree(d_w1);
    cuMemFree(d_b1);
    cuMemFree(d_w2);
    cuMemFree(d_results);
}

void gpu_setup_inputs(CUdeviceptr d_inputs, int* inputs, int n) {
    //PRINT(V_INFO, "copying inputs %ld bytes to %lld\n", n*NR_FEAT*sizeof(float), d_inputs);
    check_error(cuMemcpyHtoD(d_inputs, inputs, n*NR_FEAT*sizeof(float)), "cuMemcpyHtoD", __LINE__);
    //PRINT(V_INFO, "copied\n");
}

// float gpu_inference() {
//     dim3 b(1);
//     dim3 t(10);

//     mllb_infer_v1<<<b, t, 64*sizeof(float)>>>(d_inputs, d_w1, d_b1, NR_FEAT, 10, d_w2, *b2);
//     cudaDeviceSynchronize();
//     return 0;
// }

int gpu_inference_many(CUfunction* cufunc, int n_inputs,
        CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, float b2, CUdeviceptr d_results) {
    int total_threads = n_inputs * 16;
    int blocks = total_threads / 128;
    if (blocks == 0) blocks = 1;

    //PRINT(V_INFO, "Launching with %d blocks and %d threads\n", blocks, 128);

    //mllb_infer_v2<<<b, t, 10*8*sizeof(float)>>>(d_inputs, d_w1, d_b1, d_w2, *b2, d_results);
    //cudaDeviceSynchronize();

    void *args[] = {
		&d_inputs, &d_w1, &d_b1, &d_w2, &b2, &d_results
	};

    check_error(cuLaunchKernel(*cufunc, 
				blocks, 1, 1,          //blocks
				128, 1, 1,   //threads per block
				10*8*sizeof(float),   //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    return 0;
}

int gpu_get_result(int n_inputs) {
    return 0;
    //TODO
    
    //float res[n_inputs];
    //cudaMemcpy(&res, d_results, n_inputs*sizeof(float), cudaMemcpyDeviceToHost);
    //return 0;
}
