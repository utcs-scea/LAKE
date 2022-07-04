#include "helpers.h"
#include "consts.h"

void gpu_init(int dev, CUcontext *cuctx) {
    CUdevice cuDevice;
    CUresult res;

    cuInit(0);
    res = cuDeviceGet(&cuDevice, dev);
    if (res != CUDA_SUCCESS){
        #ifdef __KERNEL__
        PRINT(V_INFO, "cannot acquire device 0\n");
        #else
        printf("1cannot acquire device 0\n");
        #endif
    }

    res = cuCtxCreate(cuctx, 0, cuDevice);
    if (res != CUDA_SUCCESS){
        #ifdef __KERNEL__
        PRINT(V_INFO, "cannot create context\n");
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

int gpu_inference_many(CUfunction* cufunc, int n_inputs,
        CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, float b2, CUdeviceptr d_results, int sync) {
    int total_threads = n_inputs * 16;
    int blocks = total_threads / 128;
    if (blocks == 0) blocks = 1;

    void *args[] = {
		&d_inputs, &d_w1, &d_b1, &d_w2, &b2, &d_results
	};

    // struct timespec ts;
    // getnstimeofday(&ts);
    // pr_info("kernel>: sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);

    int zg = sync == 0 ? 1 : 69; 

    check_error(cuLaunchKernel(*cufunc, 
				blocks, 1, zg,      //blocks
				128, 1, 1,          //threads per block
				10*8*sizeof(float), //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    //getnstimeofday(&ts);
    //pr_info("kernel<: sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);

    //cuCtxSynchronize();
    //getnstimeofday(&ts);
    //pr_info("sync: sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);

    return 0;
}


int gpu_get_result(int n_inputs, CUdeviceptr d_results, float* outs) {
    float res[n_inputs];
    cuMemcpyDtoH(outs, d_results, n_inputs*sizeof(float));
    return 0;
}
