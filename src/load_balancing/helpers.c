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


#include "helpers.h"
#include "consts.h"

void* w1_shm;
void* b1_shm;
void* w2_shm;

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
        printf("cannot create context\n");
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

    w1_shm = kava_alloc(NR_FEAT*10*sizeof(float));
    b1_shm = kava_alloc(10*sizeof(float));
    w2_shm = kava_alloc(10*sizeof(float));

    memcpy(w1_shm, w1, NR_FEAT*10*sizeof(float));
    memcpy(b1_shm, b1, 10*sizeof(float));
    memcpy(w2_shm, w2, 10*sizeof(float));

    check_error(cuMemcpyHtoD(*d_w1, w1_shm, NR_FEAT*10*sizeof(float)), "cuMemcpyHtoD", __LINE__);
    check_error(cuMemcpyHtoD(*d_b1, b1_shm, 10*sizeof(float)), "cuMemcpyHtoD", __LINE__);
    check_error(cuMemcpyHtoD(*d_w2, w2_shm, 10*sizeof(float)), "cuMemcpyHtoD", __LINE__);
}

void gpu_clean(CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, CUdeviceptr d_results) {
    kava_free(w1_shm);
    kava_free(b1_shm);
    kava_free(w2_shm);
    
    cuMemFree(d_inputs);
    cuMemFree(d_w1);
    cuMemFree(d_b1);
    cuMemFree(d_w2);
    cuMemFree(d_results);
}

void gpu_setup_inputs(CUdeviceptr d_inputs, int* inputs, int n) {
    check_error(cuMemcpyHtoDAsync(d_inputs, inputs, n*NR_FEAT*sizeof(float), 0), "cuMemcpyHtoD", __LINE__);
}

int gpu_inference_many(CUfunction* cufunc, int n_inputs,
        CUdeviceptr d_inputs, CUdeviceptr d_w1, CUdeviceptr d_b1, CUdeviceptr d_w2, float b2, CUdeviceptr d_results, int sync) {
    int total_threads = n_inputs * 16;
    int blocks = total_threads / 128;
    if (blocks == 0) blocks = 1;

    void *args[] = {
		&d_inputs, &d_w1, &d_b1, &d_w2, &b2, &d_results
	};

    check_error(cuLaunchKernel(*cufunc, 
				blocks, 1, 1,      //blocks
				128, 1, 1,          //threads per block
				10*8*sizeof(float), //shared mem
                NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

    if (sync) cuCtxSynchronize();

    return 0;
}

int gpu_get_result(int n_inputs, CUdeviceptr d_results, float* outs) {
    cuMemcpyDtoH(outs, d_results, n_inputs*sizeof(float));
    return 0;
}
