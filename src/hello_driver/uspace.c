#include <stdio.h>
#include "cuda.h"


static inline CUresult check_error(CUresult error, const char* error_str, int line)
{
	if (error != CUDA_SUCCESS) {
		printf("ERROR: returned error %d (line %d): %s\n", error, line, error_str);
	}
	return error;
}


static int run_hello(void)
{
    int compute_mode, i;
    CUdevice dev;
    char dev_name[128];
    int compute_major, compute_minor;

    CUcontext ctx;
    CUmodule mod;
    CUfunction hello_kernel;
	//int val[10];
	int* val;	
	CUdeviceptr d_p1;

	printf("Running hello world\n");

    cuInit(0);

	// Get the GPU ready
	check_error(cuDeviceGet(&dev, 0), "cuDeviceGet", __LINE__);
	// check_error(cuDeviceGetAttribute(&compute_mode, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, dev), "cuDeviceGetAttribute", __LINE__);

	// if (compute_mode == CU_COMPUTEMODE_PROHIBITED) {
	// 	printk(KERN_ERR "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cuSetDevice().\n");
	// 	return 0;
	// }

	// check_error(cuDeviceGetName(dev_name, 128, dev), "cuDeviceGetName", __LINE__);
	// check_error(cuDeviceGetAttribute(&compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "cuDeviceGetAttribute", __LINE__);
	// check_error(cuDeviceGetAttribute(&compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "cuDeviceGetAttribute", __LINE__);
	// PRINT(V_INFO, "GPU Device %d: \"%s\" with compute capability %d.%d\n",
    //         devID, dev_name, compute_major, compute_minor);

	check_error(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate", __LINE__);

    check_error(cuModuleLoad(&mod, "/home/hfingler/hf-HACK/src/hello_driver/hello.cubin"), "cuModuleLoad", __LINE__);
    check_error(cuModuleGetFunction(&hello_kernel, mod, "_Z12hello_kernelPii"),
            "cuModuleGetFunction", __LINE__);

	val = malloc(10*sizeof(int));
	for (i = 0; i < 10; i++)
		val[i] = i;

	check_error(cuMemAlloc((CUdeviceptr*) &d_p1, 128), "cuMemAlloc d_p1", __LINE__);
	check_error(cuMemcpyHtoD(d_p1, val, 10), "cuMemcpyHtoD", __LINE__);

	int count = 10;
	void *args[] = {
		&d_p1, &count
	};

	check_error(cuLaunchKernel(hello_kernel, 
				1, 1, 1,
				10, 1, 1, 
				0, NULL, args, NULL),
			"cuLaunchKernel", __LINE__);

	check_error(cuMemcpyDtoH(val, d_p1, 10), "cuMemcpyDtoH", __LINE__);

	printf("Printing resulting array: \n");
	for (i = 0; i < 10; i++)
		printf(" %d", val[i]);

	cuMemFree(d_p1);

	free(val);

 	// check_error(cuCtxDestroy(ctx), "cuCtxDestroy", __LINE__);
	return 0;
}

/**
 * Program main
 */
int main(void)
{
	return run_hello();
}
