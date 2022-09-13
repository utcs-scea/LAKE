#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include "hello.h"
#include <linux/delay.h>

//#include <linux/bpf.h>
//#include <linux/btf_ids.h>

static int devID = 0;
module_param(devID, int, 0444);
MODULE_PARM_DESC(devID, "GPU device ID in use, default 0");

static char *cubin_path = "hello.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to firewall.cubin, default ./firewall.cubin");

/*
 *  Test kfuncs
 */

void noinline hello_test_kfunc_trigger(void) {
	PRINT(V_INFO, "Called hello_test_kfunc_trigger, do something kfunc\n");
}
EXPORT_SYMBOL(hello_test_kfunc_trigger);

void noinline hello_test_kfunc(void) {
	PRINT(V_INFO, "Hello from kfunc!\n");
}
EXPORT_SYMBOL(hello_test_kfunc);

static int run_hello(void)
{
    int i, j;
    CUdevice dev;
	int count;
    CUcontext ctx;	
    CUmodule mod;
    CUfunction hello_kernel;
	int* val;	
	CUdeviceptr d_p1;
	u64 t_start, t_stop, t_stop2;

	PRINT(V_INFO, "Running hello world\n");

	// Get the GPU ready
    cuInit(0);
	check_error(cuDeviceGet(&dev, devID), "cuDeviceGet", __LINE__);
	check_error(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate", __LINE__);
    check_error(cuModuleLoad(&mod, cubin_path), "cuModuleLoad", __LINE__);
    check_error(cuModuleGetFunction(&hello_kernel, mod, "_Z12hello_kernelPii"),
            "cuModuleGetFunction", __LINE__);

	val = kava_alloc(10*sizeof(int));
	for (i = 0; i < 10; i++)
		val[i] = 0;

	t_start = ktime_get_ns();
	check_error(cuMemAlloc((CUdeviceptr*) &d_p1, 128), "cuMemAlloc d_p1", __LINE__);
	check_error(cuMemcpyHtoD(d_p1, val, 10), "cuMemcpyHtoD", __LINE__);
	count = 10;
	void *args[] = {
		&d_p1, &count
	};

	for (j=0 ; j < 16 ; j++) {
		check_error(cuLaunchKernel(hello_kernel, 
					1, 1, 1,
					10, 1, 1, 
					0, NULL, args, NULL),
				"cuLaunchKernel", __LINE__);
	}
	t_stop = ktime_get_ns();
	cuCtxSynchronize();
	t_stop2 = ktime_get_ns();

	PRINT(V_INFO, "Times (us): %llu, %llu\n", (t_stop - t_start)/1000, (t_stop2 - t_start)/1000);
	check_error(cuMemcpyDtoH(val, d_p1, 10*sizeof(int)), "cuMemcpyDtoH", __LINE__);
	PRINT(V_INFO, "Printing resulting array: \n");
	for (i = 0; i < 10; i++)
		PRINT(V_INFO, " %d", val[i]);

	cuCtxSynchronize();
	cuMemFree(d_p1);
	kava_free(val);

	for (i = 0 ; i < 20 ; i++) {
		hello_test_kfunc_trigger();
		msleep(1000);
	}


 	// check_error(cuCtxDestroy(ctx), "cuCtxDestroy", __LINE__);
	return 0;
}


// BTF_SET_START(hello_test_kfunc_ids)
// BTF_ID(func, hello_test_kfunc)
// BTF_SET_END(hello_test_kfunc_ids)

// bool bpf_prog_test_check_kfunc_call(u32 kfunc_id)
// {
// 	return btf_id_set_contains(&hello_test_kfunc_ids, kfunc_id);
// }



/**
 * Program main
 */
static int __init hello_init(void)
{
	return run_hello();
}

static void __exit hello_fini(void)
{

}

module_init(hello_init);
module_exit(hello_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Kernel module of a hello program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
