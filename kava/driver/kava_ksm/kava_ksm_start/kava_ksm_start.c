#include <linux/err.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>

#include <cuda_kava_ksm.h>
#include <cuda_kava.h>
#include <shared_memory.h>

static volatile char dev_initialized = 0;
static int __init kava_ksm_start_init(void) {
  int res;
  struct kava_cuda_fn_table tab = { 0 };
  printk("[kava_ksm_start]: Begin initialization");

  // Copy funciton pointers
  tab.cuInit =               cuInit;
  tab.cuDeviceGet =          cuDeviceGet;
  tab.cuCtxCreate =          cuCtxCreate;
  tab.cuModuleLoad =         cuModuleLoad;
  tab.cuModuleUnload =       cuModuleUnload;
  tab.cuModuleGetFunction =  cuModuleGetFunction;
  tab.cuLaunchKernel =       cuLaunchKernel;
  tab.cuCtxDestroy =         cuCtxDestroy;
  tab.cuStreamCreate =       cuStreamCreate;
  tab.cuStreamSynchronize =  cuStreamSynchronize;
  tab.cuStreamDestroy =      cuStreamDestroy;
  tab.cuMemAlloc =           cuMemAlloc;
  tab.cuMemcpyHtoD =         cuMemcpyHtoD;
  tab.cuMemcpyDtoH =         cuMemcpyDtoH;
  tab.cuMemcpyHtoDAsync =    cuMemcpyHtoDAsync;
  tab.cuMemcpyDtoHAsync =    cuMemcpyDtoHAsync;
  tab.cuCtxSynchronize =     cuCtxSynchronize;
  tab.cuDriverGetVersion =   cuDriverGetVersion;
  tab.cuMemFree =            cuMemFree;
  tab.cuModuleGetGlobal =    cuModuleGetGlobal;
  tab.cuDeviceGetCount =     cuDeviceGetCount;
  tab.cuFuncSetCacheConfig = cuFuncSetCacheConfig;
  tab.cuCtxSetCurrent =      cuCtxSetCurrent;
  tab.__kava_stop_shadow_thread = __kava_stop_shadow_thread;
  tab.kava_alloc = kava_alloc;

  // Register function pointers and init ksm
  res = register_cuda_functions(&tab);
  if (res) {
    printk(KERN_ERR "[kava_ksm_start] Error: couldn't register cuda functions");
    return res;
  }
  printk("[kava_ksm_start] Registered KAVA CUDA functions");
  res = ksm_kava_init();
  __kava_stop_shadow_thread();
  if (res) {
    printk(KERN_ERR "[kava_ksm_start] Error: couldn't initialize kava");
    return res;
  }
  printk("[kava_ksm_start] Initialized kava_ksm");
  return 0;
}

static void __exit kava_ksm_start_exit(void) {
}

module_init(kava_ksm_start_init);
module_exit(kava_ksm_start_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Module to switch KSM to use KAVA CUDA");
