#ifndef KUDA_KAVA_KSM
#define KUDA_KAVA_KSM
#include <cuda_kava.h>

/**
  * Register KAVA CUDA API function pointers
  * CALL THIS FIRST
  */
extern int register_cuda_functions(struct kava_cuda_fn_table *src);

/**
  * Initialize KAVA CUDA API
  * CALL THIS SECOND
  */
extern int ksm_kava_init(void);

#endif
