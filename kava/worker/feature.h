#ifndef __KAVA_WORKER_FEATURE_H__
#define __KAVA_WORKER_FEATURE_H__

int kava_shm_init(void);
void kava_shm_fini(void);
void *kava_shm_address(long offset);

#endif
