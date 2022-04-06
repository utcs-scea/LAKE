#ifndef __KAVA_CONFIG_H__
#define __KAVA_CONFIG_H__

#include "api.h"
#include "debug.h"

#define KAVA_DEV_MAJOR 151
#define KAVA_DEV_MINOR 99
#define KAVA_DEV_NAME  "kava_dev"
#define KAVA_DEV_CLASS "kava_dev"

#define KAVA_SHM_DEV_MAJOR 152
#define KAVA_SHM_DEV_MINOR 99
#define KAVA_SHM_DEV_NAME  "kava_shm"
#define KAVA_SHM_DEV_CLASS "kava_shm"

int create_sysfs_entry(void);
void put_sysfs_entry(void);

#endif
