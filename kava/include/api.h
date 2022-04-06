#ifndef __KAVA_API_H__
#define __KAVA_API_H__

#ifdef __KERNEL__

#include <linux/kobject.h>

#endif

#include "channel.h"

typedef enum {
    KAVA_API_ID_UNSPEC = 0,
    KAVA_API_ID_CUDA,
    KAVA_API_ID_GENANN,
    KAVA_API_ID_MVNC,
    KAVA_API_ID_MAX,
} kava_api_id;

static const char* const kava_api_name[KAVA_API_ID_MAX] = {
    "unspec",
    "cuda",
    "genann",
    "mvnc",
};

#ifdef __KERNEL__

struct kava_api {
    unsigned id;
    const char *name;

    /* sysfs entries */
    struct kobject *sysfs_kobj;
    kava_chan_mode chan_mode;
    char worker_path[128];  // Deprecated but used in chan_file_poll.c
};

extern struct kava_api *global_kapi;

void init_global_kapi(kava_api_id id, const char *chan_mode);
void put_global_kapi(void);

#endif

#endif
