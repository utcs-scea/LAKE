#ifndef __KAPI_LAKE_H__
#define __KAPI_LAKE_H__

#include <linux/ctype.h>
#include "cuda.h"


int lake_init_socket(void);
void lake_destroy_socket(void);
CUresult lake_send_cmd(void *buf, size_t size, char sync, struct lake_cmd_ret* ret);

#endif