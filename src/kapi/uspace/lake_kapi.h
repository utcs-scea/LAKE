#ifndef __KAPI_LAKE_H__
#define __KAPI_LAKE_H__

#include <inttypes.h>
#include "commands.h"

int lake_init_socket();
void lake_handle_cmd(void* buf, struct lake_cmd_ret* cmd_ret);
void lake_recv();
void lake_destroy_socket();

//shm helpers
int lake_shm_init(void);
void lake_shm_fini(void);
void *lake_shm_address(const void* offset);


#endif