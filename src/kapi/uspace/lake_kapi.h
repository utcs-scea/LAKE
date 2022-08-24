#ifndef __KAPI_LAKE_H__
#define __KAPI_LAKE_H__

#include <inttypes.h>

int lake_init_socket();
void lake_recv_loop(volatile sig_atomic_t *stop);
void lake_handle_cmd(void* buf, struct lake_cmd_ret* cmd_ret);

void lake_recv();
void lake_destroy_socket();

#endif