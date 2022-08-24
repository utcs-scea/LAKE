#ifndef __KAPI_LAKE_H__
#define __KAPI_LAKE_H__

#include <inttypes.h>

int lake_init_socket();
void lake_destroy_socket();
void lake_send_cmd(uint32_t seqn);

#endif