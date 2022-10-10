#ifndef __IO_OPS_H
#define __IO_OPS_H

#include <stdint.h>
#include <stdio.h>

int64_t atomic_read(int64_t* ptr);
void atomic_add(int64_t* ptr, int val);
int64_t atomic_fetch_inc(int64_t* ptr);

void *perform_io_failover(void *input);
void *perform_io_baseline(void *input);

#endif