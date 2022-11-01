#ifndef __C_WRAPPER___
#define __C_WRAPPER__

#include <Python.h>
#include <stdio.h>
#include <stdint.h>

int      kleio_load_model(const char *filepath);
uint64_t kleio_inference(const void *syscalls, unsigned int n, unsigned int usegpu);
void     kleio_force_gc(void);

#endif
