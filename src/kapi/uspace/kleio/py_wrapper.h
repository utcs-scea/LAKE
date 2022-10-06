#ifndef __C_WRAPPER___
#define __C_WRAPPER__

#include <Python.h>
#include <stdio.h>

int  kleio_load_model(const char *filepath);
double  kleio_inference(const void *syscalls, unsigned int n);
void kleio_force_gc(void);

#endif
