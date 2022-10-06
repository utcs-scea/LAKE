#ifndef __C_WRAPPER___
#define __C_WRAPPER__

#include <Python.h>
#include <stdio.h>

int load_model(const char *filepath);
void dogc(void);
void close_ctx(void);
int standard_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window);

int kleio_load_model(const char *filepath);
int kleio_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window);
void kleio_close_ctx(void);
#endif
