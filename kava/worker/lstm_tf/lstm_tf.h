#ifndef __LSTM_TF_H__
#define __LSTM_TF_H__

#ifdef __KERNEL__
#include <linux/random.h>
#include <linux/module.h> /* Needed by all modules */
#include <linux/kernel.h> /* Needed for KERN_INFO */
#else
#ifndef __CAVA__
#include <stdio.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

int load_model(const char *file);
void close_ctx(void);
int standard_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window);

int kleio_load_model(const char *file);
int kleio_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window);

#ifdef __cplusplus
}
#endif

#endif
