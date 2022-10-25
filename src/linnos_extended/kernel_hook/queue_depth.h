#ifndef __QDEPTH_H
#define __QDEPTH_H

#include <linux/types.h>

int qd_init(void);
void append_qdepth(u32);
void qd_writeout(void);

#endif