#ifndef __SHIM_TRACER_H
#define __SHIM_TRACER_H

#include <stdint.h>

int tracer_constructor(void);
void tracer_append(uint64_t off, uint64_t size, uint8_t op);

#endif