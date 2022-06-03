#ifndef __UTIL_H__
#define __UTIL_H__

#include <linux/ktime.h>
#include <linux/timekeeping.h>

#include "cuda_kava.h"

CUresult cuda_driver_api_init(CUcontext *pctx, CUmodule *pmod, const char *f);
CUresult cuda_driver_api_exit(CUcontext ctx, CUmodule mod);

/* tvsub: ret = x - y. */
static inline void tvsub(struct timespec *x,
						 struct timespec *y,
						 struct timespec *ret)
{
	ret->tv_sec = x->tv_sec - y->tv_sec;
	ret->tv_nsec = x->tv_nsec - y->tv_nsec;
	if (ret->tv_nsec < 0) {
		ret->tv_sec--;
		ret->tv_nsec += 1000000000;
	}
}

struct timestamp {
    struct timespec start;
    struct timespec end;
};

void probe_time_start(struct timestamp *ts);
long probe_time_end(struct timestamp *ts);

#endif
