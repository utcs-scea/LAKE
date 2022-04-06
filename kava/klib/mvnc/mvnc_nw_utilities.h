#ifndef __MVNC_NW_UTILITIES_H__
#define __MVNC_NW_UTILITIES_H__

#define ava_utility static
#define ava_begin_utility 
#define ava_end_utility 

#undef ava_utility
#undef ava_begin_utility 
#undef ava_end_utility 

#ifdef __KERNEL__
#include <linux/time.h>
#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)

#define ELAPSED_TIME_MICRO_SEC(start, stop) ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000)
#endif

#endif // ndef __MVNC_NW_UTILITIES_H__
