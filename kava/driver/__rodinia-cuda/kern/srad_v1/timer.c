#include <linux/ktime.h>

 // Returns the current system time in nanoseconds
long get_time(void) {
	struct timespec tv;
	getnstimeofday(&tv);
	return (tv.tv_sec * 1000000000) + tv.tv_nsec;
}
