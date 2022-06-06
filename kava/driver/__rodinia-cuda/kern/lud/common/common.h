#ifndef _COMMON_H
#define _COMMON_H

#ifdef __KERNEL__
#include <linux/ktime.h>
#include <linux/timekeeping.h>
#else
#include <time.h>
#include <sys/time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MIN(i,j) ((i)<(j) ? (i) : (j))

typedef enum _FUNC_RETURN_CODE {
    RET_SUCCESS,
    RET_FAILURE
}func_ret_t;

typedef struct __stopwatch_t{
    struct timespec begin;
    struct timespec end;
}stopwatch;

void 
stopwatch_start(stopwatch *sw);

void 
stopwatch_stop (stopwatch *sw);

long
get_interval_by_nsec(stopwatch *sw);

long
get_interval_by_usec(stopwatch *sw);

func_ret_t
create_matrix_from_file(int **mp, const char *filename, int *size_p);

func_ret_t
create_matrix_from_random(int **mp, int size);

func_ret_t
create_matrix(int **mp, int size);

func_ret_t
lud_verify(int *m, int *lu, int size);

void
matrix_multiply(int *inputa, int *inputb, int *output, int size);

void
matrix_duplicate(int *src, int **dst, int matrix_dim);

void
print_matrix(int *mm, int matrix_dim);

#ifdef __cplusplus
}
#endif

#endif
