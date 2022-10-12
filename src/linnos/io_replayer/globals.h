#ifndef __GLOBALS_H
#define __GLOBALS_H

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

enum {
    READ = 1,
    WRITE = 0,
};

#define NR_DEVICE 3
#define MAX_FAIL 4

extern int LARGEST_REQUEST_SIZE; //blocks
extern int MEM_ALIGN; //bytes
extern int nr_workers[NR_DEVICE];
extern int printlatency; //print every io latency
extern int respecttime;
extern int block_size; // by default, one sector (512 bytes)
extern int single_io_limit;


extern int dev_idx_enum[NR_DEVICE];
extern int fd[NR_DEVICE];
extern int64_t DISKSZ[NR_DEVICE];
extern int64_t nr_tt_ios;
extern int64_t nr_ios[NR_DEVICE];
extern int64_t latecount;
extern int64_t slackcount;
extern uint64_t starttime;

extern int64_t *oft[NR_DEVICE];
extern int *reqsize[NR_DEVICE];
extern int *reqflag[NR_DEVICE];
extern float *timestamp[NR_DEVICE];

extern FILE *metrics; // current format: offset,size,type,latency(ms)
extern FILE *metrics_sub;

extern pthread_mutex_t lock; 
extern pthread_barrier_t sync_barrier;

extern int64_t jobtracker[NR_DEVICE];

#endif