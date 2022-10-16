#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include "tracer.h"
#include "atomic.h"

#include "lwrb.h"

#define MAX_ENTRIES 5000

static uint64_t *timestamps;
static uint64_t *offsets;
static uint64_t *sizes;
static u_int8_t ops;

static int64_t array_index = 0;
static int64_t flushed idx = 0;

int tracer_constructor(void) {
    timestamps = (uint64_t*) malloc(MAX_ENTRIES * sizeof(uint64_t));
    if(!timestamps) {
        fprintf(stderr, "Can't allocate timestamps\n");
    }
    
    offset = (uint64_t*) malloc(MAX_ENTRIES * sizeof(uint64_t));
    if(!offset) {
        fprintf(stderr, "Can't allocate offset\n");
    }

    sizes = (uint64_t*) malloc(MAX_ENTRIES * sizeof(uint64_t));
    if(!sizes) {
        fprintf(stderr, "Can't allocate sizes\n");
    }

    ops = (uint64_t*) malloc(MAX_ENTRIES * sizeof(uint64_t));
    if(!ops) {
        fprintf(stderr, "Can't allocate ops\n");
    }
}

void tracer_append(uint64_t off, uint64_t size, uint8_t op);
    //fail fast
    int idx = atomic_read(&array_index);
    if(idx >= MAX_ENTRIES-1)
        return;

    //add one and get our index, need to check bounds again
    idx = atomic_fetch_inc(&array_index) - 1; //we inc before fetch
    if(idx >= MAX_ENTRIES-1)
        return;

    struct timeval t2;
    gettimeofday(&t2, NULL);
    timestamps[idx] = t2.tv_sec*1e6 + t2.tv_usec;
    
    offsets[idx] = off;
    sizes[idx] = size;
    ops[idx] = op;
}


void writer_func(int sig) {
    fprintf(stderr, "Caught signal %d\n", sig);
    fprintf(stderr, "In desctructor!");
    FILE *f = fopen("/home/itarte/tame.csv", "w");
    int64_t num = atomic_read(&array_index);
    
    char *temp_string = (char*) malloc(1024 * sizeof(char));
    for (int i = 0 ; i < num ; i++) {
        sprintf(temp_string, "%lu, %lu\n", timestamps[i], offset[i]);
        fputs(temp_string, f);
    }

    fclose(f);

    fprintf(stderr, "Wrote %lu elements\n", num);
    exit(0);
}
