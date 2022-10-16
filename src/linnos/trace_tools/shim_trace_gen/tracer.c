#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <stdlib.h>
#include "tracer.h"
#include "atomic.h"
#include "lwrb.h"

#define MAX_ENTRIES 523776
#define CONSUME_ENTRIES 32736  // 1/16 of total
const char fpath[] = "/disk/hfingler/captured.trace";

typedef struct trace_line_t {
    uint64_t timestamp;
    uint64_t offset;
    uint64_t size;
    u_int8_t op;
} trace_line;

lwrb_t deque;
void* dq_buffer;
pthread_mutex_t dq_lock;
pthread_t tracer_thread;

void tracer_append(uint64_t off, uint64_t size, uint8_t op) {
    trace_line line;
    line.offset = off;
    line.size = size;
    line.op = op;
    struct timeval t2;
    gettimeofday(&t2, NULL);
    line.timestamp = t2.tv_sec*1e6 + t2.tv_usec;

    pthread_mutex_lock(&dq_lock);
    lwrb_write(&deque, &line, sizeof(trace_line));
    pthread_mutex_unlock(&dq_lock);
}

void* writer_fn(void* arg) {
    FILE *f = fopen(fpath, "w");
    trace_line *buf = malloc(CONSUME_ENTRIES * sizeof(trace_line));
    int bytes_read;

    while (1) {
        pthread_mutex_lock(&dq_lock);
        bytes_read = lwrb_read(&deque, buf, CONSUME_ENTRIES * sizeof(trace_line));
        pthread_mutex_unlock(&dq_lock);

        for (int off = 0 ; off < bytes_read ; off += sizeof(trace_line)) {
            //f"{total_time:.5f} 0 {int(aligned_offset)} {int(aligned_size)} {ops[i]}\n"
            trace_line *cur = buf+off;
            fprintf(f, "%ld 0 %lu %lu %u\n", cur->timestamp, cur->offset, cur->size, cur->op);
        }
        fflush(f);

        int left = lwrb_get_full(&deque);
        //deque is pressured, loop now
        if (left >= CONSUME_ENTRIES * sizeof(trace_line))
            continue;
        else
            usleep(50*1000); //sleep some time, 50ms;
    }
  
    fclose(f);
}

int tracer_constructor(void) {
    dq_buffer = malloc(MAX_ENTRIES* sizeof(trace_line));
    lwrb_init(&deque, dq_buffer, sizeof(MAX_ENTRIES* sizeof(trace_line)));
    pthread_create(&tracer_thread, 0, writer_fn, 0);
    return 0;
}
