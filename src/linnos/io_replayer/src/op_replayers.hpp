#ifndef __OP_REPLAYER_H__
#define __OP_REPLAYER_H__

#include <pthread.h>
#include <unistd.h>
#include <thread>
#include "replayer.hpp"

struct Thread_arg {
    Trace *trace;
    uint32_t device;
    pthread_barrier_t *sync_barrier;
    uint64_t start_ts;

    void (*executor)(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf);
};


void baseline_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf);
void strawman_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf);
void failover_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf);
void strawman_2ssds_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf);

void* replayer_fn(void* arg);


#endif