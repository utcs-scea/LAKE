#include "op_replayers.hpp"



static int sleep_until(uint64_t next) {
    uint64_t now = get_ns_ts();
    int64_t diff = next - now;

    //if 0 or negative, we need to issue
    if(diff <= 0) {
        //we're super late
        if (diff < 1000) return 1;
        return 0; //late but not that much
    }
    else 
        std::this_thread::sleep_for(std::chrono::nanoseconds(diff));
    return 0;
}

void baseline_execute_op(TraceOp &trace_op, int fd, char* buf) {
    int ret;
    //read
    if(trace_op.op == 0) {
        ret = pread(fd, buf, trace_op.size, trace_op.offset);
    } else if(trace_op.op == 1) {
        ret = pwrite(fd, buf, trace_op.size, trace_op.offset);
    } else {
        printf("Wrong OP code! %d\n", trace_op.op);
    }
}

void* replayer_fn(void* arg) {
    Thread_arg *targ = (Thread_arg*) arg;
    Trace *trace = targ->trace;
    uint8_t device = targ->device;
    TraceOp trace_op;
    char *buf = new char[LARGEST_REQUEST_BYTES];
    //start together
    pthread_barrier_wait(targ->sync_barrier);
    int is_late;
    while (1) {
        trace_op = trace->get_line(device);
        if (trace_op.timestamp == -1) {
            break;
        }

        //timestamp in op is in microsecond float, so convert to nano
        uint64_t next = get_ns_ts() + (uint64_t)(trace_op.timestamp*1000);
        if(sleep_until(next) == 1)
            trace->add_late_op();

        uint64_t now = get_ns_ts();
        //realize trace_op
        targ->executor(trace_op, trace->get_fd(device), buf);
        uint64_t finish = get_ns_ts();

        //store results
        trace->write_output_line(finish/1000, (finish-now)/1000, trace_op.op,
                trace_op.size, trace_op.offset, now/1000);
    }

    delete buf;
    return 0;
}
