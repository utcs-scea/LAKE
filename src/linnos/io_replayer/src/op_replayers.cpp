#include <errno.h>
#include "op_replayers.hpp"

// static int64_t get_next_multiple(uint64_t A, uint64_t B) {
//     if (A % B)
//         A = A + (B - A % B);
//     return A;
// }

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
        //ret = pread(fd, buf, 512, 1024*1024*16);
        //
        ret = pread(fd, buf, trace_op.size, trace_op.offset);
    } else if(trace_op.op == 1) {
        ret = pwrite(fd, buf, trace_op.size, trace_op.offset);
    } else {
        printf("Wrong OP code! %d\n", trace_op.op);
    }

    if (ret < 0){
        printf("err %d\n", errno);
        printf("offset in B : %lu\n", trace_op.offset );
        printf("size in kb : %lu\n", trace_op.size/(1e3));
    }
}

void* replayer_fn(void* arg) {
    Thread_arg *targ = (Thread_arg*) arg;
    Trace *trace = targ->trace;
    uint8_t device = targ->device;
    TraceOp trace_op;
    char *buf;

    if (posix_memalign((void**)&buf, MEM_ALIGN, LARGEST_REQUEST_BYTES)) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    //start together
    pthread_barrier_wait(targ->sync_barrier);
    int is_late;
    while (1) {
        trace_op = trace->get_line(device);
        if (trace_op.timestamp == -1) {
            break;
        }

        //timestamp in op is in microsecond float, so convert to nano
        uint64_t next = targ->start_ts + (uint64_t)(trace_op.timestamp*1000);
        if(sleep_until(next) == 1)
            trace->add_late_op();

        uint64_t submission = get_ns_ts();
        auto begin = std::chrono::steady_clock::now();
        //realize trace_op
        targ->executor(trace_op, trace->get_fd(device), buf);
        auto end = std::chrono::steady_clock::now();
        uint64_t elaps =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        uint64_t end_ts = get_ns_ts();

        //store results
        trace->write_output_line(end_ts/1000, elaps, trace_op.op,
                trace_op.size, trace_op.offset, submission/1000);
    }

    free(buf);
    return 0;
}
