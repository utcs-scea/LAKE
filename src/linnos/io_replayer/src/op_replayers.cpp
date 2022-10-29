/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include <errno.h>
#include "op_replayers.hpp"

// static int64_t get_next_multiple(uint64_t A, uint64_t B) {
//     if (A % B)
//         A = A + (B - A % B);
//     return A;
// }

#define MAX_FAIL 2

static int sleep_until(uint64_t next) {
    uint64_t now = get_ns_ts();
    int64_t diff = next - now;

    //if 0 or negative, we need to issue
    if(diff <= 0) {
        //we're late by at least 2 us
        if (diff <= -2000) return 1;
        return 0; //late but not that much
    }
    else 
        std::this_thread::sleep_for(std::chrono::nanoseconds(diff));
    return 0;
}

void baseline_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf) {
    int ret;
    int *fds = trace->get_fds();
    //read
    if(trace_op.op == 0) {
        trace->add_io_count(device);
        ret = pread(fds[device], buf, trace_op.size, trace_op.offset);
    } else if(trace_op.op == 1) {
        trace->add_io_count(device);
        ret = pwrite(fds[device], buf, trace_op.size, trace_op.offset);
    } else {
        printf("Wrong OP code! %d\n", trace_op.op);
    }

    if (ret < 0){
        printf("err %d\n", errno);
        printf("offset in B : %lu\n", trace_op.offset );
        printf("size in B : %lu\n", trace_op.size);
    }
}

void strawman_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf) {
    int ret;
    int *fds = trace->get_fds();
    //read
    if(trace_op.op == 0) {
        trace->add_io_count(device);
        ret = pread(fds[device], buf, trace_op.size, trace_op.offset);
        //rejected, go to next device (it should not have linnos enabled)
        if (ret < 0) {
            trace->add_fail(device);
            trace->add_unique_fail(device);
            trace->add_io_count(device+1);
            ret = pread(fds[1], buf, trace_op.size, trace_op.offset);
            if (ret < 0) {
                printf("Second IO failed, this shouldn't happen! err %d\n", errno);
                trace->add_never_finished(device);
            }
        }
    } else if(trace_op.op == 1) {
        trace->add_io_count(device);
        ret = pwrite(fds[device], buf, trace_op.size, trace_op.offset);
    } else {
        printf("Wrong OP code! %d\n", trace_op.op);
    }
}

void strawman_2ssds_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf) {
    int ret;
    int *fds = trace->get_fds();
    //read
    if(trace_op.op == 0) {
        trace->add_io_count(device);
        ret = pread(fds[device], buf, trace_op.size, trace_op.offset);
        //rejected, go to next device (it should not have linnos enabled)
        if (ret < 0) {
            trace->add_fail(device);
            trace->add_unique_fail(device);
            trace->add_io_count(2);
            ret = pread(fds[2], buf, trace_op.size, trace_op.offset);
            if (ret < 0) { 
                printf("Second IO failed, this shouldn't happen! err %d\n", ret);
                trace->add_never_finished(device);
            }
        }
    } else if(trace_op.op == 1) {
        trace->add_io_count(device);
        ret = pwrite(fds[device], buf, trace_op.size, trace_op.offset);
    } else {
        printf("Wrong OP code! %d\n", trace_op.op);
    }
}

void failover_execute_op(TraceOp &trace_op, Trace *trace, uint32_t device, char* buf) {
    int ret, i;
    int *fds = trace->get_fds();
    bool success = false;
    //read
    if(trace_op.op == 0) {
        for (i = 0 ; i < MAX_FAIL ; i++) {
            trace->add_io_count((device+i)%2);
            ret = pread(fds[(device+i)%2], buf, trace_op.size, trace_op.offset);
            if (ret > 0) {
                success = true;
                break;
            }
            trace->add_fail(device);
        }
        //max fail.. it looped around, linnos never handled this case
        if (!success) {
            //printf("IO never finished..\n");
            trace->add_unique_fail(device);
            pread(fds[device], buf, trace_op.size, 0); //this is what linnos does
        }
    } else if(trace_op.op == 1) {
        trace->add_io_count(device);
        ret = pwrite(fds[device], buf, trace_op.size, trace_op.offset);
    } else {
        printf("Wrong OP code! %d\n", trace_op.op);
    }
}

void* replayer_fn(void* arg) {
    Thread_arg *targ = (Thread_arg*) arg;
    Trace *trace = targ->trace;
    uint32_t device = targ->device;
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
            trace->add_late_op(device);

        uint64_t submission = get_ns_ts();
        auto begin = std::chrono::steady_clock::now();
        //realize trace_op
        targ->executor(trace_op, trace, targ->device, buf);
        auto end = std::chrono::steady_clock::now();
        uint32_t elaps =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        uint64_t end_ts = get_ns_ts();

        //store results
        trace->write_output_line(end_ts/1000, elaps, trace_op.op,
                trace_op.size, trace_op.offset, submission/1000,
                device);
    }
    free(buf);
    return 0;
}
