#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <inttypes.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <pthread.h>
#include <unistd.h>
#include <thread>
#include <random>

uint8_t N_THREADS = 1;

#define LARGEST_REQUEST_BYTES (64*1024*1024)
#define MEM_ALIGN 4096
int KB = 1024; int MB = 1024*1024; int GB = 1024*1024*1024;

int iops[] = { 10, 100, 1000, 2000, 5000, 10000};
int sizes[] = {4*KB, 16*KB, 64*KB, 256*KB, 1024*KB};

int fds[] = {0,0,0};
uint32_t lates[] = {0,0,0};

uint32_t runtime_us = 100000;
    
struct Thread_arg {
    uint32_t device;
    uint32_t idx;
    pthread_barrier_t *sync_barrier;
};

void* replayer_fn(void* arg) {
    Thread_arg *targ = (Thread_arg*) arg;
    uint32_t device = targ->device;
    int idx = targ->idx; 
    uint32_t iop = iops[idx];
    uint32_t size = sizes[idx];
    char *buf;
    bool is_late;
    int fd = fds[device];
    int ret;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(128*MB, 800*GB); // define the range

    if (posix_memalign((void**)&buf, MEM_ALIGN, LARGEST_REQUEST_BYTES)) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    //start together
    pthread_barrier_wait(targ->sync_barrier);
    auto start = std::chrono::steady_clock::now();
    double period_us = (1.0/(double)iop)*1000000; //period in us

    while (1) {
        auto now = std::chrono::steady_clock::now();
        uint32_t elaps = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();

        if (elaps >= runtime_us) {
            printf("Thread for dev %d done\n", device);
            break;
        }

        auto begin = std::chrono::steady_clock::now();
        ret = pread(fd, buf, size, distr(gen));
        if (ret < 0) {
            printf("Error on pread %d\n", errno);
            break;
        }
        auto end = std::chrono::steady_clock::now();        
        uint32_t latency =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        //TODO store latency somewhere so that later we can calculate avg and stddev

        if(latency > period_us) {
            //if we are late, we cant even sleep
            lates[device]++;
        } else {
            //sleep for period_us - latency
        }



    }
    free(buf);
    return 0;
}




int main (int argc, char **argv)
{   

    for (int i=0; i < 3 ; i++) {
        //TODO open fds, check replayer.hpp for "open" to know how to
    }


    Thread_arg args[3];
    //XXX, change to 6 to run all experiments
    //reset state
    for (int idx=0; idx < 2 ; idx++) {
    
        pthread_barrier_t sync_barrier;
        int err = pthread_barrier_init(&sync_barrier, NULL, 3+1);
        if (err != 0) {
            printf("Error creating barrier\n");
            exit(1);
        }

        for (int i=0; i < 3 ; i++) {
            lates[i] = 0;
            args[i].device = i;
            args[i].idx = idx;
            args[i].sync_barrier = &sync_barrier;
        }

        //TODO launch threads
        //TODO hit barrier to make everyone start at the same time.
    



    //below is some code from replayer.cpp





    pthread_t threads[n_devices_to_trace][N_THREADS];
    Thread_arg targs[n_devices_to_trace][N_THREADS];
    for (int dev=0; dev < n_devices_to_trace ; dev++) {
        for (int j = 0; j < N_THREADS; j++) {
            targs[dev][j].trace = &trace;
            targs[dev][j].device = dev;
            targs[dev][j].sync_barrier = &sync_barrier;

            if(type == "baseline")
                targs[dev][j].executor = baseline_execute_op;
            else if (type == "strawman") {
                targs[dev][j].executor = strawman_execute_op;
            } else if (type == "failover") {
                targs[dev][j].executor = failover_execute_op;
            } else if (type == "strawman2") {
                targs[dev][j].executor = strawman_2ssds_execute_op;
            } else {
                printf("I dont recognize type %s (second parameter)\n", type.c_str());
            }
            pthread_create(&threads[dev][j], NULL, replayer_fn, (void*)&targs[dev][j]);
        }
    }

    trace.set_output_file(metrics_file+"_"+type+".data");

    usleep(20); //wait until everyone hits barrier
    uint64_t now = get_ns_ts();
    //give threads most up do date starting time
    for (int dev=0; dev < n_devices_to_trace ; dev++) 
        for (int j = 0; j < N_THREADS; j++)
            targs[dev][j].start_ts = now;

    auto begin = std::chrono::steady_clock::now();
    //start workers
    pthread_barrier_wait(&sync_barrier);
    //wait for workers
    for (int dev=0; dev < n_devices_to_trace ; dev++) {
        for (int j = 0; j < N_THREADS; j++)
            pthread_join(threads[dev][j], 0);
    }
    auto end = std::chrono::steady_clock::now();
    uint64_t elaps =  std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

    printf("Trace took %lu seconds to finish.\n", elaps);

    trace.print_stats();

    return 0;
}
