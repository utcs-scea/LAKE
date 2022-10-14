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

#include "replayer.hpp"
#include "op_replayers.hpp"

uint8_t N_THREADS = 64;

int main (int argc, char **argv)
{
    if (argc < 4) {
        printf("Usage: ./replayer <baseline|failover> logfile /dev/tgt0-/dev/tgt1-/dev/tgt2 <n of devices traces>\n");
        exit(1);
    } 
    
    std::string metrics_file(argv[2]);
    Trace trace(argv[3]);

    for (int i=0; i < trace.get_ndevices() ; i++) {
        printf("parsing trace %d\n", i);
        trace.parse_file(i, argv[4+i]);
    }
    
    std::string type(argv[1]);
    std::string metrics_fname(argv[2]);

    pthread_barrier_t sync_barrier;
    int err = pthread_barrier_init(&sync_barrier, NULL, N_THREADS+1);
    if (err != 0) {
        printf("Error creating barrier\n");
        exit(1);
    }

    if (type == "baseline") {
        trace.set_output_file(metrics_file+type);
    } else if (type == "failover") {
        trace.set_output_file(metrics_file+type);
    } else {
        printf("Wrong type (second arg)\n");
        return -1;
    }

    pthread_t threads[trace.get_ndevices()][N_THREADS];
    Thread_arg targs[trace.get_ndevices()][N_THREADS];
    for (int dev=0; dev < trace.get_ndevices() ; dev++) {
        for (int j = 0; j < N_THREADS; j++) {
            targs[dev][j].trace = &trace;
            targs[dev][j].device = dev;
            targs[dev][j].sync_barrier = &sync_barrier;
            targs[dev][j].executor = baseline_execute_op;
            pthread_create(&threads[dev][j], NULL, replayer_fn, (void*)&targs[dev][j]);
        }
    }

    usleep(20); //wait until everyone hits barrier
    uint64_t now = get_ns_ts();
    //give threads most up do date starting time
    for (int dev=0; dev < trace.get_ndevices() ; dev++) 
        for (int j = 0; j < N_THREADS; j++)
            targs[dev][j].start_ts = now;

    auto begin = std::chrono::steady_clock::now();
    //start workers
    pthread_barrier_wait(&sync_barrier);
    //wait for workers
    for (int dev=0; dev < trace.get_ndevices() ; dev++) {
        for (int j = 0; j < N_THREADS; j++)
            pthread_join(threads[dev][j], 0);
    }
    auto end = std::chrono::steady_clock::now();
    uint64_t elaps =  std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();

    printf("Trace took %lu seconds to finish.\n", elaps);
    printf("There were %lu late IOs, consider upping the # of threads if too many\n", 
        trace.get_late_op());

    return 0;
}
