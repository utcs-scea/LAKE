#include "py_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define SYSCALL_MAX 230
#define MAX_SYSCALL_IDX 340
#define ITERATION 15

#define ELAPSED_TIME_MICRO_SEC(start, stop) ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec))

int main() {
    char *filepath = "/home/hfingler/hf-HACK/src/kleio/lstm_page_539";
    int ret = kleio_load_model(filepath);
   
    struct timeval micro_start, micro_stop;
    long total_time = 0;
    int num_syscall = 0;
    uint32_t *syscalls = NULL;
    
    for (int i=26 ; i <= 26 ; i++) {
        num_syscall = i;
        syscalls = (int *)malloc(sizeof(int) * num_syscall);

        for (int j=0; j<num_syscall; j++) {
            syscalls[j] = rand() % MAX_SYSCALL_IDX;
        }

        gettimeofday(&micro_start, NULL);
        kleio_inference((void*)syscalls, num_syscall);
        gettimeofday(&micro_stop, NULL);
        total_time = ELAPSED_TIME_MICRO_SEC(micro_start, micro_stop);

        printf("%d: %ld\n",num_syscall, total_time);
        free(syscalls);
    }
    kleio_force_gc();

    return 0;
}

