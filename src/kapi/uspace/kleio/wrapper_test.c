#include "py_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define SYSCALL_MAX 230
#define MAX_SYSCALL_IDX 340
#define ITERATION 15
#define NTEST 15
#define ELAPSED_TIME_MICRO_SEC(start, stop) ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec))

char outs[18][1024];

int main() {
    //char filepath[1024];
    //printf("file: %s\n", __FILE__);
    //sprintf(filepath, "%s/%s", __FILE__, "../../../kleio/lstm_page_539");
    char *filepath = "../../../kleio/lstm_page_539";
    int ret = kleio_load_model(filepath);
   
    struct timeval micro_start, micro_stop;
    long total_time = 0;
    int num_syscall = 0;
    uint32_t *syscalls = NULL;
    
    //for (int i=26 ; i <= 26 ; i++) {
    for (int i=1, j=0 ; i <= 129 ; i+=8, j++) {
        num_syscall = i;
        syscalls = (int *)malloc(sizeof(int) * num_syscall);
        int sum = 0;
        for (int j=0; j<num_syscall; j++) {
            syscalls[j] = rand() % MAX_SYSCALL_IDX;
        }

        gettimeofday(&micro_start, NULL);
        kleio_inference((void*)syscalls, num_syscall);
        gettimeofday(&micro_stop, NULL);

        for (int k=0 ; k <NTEST ; k++) {
            gettimeofday(&micro_start, NULL);
            kleio_inference((void*)syscalls, num_syscall);
            gettimeofday(&micro_stop, NULL);
            total_time = (ELAPSED_TIME_MICRO_SEC(micro_start, micro_stop))/1000;
            sum += total_time;

            usleep(500);
        }

        printf("Time of %d: %ldms\n",num_syscall, sum/NTEST);
        sprintf(outs[j], "%d,%ld\n", num_syscall, sum/NTEST);
        
        free(syscalls);
    }
    kleio_force_gc();

    for (int i = 0 ; i < 18 ; i++) 
        printf("%s", outs[i]);

    return 0;
}

