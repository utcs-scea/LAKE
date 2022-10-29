#include "py_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define SYSCALL_MAX 230
#define MAX_SYSCALL_IDX 340
#define NTEST 1
#define ELAPSED_TIME_MICRO_SEC(start, stop) ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec))

char outs[2][200][1024];

int main() {
    struct timeval micro_start, micro_stop;
    long total_time = 0;
    int num_syscall = 0;
    uint32_t *syscalls = NULL;
    int count = 0;
    //for (int i=26 ; i <= 26 ; i++) {
    //for (int i=1, j=0 ; i <= 129 ; i+=8, j++) {
    for (int dev=0 ; dev < 2 ; dev++) {
        printf("wat\n");
        int ret = kleio_load_model(__MODELPATH__);
        for (int i=20, j=0 ; i <= 1200 ; i+=60, j++) {
        //for (int i=1200, j=0 ; i <= 1201 ; i+=60, j++) {
            num_syscall = i;
            syscalls = (int *)malloc(sizeof(int) * num_syscall);
            int sum = 0;
            for (int j=0; j<num_syscall; j++) {
                syscalls[j] = rand() % MAX_SYSCALL_IDX;
            }

            gettimeofday(&micro_start, NULL);
            kleio_inference((void*)syscalls, num_syscall, dev);
            gettimeofday(&micro_stop, NULL);

            for (int k=0 ; k <NTEST ; k++) {
                gettimeofday(&micro_start, NULL);
                kleio_inference((void*)syscalls, num_syscall, dev);
                gettimeofday(&micro_stop, NULL);
                total_time = (ELAPSED_TIME_MICRO_SEC(micro_start, micro_stop))/1000;
                sum += total_time;

                usleep(500);
            }

            printf("Time of %d[%d]: %ldms\n",num_syscall, dev, sum/NTEST);
            sprintf(outs[dev][j], "%d,%ld\n", num_syscall, sum/NTEST);
            count++;
            free(syscalls);
        }
    }

    for (int dev=0 ; dev < 2 ; dev++)
        for (int i = 0 ; i < count ; i++) 
            printf("%s", outs[dev][i]);

    return 0;
}

