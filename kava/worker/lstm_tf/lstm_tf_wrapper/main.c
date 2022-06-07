#include "c_wrapper.h"
#include <stdio.h>
#include <stdlib.h>

#define SYSCALL_MAX 230
#define MAX_SYSCALL_IDX 340
#define ITERATION 15

#define ELAPSED_TIME_MICRO_SEC(start, stop) ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec))

int main() {
    char *filepath = "/home/edwardhu/kava/worker/lstm_tf/lstm_tf_wrapper/";
    int ret = load_model(filepath);
    /* printf("load model ret value: %d\n", ret); */

    /* int *data = (int *)malloc(sizeof(int) * 40); */
    /* int i = 0; */
    /* for (; i < 40; i++) { */
    /*     data[i] = 8; */
    /* } */

    struct timeval micro_start, micro_stop;
    long total_time = 0;

    int num_syscall = 0;
    int *syscalls = NULL;
    // warmup
    for (int i=20;i<SYSCALL_MAX;i++) {
        num_syscall = i;
        syscalls = (int *)malloc((size_t)sizeof(int) * num_syscall);

        for (int j=0; j<num_syscall; j++) {
            syscalls[j] = rand() % MAX_SYSCALL_IDX;
        }

        int result = standard_inference((void *)syscalls, num_syscall, 1);
        gettimeofday(&micro_start, NULL);
        for (int k = 0; k < ITERATION; k ++) {
            result = standard_inference((void *)syscalls, num_syscall, 1);
        }
        gettimeofday(&micro_stop, NULL);
        total_time = ELAPSED_TIME_MICRO_SEC(micro_start, micro_stop);

        /* printf("result is %d\n", result); */
        printf("result is [kava-lstm-tf-gpu-user] %d %ld gg\n",num_syscall, total_time / ITERATION);


        free(syscalls);
    }

    /* const void *syscalls = (void *)data; */
    /* int result = standard_inference(syscalls, 40, 1); */
    /* printf("inference result is: %d\n", result); */

    close_ctx();
}
