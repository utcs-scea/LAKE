#include "c_wrapper.h"
#include <stdio.h>

int main() {
    char *filepath = "/home/edwardhu/kava/worker/lstm_tf/lstm_tf_wrapper/";
    int ret = load_model(filepath);
    printf("load model ret value: %d\n", ret);

    int *data = (int *)malloc(sizeof(int) * 40);
    int i = 0;
    for (; i < 40; i++) {
        data[i] = 8;
    }

    const void *syscalls = (void *)data;
    int result = standard_inference(syscalls, 40, 1);
    printf("inference result is: %d\n", result);

    close_ctx();
}
