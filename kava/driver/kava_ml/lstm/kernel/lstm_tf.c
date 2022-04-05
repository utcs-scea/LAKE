#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/stat.h>
#include <linux/random.h>

#include "lstm_tf_nw_types.h"
#include "shared_memory.h"

#ifdef __KERNEL__
#include <linux/time.h>
#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)

#define ELAPSED_TIME_MICRO_SEC(start, stop) ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000)
#endif

#define BUF_LEN 1024

#define MAX_SYSCALL_IDX 340
#define SYSCALL_MAX 300
#define ITERATION 10

#define MEASURE_INFERENCE_TIME

#define MODEL_LOAD_FAILURE -1

static char model_name[BUF_LEN] __initdata;
module_param_string(model_name, model_name, BUF_LEN, S_IRUGO);

#define description_string "Kernel implementation of loading a LSTM model."
#define maintainer_string "Bodun Hu"

MODULE_AUTHOR(maintainer_string);
MODULE_DESCRIPTION(description_string);
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");

static int __init lstm_init(void) {
    int *syscalls;
    unsigned int num_syscall;
    unsigned int sliding_window;
    int i;
    int j;
    int k;
    int result;
    int model_id;
    int rand_num;
#ifdef MEASURE_INFERENCE_TIME
    struct timespec micro_inference_start, micro_inference_stop;
    long total_inference_micro = 0;
#endif
    
    // 1. load LSTM model based on given file path
    const char *modelpath = model_name;
    model_id = load_model(modelpath);
    if (model_id == MODEL_LOAD_FAILURE) {
        printk(KERN_ERR "Error loading LSTM model\n");
        return model_id;
    }
    /* printk(KERN_INFO "Return code from load_model is %d\n", model_id); */



    for (i = 20; i < SYSCALL_MAX; i++) {
        // 2. perform inference
        num_syscall = i;
        /* syscalls = (int *)kava_alloc((size_t)(sizeof(int) * num_syscall)); */
        syscalls = (int *)vmalloc((size_t)(sizeof(int) * num_syscall));
        sliding_window = 1;

        // generate random syscall traces
        for (j=0; j<num_syscall; j++) {
            get_random_bytes(&rand_num, sizeof(rand_num));
            syscalls[j] = rand_num % MAX_SYSCALL_IDX;
        }

#ifdef MEASURE_INFERENCE_TIME
        getnstimeofday(&micro_inference_start);
#endif
        for (k=0; k<ITERATION;k++) {
            result = standard_inference((void *)syscalls, num_syscall, sliding_window);
        }
#ifdef MEASURE_INFERENCE_TIME
        getnstimeofday(&micro_inference_stop); 
        total_inference_micro += ELAPSED_TIME_MICRO_SEC(micro_inference_start, micro_inference_stop);
#endif
        /* printk(KERN_INFO "Inference result is: %d\n", result); */
#ifdef MEASURE_INFERENCE_TIME
        printk(KERN_INFO "[kava-lstm-tf-gpu] %d %ld\n", num_syscall, total_inference_micro/ITERATION);
#endif

        vfree(syscalls);
        total_inference_micro = 0;
    }




    close_ctx();
    return 0;
}

static void __exit lstm_exit(void) {
    /* printk(KERN_INFO "Finished LSTM inference, exiting...\n"); */
}

module_init(lstm_init);
module_exit(lstm_exit);
