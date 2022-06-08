#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/stat.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include "shared_memory.h"

int load_model(const char *file);
void close_ctx(void);
int standard_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window);

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

#define BUF_LEN 200

//#define SYSCALL_MAX 360
#define SYSCALL_MAX 21
#define MAX_SYSCALL_IDX 340
#define ITERATION 15

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

#define N_WARM 1
#define N_RUNS 2

static int __init lstm_init(void) {
    int *syscalls;
    unsigned int num_syscall;
    unsigned int sliding_window;
    int i, j, k;
    int result;
    int model_id;
    int rand_num;

    u64 t_start, t_stop, avg_total;
    u64* total_run_times;

    // 1. load LSTM model based on given file path
    const char *modelpath = model_name;
    model_id = load_model(modelpath);

    if (model_id == MODEL_LOAD_FAILURE) {
        printk(KERN_ERR "Error loading LSTM model\n");
        return model_id;
    }
    /* printk(KERN_INFO "Return code from load_model is %d\n", model_id); */
    sliding_window = 1;

    /* syscalls = (int *)kava_alloc((size_t)(sizeof(int) * 20)); */
    /* kava_free(syscalls); */

    total_run_times = (u64*) kmalloc(N_RUNS*sizeof(u64), GFP_KERNEL);

    for (i = 20; i <= 360; i += 40) {
    //for (i = 20; i <= 40; i += 40) {
        // 2. perform inference
        num_syscall = i;
        /* syscalls = (int *)kava_alloc((size_t)(sizeof(int) * num_syscall)); */
        syscalls = (int *)vmalloc((size_t)(sizeof(int) * num_syscall));

        // generate random syscall traces
        for (j=0; j<num_syscall; j++) {
            get_random_bytes(&rand_num, sizeof(rand_num));
            //make sure we are positive
            syscalls[j] = (rand_num >= 0 ? rand_num : -rand_num)% MAX_SYSCALL_IDX;
        }

        // warmup
        for (k = 0; k < N_WARM; k++) {
            standard_inference((void *)syscalls, num_syscall, sliding_window);
            usleep_range(250, 1000);
        }

        for (k = 0; k < N_RUNS; k++) {
            // generate random syscall traces
            for (j=0; j<num_syscall; j++) {
                get_random_bytes(&rand_num, sizeof(rand_num));
                //make sure we are positive
                syscalls[j] = (rand_num >= 0 ? rand_num : -rand_num)% MAX_SYSCALL_IDX;
            }

            t_start = ktime_get_ns();
            standard_inference((void *)syscalls, num_syscall, sliding_window);
            t_stop = ktime_get_ns();

            total_run_times[k] = (t_stop - t_start);
            usleep_range(250, 1000);
        }

        avg_total = 0;
        for (k = 0; k < N_RUNS; k++) {
            avg_total += total_run_times[k];
        }
        avg_total = avg_total / (1000*N_RUNS);
        PRINT(V_INFO, "%d, %lld\n", i, avg_total);
    }

    vfree(syscalls);
    /* kava_free(syscalls); */
    
    close_ctx();
    return 0;
}

static void __exit lstm_exit(void) {
    /* printk(KERN_INFO "Finished LSTM inference, exiting...\n"); */
}

module_init(lstm_init);
module_exit(lstm_exit);
