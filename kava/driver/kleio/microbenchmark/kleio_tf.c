#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/stat.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include "shared_memory.h"

int kleio_load_model(const char *file);
int kleio_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window);

#ifdef __KERNEL__
#include <linux/time.h>
#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)
#endif

#define BUF_LEN 200
#define MODEL_LOAD_FAILURE -1

static char model_name[BUF_LEN] __initdata;
module_param_string(model_name, model_name, BUF_LEN, S_IRUGO);
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");

#define N_WARM 5
#define N_RUNS 5

int def_inputs[26] = {60, 500, 560, 60, 320, 620, 440, 180, 60, 620, 560, 240, 60, 360, 620, 380, 180, 120, 620, 620, 100, 60, 420, 620, 340, 140};

static int __init kleio_init(void) {
    int *inputs;
    unsigned int n_inputs;
    unsigned int sliding_window;
    int i, j, k;
    int result;
    int model_id;
    int rand_num;

    u64 t_start, t_stop, avg_total;
    u64* total_run_times;

    const char *modelpath = model_name;
    model_id = kleio_load_model(modelpath);

    if (model_id == MODEL_LOAD_FAILURE) {
        printk(KERN_ERR "Error loading LSTM model\n");
        return model_id;
    }
    sliding_window = 6;

    total_run_times = (u64*) kmalloc(N_RUNS*sizeof(u64), GFP_KERNEL);

    inputs = kmalloc(N_RUNS*sizeof(u64), GFP_KERNEL);
    for (i = 0; i < 26; i++) {
        inputs[i] = def_inputs[i];
    }

    kleio_inference((void*)inputs, 26, sliding_window);

    kfree(inputs);

    // for (i = 20; i <= 360; i += 40) {
    //     num_syscall = i;
    //     /* syscalls = (int *)kava_alloc((size_t)(sizeof(int) * num_syscall)); */
    //     syscalls = (int *)vmalloc((size_t)(sizeof(int) * num_syscall));

    //     // generate random syscall traces
    //     for (j=0; j<num_syscall; j++) {
    //         get_random_bytes(&rand_num, sizeof(rand_num));
    //         //make sure we are positive
    //         syscalls[j] = (rand_num >= 0 ? rand_num : -rand_num)% MAX_SYSCALL_IDX;
    //     }

    //     // warmup
    //     for (k = 0; k < N_WARM; k++) {
    //         standard_inference((void *)syscalls, num_syscall, 21);
    //         usleep_range(250, 1000);
    //     }

    //     for (k = 0; k < N_RUNS; k++) {
    //         // generate random syscall traces
    //         for (j=0; j<num_syscall; j++) {
    //             get_random_bytes(&rand_num, sizeof(rand_num));
    //             //make sure we are positive
    //             syscalls[j] = (rand_num >= 0 ? rand_num : -rand_num)% MAX_SYSCALL_IDX;
    //         }
    //         usleep_range(250, 1000);

    //         t_start = ktime_get_ns();
    //         standard_inference((void *)syscalls, num_syscall, sliding_window);
    //         t_stop = ktime_get_ns();

    //         total_run_times[k] = (t_stop - t_start);
    //         usleep_range(250, 1000);
    //     }

    //     avg_total = 0;
    //     for (k = 0; k < N_RUNS; k++) {
    //         avg_total += total_run_times[k];
    //     }
    //     avg_total = avg_total / (1000*N_RUNS);
    //     PRINT(V_INFO, "%d, %lld\n", i, avg_total);
    //     vfree(syscalls);
    // }

    // close_ctx();
    return 0;
}

static void __exit kleio_exit(void) {
}

module_init(kleio_init);
module_exit(kleio_exit);
