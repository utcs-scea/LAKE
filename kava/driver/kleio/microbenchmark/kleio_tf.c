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
void kleio_close_ctx(void);

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
//#define N_RUNS 500

int def_inputs[26] = {60, 500, 560, 60, 320, 620, 440, 180, 60, 620, 560, 240, 60, 360, 620, 380, 180, 120, 620, 620, 100, 60, 420, 620, 340, 140};

static int __init kleio_init(void) {
    int *inputs;
    unsigned int n_inputs;
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
    total_run_times = (u64*) kmalloc(N_RUNS*sizeof(u64), GFP_KERNEL);

    int max_input = 129;
    inputs = vmalloc(max_input*sizeof(u32));
    for (i = 0; i < max_input; i++) {
        inputs[i] = def_inputs[i];
    }

    for (i = 1; i <= 129; i += 8) {
    //for (i = 81; i <= 81; i += 5) {
        n_inputs = i;

        // warmup
        for (k = 0; k < N_WARM; k++) {
            kleio_inference((void*)inputs, n_inputs, n_inputs+1);
            usleep_range(250, 1000);
        }

        for (k = 0; k < N_RUNS; k++) {
            t_start = ktime_get_ns();
            kleio_inference((void*)inputs, n_inputs, n_inputs);
            t_stop = ktime_get_ns();

            total_run_times[k] = (t_stop - t_start);
            usleep_range(1000, 2000);
            dogc();
            usleep_range(1000, 2000);
        }

        avg_total = 0;
        for (k = 0; k < N_RUNS; k++) {
            avg_total += total_run_times[k];
        }
        avg_total = avg_total / (1000*N_RUNS);
        PRINT(V_INFO, "%d, %lld\n", i, avg_total);
    }

    kleio_close_ctx();
    
    vfree(inputs);
    return 0;
}

static void __exit kleio_exit(void) {
}

module_init(kleio_init);
module_exit(kleio_exit);
