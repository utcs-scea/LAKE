#ifdef __KERNEL__
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/stat.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/time.h>
#include "shared_memory.h"
#else
//if uspace
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define vmalloc(X) malloc(X)
#define vfree(X) free((void*)X)
#define kava_free(X) free(X)
#define kava_alloc(X) malloc(X)
#define u64 uint64_t
#define u32 uint32_t
#include <unistd.h>
#define usleep_range(X,Y) sleep(X/1000)
#include <sys/time.h>
u64 get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}
#define ktime_get_ns() get_tsns()
#endif


int kleio_load_model(const char *file);
int kleio_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window);
void kleio_close_ctx(void);
void dogc(void);

#ifdef __KERNEL__
#define PRINT(...) do { if (1) printk(KERN_INFO __VA_ARGS__); } while (0)
#else
#define PRINT(...) do { if (1) printf(__VA_ARGS__); } while (0)
#endif

#define MODEL_LOAD_FAILURE -1

#ifdef __KERNEL__
static char *model_name = "mllb.cubin";
module_param(model_name, charp, 0444);
MODULE_PARM_DESC(model_name, "The path to mllb.cubin, default ./mllb.cubin");
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");
#else
static char *model_name = "/disk/hfingler/HACK/kava/driver/kleio/microbenchmark/lstm_page_539";
#endif

#define N_WARM 3
#define N_RUNS 3
//#define N_RUNS 500

int def_inputs[26] = {60, 500, 560, 60, 320, 620, 440, 180, 60, 620, 560, 240, 60, 360, 620, 380, 180, 120, 620, 620, 100, 60, 420, 620, 340, 140};

void main(void) {
    int *inputs;
    unsigned int n_inputs;
    int i, j, k;
    int result;
    int model_id;
    int rand_num;

    u64 t_start, t_stop, avg_total;
    u64* total_run_times;

    const char *modelpath = model_name;
    PRINT("loading model\n");
    model_id = kleio_load_model(modelpath);

    if (model_id == MODEL_LOAD_FAILURE) {
        PRINT("Error loading LSTM model\n");
        return;
    }
    total_run_times = (u64*) vmalloc(N_RUNS*sizeof(u64));

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
            PRINT("infer %d\n", i);
            kleio_inference((void*)inputs, 26, n_inputs+1);
            usleep_range(250, 1000);
        }

        for (k = 0; k < N_RUNS; k++) {
            t_start = ktime_get_ns();
            kleio_inference((void*)inputs, 26, n_inputs);
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
        PRINT("%d, %llu\n", i, avg_total);
    }
    kleio_close_ctx();
    
    vfree(total_run_times);
    vfree(inputs);
}

#ifdef __KERNEL__
static int __init kleio_init(void) {
    main();
    return 0;
}

static void __exit kleio_exit(void) {
}

module_init(kleio_init);
module_exit(kleio_exit);
#endif