
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/stat.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/vmalloc.h>
#include <linux/time.h>
#include "cuda.h"
#include "lake_shm.h"

MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");

#define N_WARM 1
#define N_RUNS 1

int def_inputs[26] = {60, 500, 560, 60, 320, 620, 440, 180, 60, 620, 560, 240, 60, 360, 620, 380, 180, 120, 620, 620, 100, 60, 420, 620, 340, 140};

void main(void) {
    int *inputs;
    unsigned int n_inputs;
    int dev, i, j, k;

    u64 t_start, t_stop;
    u64 *cpu_times, *gpu_times;
    u64 cpu_avg, gpu_avg;
    int max_input = 1200/20;

    cpu_times = (u64*) vmalloc(N_RUNS*sizeof(u64));
    gpu_times = (u64*) vmalloc(N_RUNS*sizeof(u64));
    
    inputs = kava_alloc(max_input*sizeof(u32));
    for (i = 0; i < max_input; i++) {
        inputs[i] = def_inputs[i%26];
    }

    kleioLoadModel(0, 0);

    //for (n_inputs=20 ; n_inputs <= 1200 ; n_inputs+=60) {
    for (n_inputs=20 ; n_inputs <= 60 ; n_inputs+=60) {
        for (dev=0 ; dev < 2 ; dev++){
        //for (n_inputs=20, j=0 ; n_inputs <= 30 ; n_inputs+=60, j++) {
            // warmup
            for (k = 0; k < N_WARM; k++) {
                kleioInference((void*)inputs, 600, dev);
                usleep_range(200, 300);
            }

            for (k = 0; k < N_RUNS; k++) {
                t_start = ktime_get_ns();
                kleioInference((void*)inputs, n_inputs, dev);
                t_stop = ktime_get_ns();

                pr_warn("time %llu\n", (t_stop - t_start)/1000);
                if (dev == 0)
                    cpu_times[k] = (t_stop - t_start);
                else
                    gpu_times[k] = (t_stop - t_start);

                usleep_range(500, 600);
            }
        }
        
        //flush results for this size
        cpu_avg = 0; gpu_avg = 0;
        for (k = 0; k < N_RUNS; k++) {
            cpu_avg += cpu_times[k];
            gpu_avg += gpu_times[k];
        }

        cpu_avg = cpu_avg / (1000*N_RUNS); //us to ms
        gpu_avg = gpu_avg / (1000*N_RUNS); //us to ms
        pr_warn("kleio_%d,%llu,%llu\n", n_inputs, cpu_avg, gpu_avg);
    }

    vfree(gpu_times);
    vfree(cpu_times);
    kava_free(inputs);
}


static int __init kleio_init(void) {
    main();
    return 0;
}

static void __exit kleio_exit(void) {
}

module_init(kleio_init);
module_exit(kleio_exit);
