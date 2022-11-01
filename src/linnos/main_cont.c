/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/vmalloc.h>
#include <asm/fpu/api.h>
#include "cuda.h"
#include "lake_shm.h"

#include "test_weights.h"
#include "helpers.h"
#include "predictors.h"
#include "variables.h"
#define FEAT_31
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2

//#define RUNTIME_MS  30000
#define STEP_MS 20
#define INTERVAL_US 200

static char *cubin_path = "linnos.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to linnos.cubin, default ./linnos.cubin");

static int runtime_s = 10;
module_param(runtime_s, int, 0444);
u64 RUNTIME_MS;

u8 model_size = 0;

long *test_weights[8] = { weight_0_T, weight_1_T, bias_0, bias_1, weight_M_1_T, bias_M_1, weight_M_2_T, bias_M_2};

u64 *out_ts;
u64 *out_predicted;
struct GPU_weights state;

static void run_cpu(char *input) {
    cpu_prediction_model_plus_2(input, 1, test_weights);
}

static void run_gpu(int batch_size) {
    copy_inputs_to_gpu(batch_size);
    gpu_predict_batch_plus_2(0, batch_size, state.weights);
    copy_results_from_gpu(batch_size);
}

static int run(void) {
    int proc;
    int i, j;
    int max_batch_size = 256;
    // n needs to be at least as large as the largest batch size
    int batch_size;
    char input[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
    u64 t_start, t_stop, step_start, elapsed;
    u64 count, tput;
    bool use_gpu;
    
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;

    RUNTIME_MS = runtime_s * 1000;
    u64 cur_slot = 0;
    u64 out_slots = (RUNTIME_MS/STEP_MS) * 10; //to be safe  
    out_ts = vmalloc(out_slots*sizeof(u64));
    out_predicted = vmalloc(out_slots*sizeof(u64));

    batch_size = 32;
    initialize_gpu(cubin_path, max_batch_size*4);
    copy_weights(test_weights, &state);
    expand_input_n_times(input, batch_size);

    pr_warn("-----start-----\n");
    t_start = ktime_get_ns();
    out_ts[cur_slot]   = t_start;
    out_predicted[cur_slot] = 0;
    cur_slot++;
    while (1) { //run for RUNTIME_MS
        nvmlRunningProcs(&proc);
        //pr_warn("%d: %s\n", proc, proc > 1 ? "cpu" : "GPU");
        use_gpu = proc <= 1;
        count = 0;
        
        step_start = ktime_get_ns();
        while(1) {  // run for STEP_MS
            if (use_gpu) {
                run_gpu(batch_size);
            } else {
                for (i = 0 ; i < batch_size ; i++)
                    run_cpu(input);
            }
            count += batch_size;
            t_stop = ktime_get_ns();
            elapsed = t_stop - step_start;
            if (elapsed >= (STEP_MS*1000000)) {
                break;
            }
            //wait a bit
            usleep_range(INTERVAL_US-20, INTERVAL_US+20);
        }
        //a batch of STEP_MS was completed
        //tput = (count*1000) / elapsed;
        //pr_warn("count %llu\n",count);
        out_ts[cur_slot]   = t_stop;
        out_predicted[cur_slot] = count;

        cur_slot++;
        t_stop = ktime_get_ns();
        elapsed = t_stop - t_start;

        if (elapsed >= (RUNTIME_MS*1000000))
            break;
    }

    for (i = 1 ; i < cur_slot ; i++) {
        pr_warn("lakecont,%llu, %llu\n", out_ts[i]-t_start, out_predicted[i]);
    }

    gpu_cuda_cleanup(&state);
    vfree(out_ts);
    vfree(out_predicted);
    cuCtxDestroy(cuctx);
    return 0;
}


/**
 * Program main
 */
static int __init linnos_init(void)
{
	return run();
}

static void __exit linnos_fini(void)
{

}

module_init(linnos_init);
module_exit(linnos_fini);

MODULE_AUTHOR("Henrique Fingler and Isha Tarte");
MODULE_DESCRIPTION("Kernel module of a linnos program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
