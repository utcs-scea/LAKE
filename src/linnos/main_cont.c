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

#define RUNTIME_MS  2000
#define STEP_MS 250
#define INTERVAL_US 50

static char *cubin_path = "linnos.cubin";
#ifdef __KERNEL__
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to linnos.cubin, default ./linnos.cubin");
#endif

u8 model_size = 0;

long *test_weights[8] = { weight_0_T, weight_1_T, bias_0, bias_1, weight_M_1_T, bias_M_1, weight_M_2_T, bias_M_2};

u64 *out_ts;
u64 *out_tput;
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
    struct GPU_weights state;
    u64 t_start, t_stop, step_start, elapsed;
    u64 count, tput;
    //i64 sleepy_time;
    bool use_gpu;
    
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;

    u64 cur_slot = 0;
    u64 out_slots = (RUNTIME_MS/STEP_MS) * 4; //4 to be safe  
    out_ts = vmalloc(out_slots*sizeof(u64));
    out_tput = vmalloc(out_slots*sizeof(u64));

    batch_size = 32;
    initialize_gpu(cubin_path, max_batch_size);
    copy_weights(test_weights, &state);
    expand_input_n_times(input, batch_size);

    while (1) {
        nvmlRunningProcs(&proc);
        use_gpu = proc == 1;

        count = 0;
        t_start = ktime_get_ns();
        
        while (1) {
            step_start = ktime_get_ns();
            while(1) {
                if (use_gpu) {
                    pr_warn("using gpu\n");
                    run_gpu(batch_size);
                } else {
                    pr_warn("using cpu\n");
                    for (i = 0 ; i < batch_size ; i++)
                        run_cpu(input);
                }
                pr_warn("done\n");
                count += batch_size;
                t_stop = ktime_get_ns();
                elapsed = t_stop - step_start;
                if (elapsed >= (STEP_MS*1000000)) {
                    break;
                }
                //wait a bit
                usleep_range(INTERVAL_US-20, INTERVAL_US+20);
                break; //XXX
            }   
            pr_warn("1\n");
            tput = count / elapsed;
            out_ts[cur_slot] = t_stop;
            out_tput = tput;
            cur_slot++;
            pr_warn("2\n");
            t_stop = ktime_get_ns();
            elapsed = t_stop - t_start;

            break; //XXX
            if (elapsed >= (RUNTIME_MS*1000000)) {
                break;
            }
        }
    }
    pr_warn("3\n");
    gpu_cuda_cleanup(&state);
    vfree(out_ts);
    vfree(out_tput);
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
