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

#define RUNS 3
bool check_correctness = true; 
#define CORRECTNESS_CHECKS 1000

u8 model_size = 0;

static char *cubin_path = "linnos.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to linnos.cubin, default ./linnos.cubin");

long *test_weights[8] = { weight_0_T, weight_1_T, bias_0, bias_1, weight_M_1_T, bias_M_1, weight_M_2_T, bias_M_2};

char *gpu_patterns[3] = {
    "linnos+0_GPU_batch_", "linnos+1_GPU_batch_", "linnos+2_GPU_batch_"
};

char *cpu_patterns[3]= {
    "linnos+0_CPU_batch_", "linnos+1_CPU_batch_", "linnos+2_CPU_batch_"
};
char out[1024];

static int run_gpu(void) {
    int i, j, k;
    struct GPU_weights state;
    int this_dev = 0;
    int my_batch = 0;
    int test_size = 256;
    bool *results;
    int ndev = 3;
    results = (bool*) vmalloc(test_size*sizeof(bool));

    //initialize_gpu(cubin_path, max_batch_size);
    multi_initialize_gpu(cubin_path, 512, ndev); 

    copy_weights(test_weights, &state);
    usleep_range(250, 1000);

    char *inputs = kava_alloc(test_size * 31);

    for (k = 0 ; k < 512 ; k++) {
        get_random_bytes(inputs, test_size * 31);

        for (i = 0 ; i < test_size ; i++) {
            for (j = 0 ; j < LEN_INPUT ; j++)
                multi_inputs_to_gpu[this_dev][my_batch][i*31+j] = (long) inputs[i*31+j];
        }
        
        usleep_range(250, 1000);
        int n_vecs = test_size;
        int dev = this_dev;
        int batch_id = my_batch;
        multi_copy_inputs_to_gpu(n_vecs, dev, batch_id);
        multi_gpu_predict_batch_plus_1(0, n_vecs, state.weights, dev, batch_id);
        multi_copy_results_from_gpu(n_vecs, dev, batch_id);

        for (i = 0 ; i < test_size ; i++)
            results[i] = gpu_get_prediction(this_dev, my_batch, i);

        int cpu_result;
        for (i = 0 ; i < test_size ; i++) {
            cpu_result = cpu_prediction_model_plus_1(&inputs[i*31], 1, test_weights);
            if (cpu_result != results[i]) {
                pr_warn("Wrong result at idx %d\n", i);
            } 
        }
    }

    kava_free(inputs);
    vfree(results);

    // if(check_correctness) {
    //     char *input_64 = kava_alloc(64 * LEN_INPUT * sizeof(char));
    //     for(int k = 0; k < CORRECTNESS_CHECKS; k++) {
    //         //generate random input
    //         #ifdef __KERNEL__ 
    //             get_random_bytes(input_64, 64 * LEN_INPUT);
    //         #else
    //             getrandom(input_64, 64 * LEN_INPUT, 0);
    //         #endif

    //         //the 1's here mean we only do 1 input, easy to adapt to n
    //         copy_input_to_shm(input_64, 64);
    //         copy_inputs_to_gpu(64);
    //         gpu_predict_batch(0, 64, state.weights);
    //         copy_results_from_gpu(64);
            
    //         for(int bnum = 0; bnum < 64; bnum++) {
    //             int cpu_result = cpu_prediction_model(input_64 + LEN_INPUT * bnum * sizeof(char), 1, test_weights);
    //             res = gpu_outputs[bnum*64]>=(gpu_outputs[bnum * 64 + 32])? false: true;
    //             //PRINT("Test [%d]: (%d) %s\n", bnum, res, res==cpu_result ? "Ok" : "WRONG");
    //             if (res!=cpu_result) result_mismatches++;
    //             if (cpu_result) true_count++;
    //             else false_count++;
    //         }            
    //     }
    //     PRINT("CPU prediction summary: %llu trues, %llu falses %llu result_mismatches\n", true_count, false_count, result_mismatches);
    // }

    // gpu_cuda_cleanup(&state);
    // vfree(comp_run_times);
    // vfree(total_run_times);

    cuCtxDestroy(cuctx);
    return 0;
}

#ifdef __KERNEL__

/**
 * Program main
 */
static int __init linnos_init(void)
{
	return run_gpu();
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

#else

int main() {
    run_gpu();
    return 0;
}

#endif
