#ifdef __KERNEL__
#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/vmalloc.h>
#include <asm/fpu/api.h>
#include "cuda.h"
#include "lake_shm.h"
//uspace
#else
#define kava_free(X) free(X)
#define kava_alloc(X) malloc(X)
#define vfree(X) free(X)
#define vmalloc(X) malloc(X)
#include <stdint.h>
#define u64 uint64_t
#include <unistd.h>
#include <stdio.h>
#include <stdbool.h>
#define usleep_range(X,Y) sleep(X/1000)
#include <sys/time.h>
#include <sys/random.h>
u64 get_tsns() {
    struct timeval current_time;
    gettimeofday(&current_time, NULL);
    return current_time.tv_sec*1000000000 + current_time.tv_usec*1000;
}
#define ktime_get_ns() get_tsns()
#include <stdbool.h>
#endif

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

static char *cubin_path = "linnos.cubin";
#ifdef __KERNEL__
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to linnos.cubin, default ./linnos.cubin");
#endif

long *test_weights[4] = { weight_0_T, weight_1_T, bias_0, bias_1};

static int run_gpu(void) {
    int i, j;
    PRINT("Starting\n");
    int batch_sizes[] = {512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    int n_batches = 12;
    int max_batch_size = 1024;
    // n needs to be at least as large as the largest batch size
    const int n = 1024;
    bool res;
    u64 false_count=0, true_count=0;
    u64 result_mismatches = 0;
    int batch_size;
    char input[31] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,9,0,0,0,9,0,0,0,9};
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;
  
    struct GPU_weights state;

    initialize_gpu(cubin_path, max_batch_size);
    copy_weights(test_weights, &state);

    comp_run_times = (u64*) vmalloc(RUNS*sizeof(u64));
    total_run_times = (u64*) vmalloc(RUNS*sizeof(u64));

    // flatten_input(n, input);
    expand_input_n_times(input, n);
    // measuring GPU time
    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];

        // copy inputs to GPU each time we run
        copy_inputs_to_gpu(batch_size);

        //warmup
        gpu_predict_batch(0, batch_size, state.weights);
        copy_results_from_gpu(batch_size);
        
        cuCtxSynchronize();
        usleep_range(250, 1000);
    
        for (j = 0 ; j < RUNS ; j++) {
            PREDICT_GPU_SYNC = 0;
            t_start = ktime_get_ns();
            copy_inputs_to_gpu(batch_size);
            gpu_predict_batch(0, batch_size, state.weights);
            copy_results_from_gpu(batch_size);
            t_stop = ktime_get_ns();
            
            usleep_range(500, 2000);

            PREDICT_GPU_SYNC = 1;
            c_start = ktime_get_ns();
            gpu_predict_batch(0, batch_size, state.weights);
            c_stop = ktime_get_ns();
            
            usleep_range(500, 2000);
            comp_run_times[j] = (c_stop - c_start);
            total_run_times[j] = (t_stop - t_start);
	    }

	    avg = 0; avg_total = 0;
        best = 0; best_total = 0;
        for (j = 0 ; j < RUNS ; j++) {
            avg += comp_run_times[j];
            avg_total += total_run_times[j];
            if (best == 0 || comp_run_times[j] < best) best = comp_run_times[j];
            if (best_total == 0 || total_run_times[j] < best_total) best_total = total_run_times[j];
        }
        avg = avg / (1000*RUNS); avg_total = avg_total / (1000*RUNS);
        best = best / 1000; best_total = best_total / 1000;

        PRINT("GPU batch_%d, %lld, %lld, %lld, %lld\n", batch_size, avg, avg_total, best, best_total);
	}

    // measuring cpu time
    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];

        //warmup
        cpu_prediction_model(input, 1, test_weights);
        
        usleep_range(250, 1000);
    
        for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();
            for(int k = 0; k < batch_size; k++) {
                char input_copy[31];
                memcpy (input_copy, input, sizeof(input));
                cpu_prediction_model(input, 1, test_weights);
            }
            t_stop = ktime_get_ns();
            
            usleep_range(500, 2000);

            c_start = t_start;
            c_stop = t_stop;
            
            usleep_range(500, 2000);
            comp_run_times[j] = (c_stop - c_start);
            total_run_times[j] = comp_run_times[j];//(t_stop - t_start);
	    }

	    avg = 0; avg_total = 0;
        best = 0; best_total = 0;
        for (j = 0 ; j < RUNS ; j++) {
            avg += comp_run_times[j];
            avg_total += total_run_times[j];
            if (best == 0 || comp_run_times[j] < best) best = comp_run_times[j];
            if (best_total == 0 || total_run_times[j] < best_total) best_total = total_run_times[j];
        }
        avg = avg / (1000*RUNS); avg_total = avg_total / (1000*RUNS);
        best = best / 1000; best_total = best_total / 1000;

        PRINT("CPU batch_%d, %lld, %lld, %lld, %lld\n", batch_size, avg, avg_total, best, best_total);
	}

    if(check_correctness) {
        for(int k = 0; k < CORRECTNESS_CHECKS; k++) {
            //generate random input
            #ifdef __KERNEL__ 
                get_random_bytes(input, LEN_INPUT);
            #else
                getrandom(input, LEN_INPUT, 0);
            #endif

            int cpu_result = cpu_prediction_model(input, 1, test_weights);

            //the 1's here mean we only do 1 input, easy to adapt to n
            expand_input_n_times(input, 1);
            copy_inputs_to_gpu(1);
            gpu_predict_batch(0, 1, state.weights);
            copy_results_from_gpu(1);
            
            res = gpu_outputs[0]>=(gpu_outputs[32])? false: true;
            //PRINT("Test [%d]: (%d) %s\n", k, res, res==cpu_result ? "Ok" : "WRONG");
            if (res!=cpu_result) result_mismatches++;
            if (cpu_result) true_count++;
            else false_count++;
        }
        PRINT("CPU prediction summary: %llu trues, %llu falses %llu result_mismatches\n", true_count, false_count, result_mismatches);
    }

    gpu_cuda_cleanup(&state);
    vfree(comp_run_times);
    vfree(total_run_times);
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
