#include "helpers.h"
#include "consts.h"
#include <linux/delay.h>
#include <linux/ktime.h>
//#include <asm/fpu/api.h>


//kernel_fpu_begin()
//kernel_fpu_end()

// struct timespec t_start, t_stop
// long total_time;
// getnstimeofday(&micro_proc_stop);
// total_time = (micro_proc_stop.tv_sec - micro_proc_start.tv_sec) * 1000000 + (micro_proc_stop.tv_nsec - micro_proc_start.tv_nsec) / 1000;

// https://stackoverflow.com/questions/69748923/how-to-measure-the-execution-time-of-a-function-in-linux-kernel-module

static char *cubin_path = "mllb.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to mllb.cubin, default ./mllb.cubin");

static int run_cpu(void) {
    return 0;
}

static int run_gpu(void) {
    int i, j;
    //these are changeable
    //int batch_sizes[] = {512};
    int batch_sizes[] = {1,2,4,8,16,32,64, 128, 256, 512,1024};
    int n_batches = 11;
    const int max_batch = 1024;
    
    //int WARMUP_RUNS = 2;
    int RUNS = 10;

    int batch_size;
    int rand_floats_as_int[] = {1036831949, 1045220557, 1050253722, -1110651699};
    int rand_counter = 0;
    int* linear_inputs;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;

    CUcontext cuContext;
    CUfunction batch_mllb_kernel;
    CUdeviceptr d_inputs, d_w1, d_b1, d_w2, d_results;

    //linear_inputs = (int*) kmalloc(NR_FEAT*n*sizeof(float), GFP_KERNEL);
    float* linear_inputs = kava_alloc(NR_FEAT*n);
    check_malloc(linear_inputs, "check_malloc", __LINE__);

    //init cuda context
    gpu_init(0, &cuContext);

    //initialize a linear matrix with fake inputs
    for (j = 0 ; j < max_batch ; j++) {
        for (i = 0; i < NR_FEAT; i++) {
            linear_inputs[j*NR_FEAT + i] = rand_floats_as_int[rand_counter];
            rand_counter++;
            if (rand_counter == 4) rand_counter = 0;
        }
    }

    gpu_get_cufunc(cubin_path, "_Z13mllb_infer_v2PfS_S_S_fS_", &batch_mllb_kernel);
    comp_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
    total_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);

    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];
        gpu_setup(batch_size, &d_inputs, &d_w1, &d_b1, &d_w2, &d_results);

        //warmup
        gpu_setup_inputs(d_inputs, linear_inputs, batch_size);
        gpu_inference_many(&batch_mllb_kernel, batch_size, d_inputs, d_w1, d_b1, d_w2, *b2, d_results);
        usleep_range(1000, 2000);
        cuCtxSynchronize();

        for (j = 0 ; j < RUNS ; j++) {
            //PRINT(V_INFO, "Runing batch %d/%d for batch size %d\n", j+1, n/batch_size, batch_size);
            t_start = ktime_get_ns();
            gpu_setup_inputs(d_inputs, linear_inputs, batch_size);
            c_start = ktime_get_ns();
            gpu_inference_many(&batch_mllb_kernel, batch_size, d_inputs, d_w1, d_b1, d_w2, *b2, d_results);
            c_stop = ktime_get_ns();
            gpu_get_result(batch_size);
            t_stop = ktime_get_ns();

            comp_run_times[j] = (c_stop - c_start);
            total_run_times[j] = (t_stop - t_start);
            usleep_range(250, 1000);
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

        PRINT(V_INFO, "GPU batch_%d, %lld, %lld, %lld, %lld\n", batch_size, avg, avg_total, best, best_total);
        gpu_clean(d_inputs, d_w1, d_b1, d_w2, d_results);
    }

    kfree(linear_inputs);
    kfree(comp_run_times);
    kfree(total_run_times);
    return 0;
}


/**
 * Program main
 */
static int __init mllb_init(void)
{
	return run_gpu();
}

static void __exit mllb_fini(void)
{
    //cleanup
}

module_init(mllb_init);
module_exit(mllb_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Kernel module of a mllb program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
