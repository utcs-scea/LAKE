#include "helpers.h"
#include "consts.h"
#include <linux/delay.h>
#include <linux/ktime.h>
#include <asm/fpu/api.h>

//kernel_fpu_begin()
//kernel_fpu_end()

// https://stackoverflow.com/questions/69748923/how-to-measure-the-execution-time-of-a-function-in-linux-kernel-module

static char *cubin_path = "mllb.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to mllb.cubin, default ./mllb.cubin");

int use_kshm = 1;
module_param(use_kshm, int, 0);
MODULE_PARM_DESC(use_kshm, "Set to 1 (default) to use zero copy");

static inline void check_malloc(void *p, const char* error_str, int line) {
	if (p == NULL) printk(KERN_ERR "ERROR: Failed to allocate %s (line %d)\n", error_str, line);
}

struct matrix {
    int nrow;
    int ncol;
    float *values;
};
#define m2d(x, i, j) (x)->values[i * (x)->ncol + j]
#define m1d(x, i) (x)->values[i]
#define _ReLU(x) (x > 0 ?  x : 0)
__attribute__((target("sse")))
int matmul(struct matrix *X, struct matrix *Y, struct matrix *Z) 
{
    int i, j, k;
    for(i = 0; i < X->nrow; i++)
        for(j = 0; j < Y->ncol; j++)
            for(k = 0; k < X->ncol; k++) {
                m2d(Z, i, j) = m2d(Z, i, j) + (m2d(X, i, k) * m2d(Y, k, j));
            }
    return 0;
}
__attribute__((target("sse")))
void matadd(struct matrix *X, struct matrix *Y, struct matrix *Z)
{
    int i;
    for (i = 0; i < X->nrow * X->ncol; i++) {
        Z->values[i] = X->values[i] + Y->values[i];
    }
}
__attribute__((target("sse")))
void ReLU(struct matrix *X)
{
    int i;
    for (i = 0; i < X->nrow * X->ncol; i++) {
        X->values[i] = _ReLU(X->values[i]);
    }
}
int forward_pass(struct matrix *input){
    float output;
    int ret;
    kernel_fpu_begin();
    float o1[10] = {0};
    float o2[10] = {0};

    struct matrix W1 = {NR_FEAT, 10, w1};
    struct matrix out1 = {1, 10, o1};
    struct matrix B1 = {1, 10, b1};
    struct matrix W2 = {10, 1, w2};
    struct matrix out2 = {1, 1, o2};
    struct matrix B2 = {1, 1, b2};

    matmul(input, &W1, &out1);
    matadd(&out1, &B1, &out1);
    ReLU(&out1);
    matmul(&out1, &W2, &out2);
    matadd(&out2, &B2, &out2);
    output = m1d(&out2, 0);
    /* printf("output: %f\n", output); */
    
    ret = output > 0.5 ? 1 : 0;
    kernel_fpu_end();
    return ret;
}

static int run_cpu(int* batch_sizes, int n_batches, int max_batch, int RUNS, int* rand_floats_as_int) {
    int i, j, k;
    int *tmp;
    int batch_size;
    int rand_counter = 0;
    u64 t_start, t_stop;
    u64* total_run_times;
    u64 avg, best;

    struct matrix *inputs = (struct matrix*) kmalloc(max_batch*sizeof(struct matrix), GFP_KERNEL);
    for (j = 0 ; j < max_batch ; j++) {
        inputs[j].values = (float*) kmalloc(NR_FEAT*sizeof(float), GFP_KERNEL);
        inputs[j].nrow = 1;
        inputs[j].ncol = NR_FEAT;
        for (i = 0 ; i < NR_FEAT ; i++) {
            tmp = (int*) inputs[j].values+i;
            *tmp = rand_floats_as_int[rand_counter];
            rand_counter++;
            if (rand_counter == 4) rand_counter = 0;
        }
    }

    total_run_times = (u64*) kmalloc(RUNS*sizeof(u64), GFP_KERNEL);
    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];

        //warmup
        forward_pass(inputs);
        usleep_range(250, 1000);

        for (j = 0 ; j < RUNS ; j++) {
            t_start = ktime_get_ns();
            for (k = 0 ; k < batch_size ; k++) {
                forward_pass(inputs+k);
            }
            t_stop = ktime_get_ns();
            total_run_times[j] = (t_stop - t_start);
            usleep_range(250, 1000);
        }

        avg = 0;
        for (j = 0 ; j < RUNS ; j++) {
            avg += total_run_times[j];
        }
        avg = avg / (1000*RUNS); 
        PRINT(V_INFO, "cpu %d, %lld\n", batch_size, avg);
    }

    for (j = 0 ; j < max_batch ; j++) {
        kfree(inputs[j].values);
    }
    kfree(inputs);
    return 0;
}

static int run_gpu(int* batch_sizes, int n_batches, int max_batch, int RUNS, int* rand_floats_as_int) {
    int i, j;
    int* linear_inputs;
    int batch_size;
    int rand_counter = 0;
    u64 t_start, t_stop, c_start, c_stop;
    u64* comp_run_times;
    u64* total_run_times;
    u64 avg, avg_total;
    u64 best, best_total;

    CUcontext cuContext;
    CUfunction batch_mllb_kernel;
    CUdeviceptr d_inputs, d_w1, d_b1, d_w2, d_results;

    if (!use_kshm)
        linear_inputs = (int*) kmalloc(NR_FEAT*max_batch*sizeof(float), GFP_KERNEL);
    else
        linear_inputs = kava_alloc(NR_FEAT*max_batch);
    check_malloc(linear_inputs, "check_malloc", __LINE__);

    //init cuda context
    gpu_init(0, &cuContext);

    //initialize a linear matrix with fake inputs
    for (j = 0 ; j < max_batch*NR_FEAT ; j++) {
        linear_inputs[j] = rand_floats_as_int[rand_counter];
        rand_counter++;
        if (rand_counter == 4) rand_counter = 0;
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
        cuCtxSynchronize();
        gpu_inference_many(&batch_mllb_kernel, batch_size, d_inputs, d_w1, d_b1, d_w2, *b2, d_results);
        cuCtxSynchronize();
        usleep_range(250, 1000);

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
        avg = avg / (1000*RUNS); 
        avg_total = avg_total / (1000*RUNS);
        best = best / 1000; 
        best_total = best_total / 1000;

        //PRINT(V_INFO, "GPU batch_%d, %lld, %lld, %lld, %lld\n", batch_size, avg, avg_total, best, best_total);
        PRINT(V_INFO, "%d, %lld, %lld\n", batch_size, avg, avg_total);
        gpu_clean(d_inputs, d_w1, d_b1, d_w2, d_results);
    }

    if (!use_kshm) 
        kfree(linear_inputs);
    else
        kava_free(linear_inputs);

    kfree(comp_run_times);
    kfree(total_run_times);
    return 0;
}

static int run(void) {
    //these are configurable
    //int batch_sizes[] = {512};
    //int n_batches = 1;
    int batch_sizes[] = {1,2,4,8,16,32,64, 128, 256, 512,1024};
    int n_batches = 11;
    const int max_batch = 1024;
    int RUNS = 2;
    int rand_floats_as_int[] = {1036831949, 1045220557, 1050253722, -1110651699};

    run_cpu(batch_sizes, n_batches, max_batch, RUNS, rand_floats_as_int);
    run_gpu(batch_sizes, n_batches, max_batch, RUNS, rand_floats_as_int);

    return 0;
}


/**
 * Program main
 */
static int __init mllb_init(void)
{
	return run();
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
