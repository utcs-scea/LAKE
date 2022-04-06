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

static char *cubin_path = "mllb.cubin";
module_param(cubin_path, charp, 0444);
MODULE_PARM_DESC(cubin_path, "The path to mllb.cubin, default ./mllb.cubin");

static int run_cpu(void) {
    return 0;
}

static int run_gpu(void) {
    int i, j;
    //int batch_sizes[] = {64, 128, 256, 512};
    int batch_sizes[] = {512};

    //these are changeable
    const int n = 1024;
    int n_batches = 1;
    
    int batch_size;
    int rand_floats_as_int[] = {1036831949, 1045220557, 1050253722, -1110651699};
    struct timespec t_start, t_stop, c_start, c_stop;
    long total_time, computation_time;
    int rand_counter = 0;
    
    //int linear_inputs[NR_FEAT*n];
    int* linear_inputs;
    
    CUcontext cuContext;
    CUfunction batch_mllb_kernel;
    CUdeviceptr d_inputs, d_w1, d_b1, d_w2, d_results;

    linear_inputs = (int*) kmalloc(NR_FEAT*n*sizeof(float), GFP_KERNEL);

    //init cuda context
    gpu_init(0, &cuContext);

    //initialize a linear matrix with fake inputs
    for (j = 0 ; j < n ; j++) {
        for (i = 0; i < NR_FEAT; i++) {
            linear_inputs[j*NR_FEAT + i] = rand_floats_as_int[rand_counter];
            rand_counter++;
            if (rand_counter == 4) rand_counter = 0;
        }
    }

    gpu_get_cufunc(cubin_path, "_Z13mllb_infer_v2PfS_S_S_fS_", &batch_mllb_kernel);

    for (i = 0 ; i < n_batches ; i++) {
        batch_size = batch_sizes[i];
        total_time = 0;
        computation_time = 0;

        gpu_setup(batch_size, &d_inputs, &d_w1, &d_b1, &d_w2, &d_results);

        // //warmup
        // msleep(100);
        gpu_setup_inputs(d_inputs, linear_inputs, batch_size);
        gpu_inference_many(&batch_mllb_kernel, batch_size, d_inputs, d_w1, d_b1, d_w2, *b2, d_results);
        cuCtxSynchronize();
            
        //for each batch, measure
        for (j = 0 ; j < n/batch_size ; j++) {
            //PRINT(V_INFO, "Runing batch %d/%d for batch size %d\n", j+1, n/batch_size, batch_size);

            getnstimeofday(&t_start);
            //gpu_setup_inputs(d_inputs, linear_inputs+j*batch_size, batch_size);
            gpu_setup_inputs(d_inputs, linear_inputs, batch_size);

            //main computation
            getnstimeofday(&c_start);
            gpu_inference_many(&batch_mllb_kernel, batch_size, d_inputs, d_w1, d_b1, d_w2, *b2, d_results);
            getnstimeofday(&c_stop);

            gpu_get_result(batch_size);
            getnstimeofday(&t_stop);

            total_time       += (t_stop.tv_sec - t_start.tv_sec) * 1000000 + (t_stop.tv_nsec - t_start.tv_nsec) / 1000;
            computation_time += (c_stop.tv_sec - c_start.tv_sec) * 1000000 + (c_stop.tv_nsec - c_start.tv_nsec) / 1000;
        }
        //std::cout << "Batched GPU time for " << n << " inferences (batch size " << batch_size << "): " << gpubatch_total << "ns. Average per inference:" << gpubatch_total/n << "ns." << std::endl;
        //std::cout << "Including data transfers: " << gpubatch_all_total << "ns. Average per inference:" << gpubatch_all_total/n << "ns." << std::endl;
        //csv << "GPU batch" << batch_size << "," << gpubatch_total << "," << gpubatch_total/n << "," << gpubatch_all_total << "," << gpubatch_all_total/n << "," << std::endl;
        
        PRINT(V_INFO, "GPU batch_%d, %ld, %ld\n", batch_size, computation_time, total_time);

        gpu_clean(d_inputs, d_w1, d_b1, d_w2, d_results);
    }

    kfree(linear_inputs);
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
