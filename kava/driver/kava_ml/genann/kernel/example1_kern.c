#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

#include <asm/fpu/api.h>

#include "genann_kava.h"

#define description_string "Kernel implementation of example 1 in genann."
#define maintainer_string "Bodun Hu"

MODULE_AUTHOR(maintainer_string);
MODULE_DESCRIPTION(description_string);
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");  /* required for kernel_fpu_begin */

#define TRAINING_ITERATIONS 300    // TODO: increase it to boost accuracy
#define INPUT_DIM_X 4
#define INPUT_DIM_Y 2
#define OUTPUT_DIM INPUT_DIM_X

#undef MEASURE_MICROBENCHMARKS

#define MEASURE_END2END_TIME

static int __init genann_example_1_init(void) {
    /* initialize neural network differently each run. */

    /* Input and expected out data for the XOR function. */
    // FPU in kernel might corrupt FPU state. However, doing so can be "safe"
    // on x86

#ifdef MEASURE_END2END_TIME
    struct timespec micro_end2end_start, micro_end2end_stop;
    long total_end2end_micro = 0;
#endif

#ifdef MEASURE_MICROBENCHMARKS
    struct timespec micro_init_start, micro_init_stop, micro_train_start, micro_train_stop,
                    micro_eval_start, micro_eval_stop, micro_free_start, micro_free_stop;
    long avg_micro_init = 0;
    long total_micro_free = 0;
    long total_micro_train = 0;
    long micro_nIters_train = TRAINING_ITERATIONS;
    long total_micro_eval = 0;
#endif

    const double input[INPUT_DIM_X][INPUT_DIM_Y] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double output[OUTPUT_DIM] = {0, 1, 1, 0};
    double *run_result_0;
    double *run_result_1;
    double *run_result_2;
    double *run_result_3;

    /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */
    genann *ann;
    int i;

    printk(KERN_INFO "GENANN example 1.\n");
    printk(KERN_INFO "Train a small ANN to the XOR function using backpropagation.\n");

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_start); 
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_init_start); 
#endif /* MEASURE_MICROBENCHMARKS */

    ann = genann_init(2, 1, 2, 1);

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_init_stop);
    avg_micro_init += ELAPSED_TIME_MICRO_SEC(micro_init_start, micro_init_stop);
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_train_start);
#endif

    for (i = 0; i < TRAINING_ITERATIONS; i++) {
        genann_train(ann, input[0], output + 0, 3);
        genann_train(ann, input[1], output + 1, 3);
        genann_train(ann, input[2], output + 2, 3);
        genann_train(ann, input[3], output + 3, 3);
    }

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_train_stop);
    total_micro_train += ELAPSED_TIME_MICRO_SEC(micro_train_start, micro_train_stop);
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_eval_start);
#endif

    run_result_0 = (double *)genann_run(ann, input[0]);
    run_result_1 = (double *)genann_run(ann, input[1]);
    run_result_2 = (double *)genann_run(ann, input[2]);
    run_result_3 = (double *)genann_run(ann, input[3]);

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_eval_stop);
    total_micro_eval += ELAPSED_TIME_MICRO_SEC(micro_eval_start, micro_eval_stop);
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_free_start);
#endif

    genann_free(ann);

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_free_stop);
    total_micro_free += ELAPSED_TIME_MICRO_SEC(micro_free_start, micro_free_stop);
#endif

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_stop);
    total_end2end_micro += ELAPSED_TIME_MICRO_SEC(micro_end2end_start, micro_end2end_stop);
#endif

    kernel_fpu_begin();
    printk(KERN_INFO "Output for [%d, %d] is %d%%.\n", (int)input[0][0], (int)input[0][1], (int)(100 * (*run_result_0)));
    printk(KERN_INFO "Output for [%d, %d] is %d%%.\n", (int)input[1][0], (int)input[1][1], (int)(100 * (*run_result_1)));
    printk(KERN_INFO "Output for [%d, %d] is %d%%.\n", (int)input[2][0], (int)input[2][1], (int)(100 * (*run_result_2)));
    printk(KERN_INFO "Output for [%d, %d] is %d%%.\n", (int)input[3][0], (int)input[3][1], (int)(100 * (*run_result_3)));
    kernel_fpu_end();

    vfree(run_result_0);
    vfree(run_result_1);
    vfree(run_result_2);
    vfree(run_result_3);

#ifdef MEASURE_MICROBENCHMARKS
    PRINT(V_INFO, "Average genann init time: %ld usec\n", avg_micro_init);
    PRINT(V_INFO, "Average genann train time: %ld usec\n", total_micro_train / micro_nIters_train);
    PRINT(V_INFO, "Average genann run time: %ld usec\n", total_micro_eval / INPUT_DIM_X);
    PRINT(V_INFO, "Average genann free time: %ld usec\n", total_micro_free);
#endif

#ifdef MEASURE_END2END_TIME
    PRINT(V_INFO, "Total execution time: %ld usec\n", total_end2end_micro);
#endif

    return 0;
}

static void __exit genann_example_1_exit(void) {
    printk(KERN_INFO "ANN is freed.\n");
}

module_init(genann_example_1_init);
module_exit(genann_example_1_exit);
