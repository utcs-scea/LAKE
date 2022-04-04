#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched/signal.h>

#include <asm/fpu/api.h>

#include "genann_kava.h"

#define description_string "Kernel implementation of example 2 in genann."
#define maintainer_string "Bodun Hu"

MODULE_AUTHOR(maintainer_string);
MODULE_DESCRIPTION(description_string);
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");  /* required for kernel_fpu_begin */

#undef MEASURE_MICROBENCHMARKS
#define MEASURE_END2END_TIME

static int __init genann_example_2_init(void)
{
#ifdef MEASURE_END2END_TIME
    struct timespec micro_end2end_start, micro_end2end_stop;
    long total_end2end_micro = 0;
#endif

#ifdef MEASURE_MICROBENCHMARKS
    struct timespec micro_init_start, micro_init_stop, micro_climb_start, micro_climb_stop,
                    micro_randomize_start, micro_randomize_stop, micro_copy_start,
                    micro_copy_stop, micro_run_start, micro_run_stop, micro_free_start,
                    micro_free_stop;
    long total_init_micro = 0;
    long total_randomize_micro = 0;
    long micro_nIter_randomize = 0;
    long total_copy_micro = 0;
    long total_climb_micro = 0;
    long total_run_micro = 0;
    long total_nIter_run = 0;
    long total_free_micro = 0;
#endif

    /* Input and expected out data for the XOR function. */
    const double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const double output[4] = {0, 1, 1, 0};
    double *run_results[4];

    double err;
    double last_err = 1000;
    int count = 0;
    int i;

    printk(KERN_INFO "GENANN example 2.\n");
    printk(KERN_INFO "Train a small ANN to the XOR function using random search.\n");

     /* New network with 2 inputs,
     * 1 hidden layer of 2 neurons,
     * and 1 output. */

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_start);
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_init_start);
#endif

    genann *ann = genann_init(2, 1, 2, 1);

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_init_stop);
    total_init_micro += ELAPSED_TIME_MICRO_SEC(micro_init_start, micro_init_stop);
#endif

    do {
        ++count;
        if (count % 1000 == 0) {
            /* We're stick, start over */
#ifdef MEASURE_MICROBENCHMARKS
            getnstimeofday(&micro_randomize_start);
#endif
            genann_randomize(ann);
#ifdef MEASURE_MICROBENCHMARKS
            getnstimeofday(&micro_randomize_stop);
            total_randomize_micro += ELAPSED_TIME_MICRO_SEC(micro_randomize_start, micro_randomize_stop);
            micro_nIter_randomize++;
#endif
            last_err = 1000;
        }

#ifdef MEASURE_MICROBENCHMARKS
        getnstimeofday(&micro_copy_start);
#endif
        genann *save = genann_copy(ann);
#ifdef MEASURE_MICROBENCHMARKS
        getnstimeofday(&micro_copy_stop);
        total_copy_micro += ELAPSED_TIME_MICRO_SEC(micro_copy_start, micro_copy_stop);
#endif

        /* Take a random guess at the ANN weights */

#ifdef MEASURE_MICROBENCHMARKS
        getnstimeofday(&micro_climb_start);
#endif
        genann_hill_climb(ann, 0.5);
#ifdef MEASURE_MICROBENCHMARKS
        getnstimeofday(&micro_climb_stop);
        total_climb_micro += ELAPSED_TIME_MICRO_SEC(micro_climb_start, micro_climb_stop);
#endif

        err = 0;
        for (i = 0; i < 4; i++) {
#ifdef MEASURE_MICROBENCHMARKS
            getnstimeofday(&micro_run_start);
#endif
            double *err_tmp = (double *)genann_run(ann, input[i]);
#ifdef MEASURE_MICROBENCHMARKS
            getnstimeofday(&micro_run_stop);
            total_run_micro += ELAPSED_TIME_MICRO_SEC(micro_run_start, micro_run_stop);
            total_nIter_run ++;
#endif
            // err += pow((*err_tmp) - output[i], 2.0);
            kernel_fpu_begin();
            err += ((*err_tmp) - output[i]) * ((*err_tmp) - output[i]);
            kernel_fpu_end();
            vfree(err_tmp);
        }

        /* keep these weights if they're an improvement. */
#ifdef MEASURE_MICROBENCHMARKS
        getnstimeofday(&micro_free_start);
#endif
        if (err < last_err) {
            last_err = err;
            genann_free(save);
        } else {
            genann_free(ann);
            ann = save;
        }
#ifdef MEASURE_MICROBENCHMARKS
        getnstimeofday(&micro_free_stop);
        total_free_micro += ELAPSED_TIME_MICRO_SEC(micro_free_start, micro_free_stop);
#endif

        /* See how we did */
        kernel_fpu_begin();
        if (err <= 0.01) {
            kernel_fpu_end();
            break;
        }
        kernel_fpu_end();
    } while(!fatal_signal_pending(current));

    printk(KERN_INFO "Finished in %d loops.\n", count);

#ifdef MEASURE_MICROBENCHMARKS
        getnstimeofday(&micro_run_start);
#endif
    for (i = 0; i < 4; i++) {
        run_results[i] = (double *)genann_run(ann, input[i]);
    }
#ifdef MEASURE_MICROBENCHMARKS
        getnstimeofday(&micro_run_stop);
        total_run_micro += ELAPSED_TIME_MICRO_SEC(micro_run_start, micro_run_stop);
        total_nIter_run ++;
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_free_start);
#endif
    genann_free(ann);
#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_free_stop);
    total_free_micro += ELAPSED_TIME_MICRO_SEC(micro_free_start, micro_free_stop);
#endif

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_stop);
#endif

    kernel_fpu_begin();
    for (i = 0; i < 4; i++) {
        printk(KERN_INFO "Output for [%d, %d] is %d%%.\n", (int)input[i][0], (int)input[i][1], (int)(100 * (*run_results[i])));
        vfree(run_results[i]);
    }
    kernel_fpu_end();

#ifdef MEASURE_MICROBENCHMARKS
    PRINT(V_INFO, "Average genann init time: %ld usec\n", total_init_micro);
    PRINT(V_INFO, "Average genann randomize time: %ld usec\n", total_randomize_micro / micro_nIter_randomize);
    PRINT(V_INFO, "Average genann copy time: %ld usec\n", total_copy_micro / (long)count);
    PRINT(V_INFO, "Average genann hill climb time: %ld usec\n", total_climb_micro / (long)count);
    PRINT(V_INFO, "Average genann run time: %ld usec\n", total_run_micro / total_nIter_run);
    PRINT(V_INFO, "Average genann free time: %ld usec\n", total_free_micro / (long)(count + 1));
#endif

#ifdef MEASURE_END2END_TIME
    total_end2end_micro += ELAPSED_TIME_MICRO_SEC(micro_end2end_start, micro_end2end_stop);
    PRINT(V_INFO, "Total execution time: %ld usec\n", total_end2end_micro);
#endif

    return 0;
}

static void __exit genann_example_2_exit(void) {
    printk(KERN_INFO "ANN is freed.\n");
}

module_init(genann_example_2_init);
module_exit(genann_example_2_exit);
