#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/stat.h>

#include <asm/fpu/api.h>

#include "genann_kava.h"

#define BUF_LEN 1024

/* example/xor.ann */
static char save_name[BUF_LEN] __initdata;
module_param_string(save_name, save_name, BUF_LEN, S_IRUGO);

#define description_string "Kernel implementation of example 3 in genann."
#define maintainer_string "Bodun Hu"

MODULE_AUTHOR(maintainer_string);
MODULE_DESCRIPTION(description_string);
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");

#undef MEASURE_MICROBENCHMARKS
#define MEASURE_END2END_TIME

static int __init genann_ex3_init(void) {
#ifdef MEASURE_END2END_TIME
    struct timespec micro_end2end_start, micro_end2end_stop;
    long total_end2end_micro = 0;
#endif

#ifdef MEASURE_MICROBENCHMARKS
    struct timespec micro_read_file_start, micro_read_file_stop, micro_run_start,
                    micro_run_stop, micro_free_start, micro_free_stop;
    long total_read_file_micro = 0;
    long total_run_micro = 0;
    long total_free_micro = 0;
#endif

    const double input[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double *run_results[4];
    int i;

    printk(KERN_INFO "GENANN example 3.\n");
    printk(KERN_INFO "Load a saved ANN to solve the XOR function.\n");

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_start);
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_read_file_start);
#endif

    genann *ann = genann_read_file(save_name);
    if (!ann) {
        pr_err("Error loading ANN from file: %s.\n", save_name);
    }

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_read_file_stop);
    total_read_file_micro += ELAPSED_TIME_MICRO_SEC(micro_read_file_start, micro_read_file_stop);
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_run_start);
#endif
    for (i = 0; i < 4; i++) {
        run_results[i] = (double *)genann_run(ann, input[i]);
    }
#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_run_stop);
    total_run_micro += ELAPSED_TIME_MICRO_SEC(micro_run_start, micro_run_stop);
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
        printk(KERN_INFO "Output for [%d, %d] is %d%%.\n", (int)input[i][0], (int)input[i][1],
               (int)(100 * (*run_results[i])));
        vfree(run_results[i]);
    }
    kernel_fpu_end();

#ifdef MEASURE_MICROBENCHMARKS
    PRINT(V_INFO, "Average genann read file time: %ld usec\n", total_read_file_micro);
    PRINT(V_INFO, "Average genann run time: %ld usec\n", total_run_micro / 4);
    PRINT(V_INFO, "Average genann free time: %ld usec\n", total_free_micro);
#endif

#ifdef MEASURE_END2END_TIME
    total_end2end_micro = ELAPSED_TIME_MICRO_SEC(micro_end2end_start, micro_end2end_stop);
    PRINT(V_INFO, "Total execution time: %ld usec\n", total_end2end_micro);
#endif

    return 0;
}

static void __exit genann_ex3_exit(void) {
    printk(KERN_INFO "ANN is freed.\n");
}

module_init(genann_ex3_init);
module_exit(genann_ex3_exit);
