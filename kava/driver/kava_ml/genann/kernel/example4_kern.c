#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/stat.h>

#include <asm/fpu/api.h>

#include "genann_kava.h"
#include "shared_memory.h"

#define BUF_LEN 1024

static char iris_data[BUF_LEN] __initdata;
module_param_string(iris_data, iris_data, BUF_LEN, S_IRUGO);

#define description_string "Kernel implementation of example 4 in genann."
#define maintainer_string "Bodun Hu"

MODULE_AUTHOR(maintainer_string);
MODULE_DESCRIPTION(description_string);
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");

//#define MEASURE_MICROBENCHMARKS
#define MEASURE_END2END_TIME

static int __init genann_ex4_init(void) {
#ifdef MEASURE_END2END_TIME
    struct timespec micro_end2end_start, micro_end2end_stop;
    long total_end2end_micro = 0;
#endif

#ifdef MEASURE_MICROBENCHMARKS
    struct timespec micro_init_start, micro_init_stop, micro_train_start,
                    micro_train_stop, micro_run_start, micro_run_stop,
                    micro_free_start, micro_free_stop;
    long total_init_micro = 0;
    long total_train_micro = 0;
    long micro_nIter_train = 0;
    long total_run_micro = 0;
    long micro_nIter_run = 0;
    long total_free_micro = 0;
#endif

    // TODO: get num of samples
    char *class_names[] = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    /* Load the data in userspace by invoking kava */
    int samples;
    int result;
    int correct = 0;
    const int inputs = 4;
    const int outputs = 3;
    double *input;
    double *class;
    const size_t lengths[] = {strlen(class_names[0]) + 1, strlen(class_names[1]) + 1, strlen(class_names[2]) + 1};
    double **guesses;

    int i, j;
    int loops = 5000;   /* Setting this too large could cause long execution time
                         * since current implementation requires training data
                         * to be copied back and forth between user and kernel
                         * space. */

    printk(KERN_INFO "GENANN example 4.\n");
    printk(KERN_INFO "Train an ANN on the IRIS dataset using backpropagation.\n");

    samples = get_data_sample_size(iris_data);
    if (samples == GENANN_FILE_ERROR) {
        pr_err("Failed to get data file: %s\n", iris_data);
    }

    input = (double *)kava_alloc(sizeof(double) * inputs * samples);
    class = (double *)kava_alloc(sizeof(double) * outputs * samples);
    guesses = (double **)vmalloc(sizeof(double *) * samples);

    /* Read the file into the given arrays */
    result = read_training_data(input, class, inputs, outputs, iris_data, class_names, lengths, samples);
    if (result == GENANN_TRAINING_DATA_ERROR || result == GENANN_FILE_ERROR) {
        pr_err("Failed to read training data into buffers.\n");
    }

#ifdef MEASURE_END2END_TIME
    getnstimeofday(&micro_end2end_start);
#endif

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_init_start);
#endif

    genann *ann = genann_init(4, 1, 4, 3);

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_init_stop);
    total_init_micro += ELAPSED_TIME_MICRO_SEC(micro_init_start, micro_init_stop);
#endif

    /* Traing the network with backpropagation */
    printk(KERN_INFO "Training for %d loops over data.\n", loops);
    for (i = 0; i < loops; i++) {
        for (j = 0; j < samples; j++) {
#ifdef MEASURE_MICROBENCHMARKS
            getnstimeofday(&micro_train_start);
#endif
            genann_train(ann, input + j*4, class + j*3, 0.01);
#ifdef MEASURE_MICROBENCHMARKS
            getnstimeofday(&micro_train_stop);
            total_train_micro += ELAPSED_TIME_MICRO_SEC(micro_train_start, micro_train_stop);
            micro_nIter_train ++;
#endif
        }
    }

#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_run_start);
#endif
    for (j = 0; j < samples; ++j) {
        guesses[j] = genann_run(ann, input + j*4);
    }
#ifdef MEASURE_MICROBENCHMARKS
    getnstimeofday(&micro_run_stop);
    total_run_micro += ELAPSED_TIME_MICRO_SEC(micro_run_start, micro_run_stop);
    micro_nIter_run ++;
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
    for (j = 0; j < samples; ++j) {
        if (class[j*3+0] == 1.0) {
            if (guesses[j][0] > guesses[j][1] && guesses[j][0] > guesses[j][2]) {++correct;}
        }
        else if (class[j*3+1] == 1.0) {
            if (guesses[j][1] > guesses[j][0] && guesses[j][1] > guesses[j][2]) {++correct;}
        }
        else if (class[j*3+2] == 1.0) {
            if (guesses[j][2] > guesses[j][0] && guesses[j][2] > guesses[j][1]) {++correct;}
        }
        else {
            pr_err("Logic error.\n");
        }
        vfree(guesses[j]);
    }
    printk(KERN_INFO "%d/%d correct (%d%%).\n", correct, samples, (int)((double)correct / samples * 100.0));
    kernel_fpu_end();

#ifdef MEASURE_MICROBENCHMARKS
    PRINT(V_INFO, "Average genann init time: %ld usec\n", total_init_micro);
    PRINT(V_INFO, "Average genann train time: %ld usec\n", total_train_micro / micro_nIter_train);
    PRINT(V_INFO, "Average genann run time: %ld usec\n", total_run_micro / micro_nIter_run);
    PRINT(V_INFO, "Average genann free time: %ld usec\n", total_free_micro);
#endif
    kava_free(input);
    kava_free(class);
    vfree(guesses);

#ifdef MEASURE_END2END_TIME
    total_end2end_micro = ELAPSED_TIME_MICRO_SEC(micro_end2end_start, micro_end2end_stop);
    PRINT(V_INFO, "Total execution time: %ld usec\n", total_end2end_micro);
#endif
    return 0;
}

static void __exit genann_ex4_exit(void) {
    printk(KERN_INFO "ANN is freed.\n");
}

module_init(genann_ex4_init);
module_exit(genann_ex4_exit);
