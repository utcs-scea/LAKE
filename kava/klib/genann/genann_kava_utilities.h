#ifndef __GENANN_KAVA_UTILITIES_H__
#define __GENANN_KAVA_UTILITIES_H__

#define kava_utility static
#define kava_begin_utility
#define kava_end_utility

kava_begin_utility;
#include "genann.h"
#include "mnist.h"
kava_end_utility;

#ifdef __KERNEL__
#include <linux/time.h>
#define V_ERROR 0
#define V_INFO 1
#define V_DEBUG_TIMING 2
#define V_DEBUG 3
#define VERBOSITY V_INFO
#define PRINT(verbosity, ...) do { if (verbosity <= VERBOSITY) printk(KERN_INFO __VA_ARGS__); } while (0)

#define ELAPSED_TIME_MICRO_SEC(start, stop) ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_nsec - start.tv_nsec) / 1000)
#endif

typedef struct {
    /* How many inputs, outputs, and hidden neurons. */
    int inputs, hidden_layers, hidden, outputs;

    /* Which activation function to use for hidden neurons. Default: gennann_act_sigmoid_cached*/
    genann_actfun activation_hidden;

    /* Which activation function to use for output. Default: gennann_act_sigmoid_cached*/
    genann_actfun activation_output;

    /* Total number of weights, and size of weights buffer. */
    int total_weights;

    /* Total number of neurons + inputs and size of output buffer. */
    int total_neurons;

    /* This object should hide the following three parameters so kernel can't
     * access them. */

    /* All weights (total_weights long). */
    // double *weight;

    /* Stores input array and output of each neuron (total_neurons long). */
    // double *output;

    /* Stores delta of each hidden and output neuron (total_neurons - inputs long). */
    // double *delta;

} genann_attr;

int get_data_sample_size(const char *file);
int read_training_data(double *input, double *label, const int inputs,
        const int outputs, char *file, char **class_name, const size_t *lengths,
        const int samples);
void genann_hill_climb(const genann *ann, const double rate);
SSE double pow(double x, double y);
genann *genann_read_file(const char *file);

#undef ava_utility
#undef ava_begin_utility
#undef ava_end_utility

#endif // undef __GENANN_KAVA_UTILITIES_H__

