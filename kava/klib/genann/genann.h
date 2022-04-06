/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * Edit: Edward Hu (bodunhu@utexas.edu) 06/19/2020
 *       Convert userspace API to kernel equivalent
 */


#ifndef __GENANN_H__
#define __GENANN_H__

#ifdef __KERNEL__
#include <asm/fpu/api.h>
#include <linux/random.h>
#include <linux/module.h> /* Needed by all modules */
#include <linux/kernel.h> /* Needed for KERN_INFO */
#else
#include <stdio.h>
#endif

#ifdef __GNUC__

#ifdef __KERNEL__
#ifdef __amd64__
#define SSE __attribute__((target("sse")))
#else
#define SSE
#endif /* __amd64__ */
#else
#define SSE
#endif /* __KERNEL__ */

#else
#define SSE

#endif /* __GNUC__ */


#ifdef __cplusplus
extern "C" {
#endif

/* Kernel assertion */
#ifdef __KERNEL__
#define ASSERT(x)                                                       \
do {    if (x) break;                                                   \
        printk(KERN_EMERG "### ASSERTION FAILED %s: %s: %d: %s\n",      \
               __FILE__, __func__, __LINE__, #x); dump_stack(); BUG();  \
} while (0)
#endif  /* __KERNEL__ */

#ifdef __KERNEL__
#define GENANN_RANDOM() ((double)(get_random_int() % 50) / 100)
#else
#define GENANN_RANDOM() ((double)rand()/RAND_MAX)
#endif /* __KERNEL__ */

struct genann;

typedef double (*genann_actfun)(const struct genann *ann, double a);

typedef struct genann {
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

    /* All weights (total_weights long). */
    double *weight;

    /* Stores input array and output of each neuron (total_neurons long). */
    double *output;

    /* Stores delta of each hidden and output neuron (total_neurons - inputs long). */
    double *delta;

    /* (Used in GPU version) Copy of the struct on GPU (GPU memory address). */
    struct genann *d_ann;

} genann;

/* Creates and returns a new ann. */
// TODO figure out ambiguous name
genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs);

/* Creates ANN from file saved with genann_write. */
#ifndef __KERNEL__
genann *genann_read(FILE *in);
#endif

/* Sets weights randomly. Called by init. */
void genann_randomize(genann *ann);

/* Returns a new copy of ann. */
genann *genann_copy(genann const *ann);

/* Frees the memory used by an ann. */
void genann_free(genann *ann);

/* Runs the feedforward algorithm to calculate the ann's output. */
SSE
double const *genann_run(genann const *ann, double const *inputs);

/* Does a single backprop update. */
void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate);

/* Saves the ann. */
#if 0
// TODO
void genann_write(genann const *ann, FILE *out);
#endif

void genann_init_sigmoid_lookup(const genann *ann);

SSE
double genann_act_sigmoid(const genann *ann, double a);

SSE
double genann_act_sigmoid_cached(const genann *ann, double a);

SSE
double genann_act_threshold(const genann *ann, double a);

SSE
double genann_act_linear(const genann *ann, double a);


#ifdef __cplusplus
}
#endif

#endif /*__GENANN_H__*/
