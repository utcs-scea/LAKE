#ifndef __LINNOS_PREDICTORS_H
#define __LINNOS_PREDICTORS_H

#define FEAT_31
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2

#ifdef __KERNEL__
#include <linux/types.h>
#include <linux/completion.h>
#else
#include <stdbool.h>
#endif

#ifdef __KERNEL__
//these externs are for batching
extern struct completion batch_barrier;
extern bool* gpu_results;
extern u32* window_size_hist;
bool batch_test(char *feat_vec, int n_vecs, long **weights);
#endif

bool fake_prediction_model(char *feat_vec, int n_vecs, long **weights);
void gpu_prediction_model(char *feat_vec, int n_vecs, long **weights);
bool cpu_prediction_model(char *feat_vec, int n_vecs, long **weights);

#endif