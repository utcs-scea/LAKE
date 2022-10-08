#ifndef __LINNOS_PREDICTORS_H
#define __LINNOS_PREDICTORS_H

#define FEAT_31
#define LEN_INPUT 31
#define LEN_LAYER_0 256
#define LEN_LAYER_0_HALF 128
#define LEN_LAYER_1 2

#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdbool.h>
#endif

bool fake_prediction_model(char *feat_vec, int n_vecs, long **weights);
bool gpu_prediction_model(char *feat_vec, int n_vecs, long **weights);
bool cpu_prediction_model(char *feat_vec, int n_vecs, long **weights);

#endif