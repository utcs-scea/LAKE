/*
 * Copyright (c) 2019-2021 Ibrahim Umit Akgun
 * Copyright (c) 2019-2021 Ali Selman Aydin
 * Copyright (c) 2019-2021 Erez Zadok
 * Copyright (c) 2019-2021 Stony Brook University
 * Copyright (c) 2019-2021 The Research Foundation of SUNY
 *
 * You can redistribute it and/or modify it under the terms of the Apache
 * License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
 */

#ifndef LINEAR_H
#define LINEAR_H

#include "matrix.h"

typedef struct linear_layer {
  matrix *w;
  matrix *bias_vector;
  matrix *prev_bias_vector;
  matrix *gradient;
  matrix *bias_gradient;
  matrix *input, *output;
} linear_layer;


#endif
