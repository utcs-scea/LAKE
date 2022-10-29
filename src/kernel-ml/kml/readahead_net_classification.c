/*
 * Copyright (c) 2019-2021 Ibrahim Umit Akgun
 * Copyright (c) 2019-2021 Erez Zadok
 * Copyright (c) 2019-2021 Stony Brook University
 * Copyright (c) 2019-2021 The Research Foundation of SUNY
 *
 * You can redistribute it and/or modify it under the terms of the Apache
 * License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
 */


#include "readahead_net_classification.h"
#include "matrix.h"
#include "linear.h"

#define traverse_layers_forward(layers_list, traverse)            \
  for (traverse = layers_list->layer_list_head; traverse != NULL; \
       traverse = traverse->next)

matrix *linear_layer_forward(matrix *x, linear_layer *linear) {
  matrix *y_hat = allocate_matrix(x->rows, linear->w->rows, x->type);
  matrix *wx, *bias;
  matrix *wt;
  wt = matrix_transpose(linear->w);

  // wx+b
  wx = matrix_mult(x, wt);
  bias = matrix_repmat(linear->bias_vector, wx->rows, 1);
  matrix_add(wx, bias, y_hat);

  // set input & output
  linear->input = x;
  linear->output = y_hat;

  free_matrix(wx);
  free_matrix(bias);
  free_matrix(wt);

  return y_hat;
}

// matrix *autodiff_forward(layers *layer_list, matrix *input) {
//     layer *current_layer = NULL;
//     matrix *output = NULL;
//     traverse_layers_forward(layer_list, current_layer) {
//     output = linear_layer_forward(input, current_layer->internal);
//     input = output; //??
//     return output;
// }

void autodiff_forward(float *input, int batch_size) { 
    // layer 0
    out0 = allocate(w0_rows * batch_size);
    linear_layer_forward(input, w0, w0_rows, w0_cols, b0, 0, out0, batch_size);
    //layer 1
    out1 = allocate(w1_rows * out0_rows);
    linear_layer_forward(out0, w1, w1_rows, w1_cols, b1, 1, out1, batch_size);
    //layer 2
    out2 = allocate(w2_rows * out1_rows);
    linear_layer_forward(out1, w2, w2_rows, w2_cols, b2, 2, out2, batch_size);
    matrix_argmax(out2, w2_rows,out2_rows, result_cols);
}



// matrix *readahead_class_net_inference(matrix *input,
//                                       readahead_class_net *readahead) {
//   return autodiff_forward(readahead->layer_list, input);
// }

void readahead_class_net_inference(float *input, int batch_size) {
    autodiff_forward(input, batch_size);
}








// int predict_readahead_class(readahead_class_net *readahead,
//                             int current_readahead_val) {
//   matrix *normalized_data = NULL, *indv_result = NULL;
//   int class = 0;

//   normalized_data =
//       get_normalized_readahead_data(readahead, current_readahead_val);

//   // kml_debug("normalized per-disk data:\n");
//   // print_matrix(normalized_data);
//   indv_result = readahead_class_net_inference(normalized_data, readahead);
//   class = matrix_argmax(indv_result);

//   cleanup_autodiff(readahead->layer_list);
//   free_matrix(normalized_data);

//   return class;
// }

