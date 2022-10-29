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

#include <linux/slab.h>
#include "matrix.h"


static int sizeof_dtype(dtype type) {
  int size = 0;
  switch (type) {
    case INTEGER: {
      size = sizeof(int);
      break;
    }
    case FLOAT: {
      size = sizeof(float);
      break;
    }
    case DOUBLE: {
      size = sizeof(double);
      break;
    }
  }
  return size;
}

matrix *allocate_matrix(int number_of_rows, int number_of_cols,
                        dtype type_of_matrix) {
    void *allocated_memory;
    matrix *ret;

    //kml_assert(number_of_rows != 0 && number_of_cols != 0);
    //ret = kml_calloc(1, sizeof(matrix));
    ret = kcalloc(1, sizeof(matrix), GFP_KERNEL);
    if (ret == NULL) return NULL;

    ret->rows = number_of_rows;
    ret->cols = number_of_cols;
    ret->type = type_of_matrix;

    //kml_calloc(ret->rows * ret->cols, sizeof_dtype(type_of_matrix));
    allocated_memory = kcalloc(ret->rows * ret->cols, sizeof_dtype(type_of_matrix), GFP_KERNEL);
    if (allocated_memory == NULL) {
        kfree(ret);
        return NULL;
    }

    switch (ret->type) {
        case INTEGER:
        ret->vals.i = (int *)allocated_memory;
        break;
        case FLOAT:
        ret->vals.f = (float *)allocated_memory;
        break;
        case DOUBLE:
        ret->vals.d = (double *)allocated_memory;
        break;
    }

  return ret;
}

void free_matrix(matrix *m) {
  if (m == NULL) return;
  switch (m->type) {
    case INTEGER:
      kfree(m->vals.i);
      break;
    case FLOAT:
      kfree(m->vals.f);
      break;
    case DOUBLE:
      kfree(m->vals.d);
      break;
  }
  kfree(m);
}

matrix *matrix_transpose(matrix *m) {
  int row_idx, col_idx;

  matrix *ret = allocate_matrix(m->cols, m->rows, m->type);
  if (ret == NULL) {
    return NULL;
  }

  foreach_mat(m, rows, row_idx) {
    foreach_mat(m, cols, col_idx) {
      switch (m->type) {
        case INTEGER:
          ret->vals.i[mat_index(ret, col_idx, row_idx)] =
              m->vals.i[mat_index(m, row_idx, col_idx)];
          break;
        case FLOAT:
          ret->vals.f[mat_index(ret, col_idx, row_idx)] =
              m->vals.f[mat_index(m, row_idx, col_idx)];
          break;
        case DOUBLE:
          ret->vals.d[mat_index(ret, col_idx, row_idx)] =
              m->vals.d[mat_index(m, row_idx, col_idx)];
          break;
      }
    }
  }

  return ret;
}


matrix *matrix_mult(matrix *src, matrix *mult) {
  int row_idx, col_idx, dest_col_idx;
  matrix *ret;

  kml_assert(src->cols == mult->rows && src->type == mult->type);

  ret = allocate_matrix(src->rows, mult->cols, src->type);
  if (ret == NULL) return NULL;

  foreach_mat(src, rows, row_idx) {
    foreach_mat(mult, cols, dest_col_idx) {
      foreach_mat(src, cols, col_idx) {
        switch (src->type) {
          case INTEGER:
            ret->vals.i[mat_index(ret, row_idx, dest_col_idx)] +=
                src->vals.i[mat_index(src, row_idx, col_idx)] *
                mult->vals.i[mat_index(mult, col_idx, dest_col_idx)];
            break;
          case FLOAT:
            ret->vals.f[mat_index(ret, row_idx, dest_col_idx)] +=
                src->vals.f[mat_index(src, row_idx, col_idx)] *
                mult->vals.f[mat_index(mult, col_idx, dest_col_idx)];
            break;
          case DOUBLE:
            ret->vals.d[mat_index(ret, row_idx, dest_col_idx)] +=
                src->vals.d[mat_index(src, row_idx, col_idx)] *
                mult->vals.d[mat_index(mult, col_idx, dest_col_idx)];
            break;
        }
      }
    }
  }

  return ret;
}


matrix *matrix_repmat(matrix *m, int row_repeat, int col_repeat) {
  int col_copy, row_copy, row_idx, col_idx;
  matrix *ret =
      allocate_matrix(row_repeat * m->rows, col_repeat * m->cols, m->type);

  if (col_repeat > 1) {
    foreach_mat(m, rows, row_idx) {
      for (col_copy = 0; col_copy < ret->cols; col_copy += m->cols) {
        foreach_mat(m, cols, col_idx) {
          switch (ret->type) {
            case FLOAT:
              ret->vals.f[mat_index(ret, row_idx, (col_copy + col_idx))] =
                  m->vals.f[mat_index(m, row_idx, col_idx)];
              break;
            case DOUBLE:
              ret->vals.d[mat_index(ret, row_idx, (col_copy + col_idx))] =
                  m->vals.d[mat_index(m, row_idx, col_idx)];
              break;
            case INTEGER:
              kml_assert(false);  // not implemented
              break;
          }
        }
      }
    }
  } else {
    foreach_mat(m, rows, row_idx) {
      foreach_mat(m, cols, col_idx) {
        switch (ret->type) {
          case FLOAT:
            ret->vals.f[mat_index(ret, row_idx, col_idx)] =
                m->vals.f[mat_index(m, row_idx, col_idx)];
            break;
          case DOUBLE:
            ret->vals.d[mat_index(ret, row_idx, col_idx)] =
                m->vals.d[mat_index(m, row_idx, col_idx)];
            break;
          case INTEGER:
            kml_assert(false);  // not implemented
            break;
        }
      }
    }
  }

  if (row_repeat > 1) {
    for (row_copy = m->rows; row_copy < ret->rows; row_copy += m->rows) {
      foreach_mat(m, rows, row_idx) {
        foreach_mat(ret, cols, col_idx) {
          switch (ret->type) {
            case FLOAT:
              ret->vals.f[mat_index(ret, (row_copy + row_idx), col_idx)] =
                  m->vals.f[mat_index(m, row_idx, col_idx)];
              break;
            case DOUBLE:
              ret->vals.d[mat_index(ret, (row_copy + row_idx), col_idx)] =
                  m->vals.d[mat_index(m, row_idx, col_idx)];
              break;
            case INTEGER:
              kml_assert(false);  // not implemented
              break;
          }
        }
      }
    }
  }
  return ret;
}

void matrix_add(matrix *src, matrix *add, matrix *dest) {
  int row_idx, col_idx;

  kml_assert(src->cols == add->cols && src->cols == dest->cols &&
             src->rows == add->rows && src->rows == dest->rows);

  foreach_mat(src, rows, row_idx) {
    foreach_mat(src, cols, col_idx) {
      switch (src->type) {
        case INTEGER:
          dest->vals.i[mat_index(dest, row_idx, col_idx)] =
              src->vals.i[mat_index(src, row_idx, col_idx)] +
              add->vals.i[mat_index(add, row_idx, col_idx)];
          break;
        case FLOAT:
          dest->vals.f[mat_index(dest, row_idx, col_idx)] =
              src->vals.f[mat_index(src, row_idx, col_idx)] +
              add->vals.f[mat_index(add, row_idx, col_idx)];
          break;
        case DOUBLE:
          dest->vals.d[mat_index(dest, row_idx, col_idx)] =
              src->vals.d[mat_index(src, row_idx, col_idx)] +
              add->vals.d[mat_index(add, row_idx, col_idx)];
          break;
      }
    }
  }
}