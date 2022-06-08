#ifndef __LSTM_TF_NW_H__
#define __LSTM_TF_NW_H__

#include "command.h"
#include "util.h"

#include "lstm_tf_nw_types.h"

#define LSTM_TF_API 3

enum lstm_tf_functions {
    CALL_LSTM_TF_LOAD_MODEL, RET_LSTM_TF_LOAD_MODEL,
    CALL_LSTM_TF_KLEIO_LOAD_MODEL, RET_LSTM_TF_KLEIO_LOAD_MODEL
};

#include "lstm_tf_nw_utility_types.h"

struct lstm_tf_metadata {
    struct kava_metadata_base base;
    struct metadata application;
};

struct lstm_tf_load_model_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    char *file;
};

struct lstm_tf_load_model_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int ret;
};

struct lstm_tf_load_model_call_record {
    char *file;
    int ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct lstm_tf_kleio_load_model_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    char *file;
};

struct lstm_tf_kleio_load_model_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int ret;
};

struct lstm_tf_kleio_load_model_call_record {
    char *file;
    int ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

#endif                                           // ndef __LSTM_TF_NW_H__
