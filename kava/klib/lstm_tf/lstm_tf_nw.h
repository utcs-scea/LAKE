#ifndef __LSTM_TF_NW_H__
#define __LSTM_TF_NW_H__

#include "command.h"
#include "util.h"

#include "lstm_tf_nw_types.h"

#define LSTM_TF_API 3

enum lstm_tf_functions {
    CALL_LSTM_TF_LOAD_MODEL, RET_LSTM_TF_LOAD_MODEL, CALL_LSTM_TF_CLOSE_CTX, RET_LSTM_TF_CLOSE_CTX, CALL_LSTM_TF_DOGC,
        CALL_LSTM_TF_STANDARD_INFERENCE, RET_LSTM_TF_STANDARD_INFERENCE,
        CALL_LSTM_TF_KLEIO_LOAD_MODEL, RET_LSTM_TF_KLEIO_LOAD_MODEL, RET_LSTM_TF_DOGC,
        CALL_LSTM_TF_KLEIO_INFERENCE, RET_LSTM_TF_KLEIO_INFERENCE,
        CALL_LSTM_TF_KLEIO_CLOSE_CTX, RET_LSTM_TF_KLEIO_CLOSE_CTX
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

struct lstm_tf_dogc_call {
    struct kava_cmd_base base;
    intptr_t __call_id;

};

struct lstm_tf_dogc_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;

};

struct lstm_tf_dogc_call_record {
    char __handler_deallocate;
    volatile char __call_complete;
};



struct lstm_tf_close_ctx_call {
    struct kava_cmd_base base;
    intptr_t __call_id;

};

struct lstm_tf_close_ctx_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;

};

struct lstm_tf_close_ctx_call_record {
    char __handler_deallocate;
    volatile char __call_complete;
};


struct lstm_tf_kleio_close_ctx_call {
    struct kava_cmd_base base;
    intptr_t __call_id;

};

struct lstm_tf_kleio_close_ctx_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;

};

struct lstm_tf_kleio_close_ctx_call_record {
    char __handler_deallocate;
    volatile char __call_complete;
};


struct lstm_tf_standard_inference_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int num_syscall;
    unsigned int sliding_window;
    void *syscalls;
};

struct lstm_tf_standard_inference_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int ret;
};

struct lstm_tf_standard_inference_call_record {
    unsigned int num_syscall;
    unsigned int sliding_window;
    void *syscalls;
    int ret;
    char __handler_deallocate;
    volatile char __call_complete;
};


struct lstm_tf_kleio_inference_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int num_syscall;
    unsigned int sliding_window;
    void *syscalls;
};

struct lstm_tf_kleio_inference_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int ret;
};

struct lstm_tf_kleio_inference_call_record {
    unsigned int num_syscall;
    unsigned int sliding_window;
    void *syscalls;
    int ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

#endif                                           // ndef __LSTM_TF_NW_H__
