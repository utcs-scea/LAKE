#ifndef __GENANN_KAVA_H__
#define __GENANN_KAVA_H__

#include "command.h"
#include "util.h"

#include "genann.h"
#include "genann_kava_utilities.h"

enum genann_functions {
    CALL_GENANN___GENANN_INIT,                  RET_GENANN___GENANN_INIT,
    CALL_GENANN___GENANN_COPY,                  RET_GENANN___GENANN_COPY,
    CALL_GENANN___GENANN_ACT_HIDDEN_INDIRECT,   RET_GENANN___GENANN_ACT_HIDDEN_INDIRECT,
    CALL_GENANN___GENANN_INIT_SIGMOID_LOOKUP,   RET_GENANN___GENANN_INIT_SIGMOID_LOOKUP,
    CALL_GENANN___GENANN_ACT_SIGMOID_CACHED,    RET_GENANN___GENANN_ACT_SIGMOID_CACHED,
    CALL_GENANN___GENANN_RANDOMIZE,             RET_GENANN___GENANN_RANDOMIZE,
    CALL_GENANN___GENANN_FREE,                  RET_GENANN___GENANN_FREE,
    CALL_GENANN___GENANN_RUN,                   RET_GENANN___GENANN_RUN,
    CALL_GENANN___GENANN_TRAIN,                 RET_GENANN___GENANN_TRAIN,
    /* genann utilities */
    CALL_GENANN___GET_DATA_SAMPLE_SIZE,         RET_GENANN___GET_DATA_SAMPLE_SIZE,
    CALL_GENANN___READ_TRAINING_DATA,           RET_GENANN___READ_TRAINING_DATA,
    CALL_GENANN___GENANN_HILL_CLIMB,            RET_GENANN___GENANN_HILL_CLIMB,
    CALL_GENANN___POW,                          RET_GENANN___POW,
    CALL_GENANN___GENANN_READ_FILE,             RET_GENANN___GENANN_READ_FILE,
    /* end of genann util */
};

enum genann_utils {
    GENANN_FILE_ERROR = (-1),
    GENANN_TRAINING_DATA_ERROR = (-2),
};

struct genann_genann_init_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int inputs;
    int hidden_layers;
    int hidden;
    int outputs;
};

struct genann_genann_init_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ret;
};

struct genann_genann_init_call_record {
    int inputs;
    int hidden_layers;
    int hidden;
    int outputs;
    // TODO return value
    genann *ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_copy_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ann;
};

struct genann_genann_copy_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ret;
};

struct genann_genann_copy_call_record {
    genann *ann;
    genann *ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_act_hidden_indirect_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct genann *ann;
    double a;
};

struct genann_genann_act_hidden_indirect_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    double ret;
};

struct genann_genann_act_hidden_indirect_call_record {
    struct genann *ann;
    double a;
    double ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_init_sigmoid_lookup_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ann;
};

struct genann_genann_init_sigmoid_lookup_call_record {
    genann *ann;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_act_sigmoid_cached_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ann;
    double a;
};

struct genann_genann_act_sigmoid_cached_call_record {
    genann *ann;
    double a;
    double ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_randomize_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ann;
};

struct genann_genann_randomize_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
};

struct genann_genann_randomize_call_record {
    genann *ann;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_free_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ann;
};

struct genann_genann_free_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
};

struct genann_genann_free_call_record {
    genann *ann;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_run_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ann;
    double *inputs;
    char __shm_inputs;
};

struct genann_genann_run_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    double *ret;
};

struct genann_genann_run_call_record {
    genann *ann;
    double *inputs;
    double *ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_train_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ann;
    double *inputs;
    char __shm_inputs;
    double *desired_outputs;
    char __shm_desired_outputs;
    double learning_rate;
};

struct genann_genann_train_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
};

struct genann_genann_train_call_record {
    genann *ann;
    double *inputs;
    double *desired_outputs;
    double learning_rate;
    char __handler_deallocate;
    volatile char __call_complete;
};

/*******************************************************************************
 * genann utilities
 ******************************************************************************/

struct genann_get_data_sample_size_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    char *file;
};

struct genann_get_data_sample_size_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int ret;
};

struct genann_get_data_sample_size_call_record {
    char *file;
    int ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_read_training_data_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    double *input;
    char __shm_input;
    double *label;
    char __shm_label;
    int inputs;
    int outputs;
    char *file;
    char **class_name;
    size_t *lengths;
    int samples;
};

struct genann_read_training_data_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int ret;
    double *input;
    double *label;
};

struct genann_read_training_data_call_record {
    double *input;
    double *label;
    int inputs;
    int outputs;
    char *file;
    char **class_name;
    size_t *lengths;
    int samples;
    int ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_hill_climb_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ann;
    double rate;
};

struct genann_genann_hill_climb_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
};

struct genann_genann_hill_climb_call_record {
    genann *ann;
    double rate;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_pow_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    double x;
    double y;
};

struct genann_pow_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    double ret;
};

struct genann_pow_call_record {
    double x;
    double y;
    double ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct genann_genann_read_file_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    char *file;
};

struct genann_genann_read_file_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    genann *ret;
    int ann_inputs;
    int ann_outputs;
};

struct genann_genann_read_file_call_record {
    char *file;
    genann *ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

/*******************************************************************************
 * end of genann utilities
 ******************************************************************************/


#endif
