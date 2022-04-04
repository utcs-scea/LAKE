#include <stdio.h>
#include <stdlib.h>
#ifdef KAVA_HAS_GPU
#include <cuda.h>
#endif

#define kava_is_worker 1

#include "api.h"
#include "channel.h"
#include "command_handler.h"
#include "debug.h"
#include "endpoint.h"

#include <worker.h>
#include "genann_kava_utilities.h"
#include "../klib/genann/genann_kava.h"

#define WORKER_ASSERTION_MSG \
    "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, expecially using `strlen(s)` instead of `strlen(s)+1`)"

static char seeded = 0;

static struct kava_endpoint __kava_endpoint;

typedef struct {
    int inputs;
    int hidden_layers;
    int hidden;
    int outputs;
} Metadata;

struct genann_metadata {
    struct kava_metadata_base base;
    Metadata application;
};

static void __handle_command_genann_init();
static void __handle_command_genann_destroy();
void __handle_command_genann(struct kava_chan *__chan, const struct kava_cmd_base *__cmd);
void __print_command_genann(FILE *file, const struct kava_chan *__chan,
        const struct kava_cmd_base *__cmd);

#define kava_metadata(p) (&((struct genann_metadata *)kava_internal_metadata(&__kava_endpoint, p))->application)

void enable_constructor(void) { /* Nothing */ }

void __attribute((constructor)) init_genann_worker(void) {
    worker_common_init();
    __handle_command_genann_init();
#ifdef KAVA_HAS_GPU
    //cuInit(0);
#endif
}

void __handle_command_genann_init() {
    kava_endpoint_init(&__kava_endpoint, sizeof(struct genann_metadata));
    kava_register_cmd_handler(KAVA_CMD_MODE_API,
                            __handle_command_genann,
                            __print_command_genann);
}

void __attribute((destructor)) fini_genann_worker(void) {
    worker_common_fini();
}

void __handle_command_genann_destroy() {
    kava_endpoint_destroy(&__kava_endpoint);
}

static void
__wrapper_genann_train(genann const *ann, double const *inputs,
        double const *desired_outputs, double learning_rate) {
    genann_train(ann, inputs, desired_outputs, learning_rate);
}

static void
__wrapper_genann_randomize(genann *ann) {
    genann_randomize(ann);
}

static genann *
__wrapper_genann_copy(genann const *ann) {
    genann *ret = genann_copy(ann);
    return ret;
}

static double const *
__wrapper_genann_run(genann const *ann, double const *inputs) {
    double const * ret = genann_run(ann, inputs);
    return ret;
}

static genann *
__wrapper_genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    genann *ret;
    ret = genann_init(inputs, hidden_layers, hidden, outputs);
    return ret;
}

static void
__wrapper_genann_free(genann *ann) {
    genann_free(ann);
}

/*******************************************************************************
 * genann util
 ******************************************************************************/

static int
__wrapper_get_data_sample_size(const char *file) {
    FILE *in = fopen(file, "r");
    int ret = 0;
    if (!in) {
        pr_err("Could not open file: %s\n", file);
        ret = GENANN_FILE_ERROR;
        return ret;
    }
    /* Loop through the data to get a count. */
    char line[1024];
    while (!feof(in) && fgets(line, 1024, in)) {
        ++ret;
    }

    fclose(in);
    return ret;
}

static int
__wrapper_read_training_data(double *input, double *class,
        const int inputs, const int outputs, char *file,
        char *class_name[], const int samples) {
    FILE *in = fopen(file, "r");
    int ret = 1;
    if (!in) {
        pr_err("Could not open file: %s\n", file);
        ret = GENANN_FILE_ERROR;
        return ret;
    }
    char line[1024];
    int i, j, tmp, tmp2;
    for (i = 0; i < samples; ++i) {
        double *p = input + i * inputs;
        double *c = class + i * outputs;
        for (tmp = 0; tmp < outputs; tmp++) {
            c[tmp] = 0.0;
        }

        if (fgets(line, 1024, in) == NULL) {
            pr_err("read_trainning_data failed due to fgets\n");
            ret = GENANN_FILE_ERROR;
            return ret;
        }

        char *split = strtok(line, ",");
        for (j = 0; j < inputs; ++j) {
            p[j] = atof(split);
            split = strtok(0, ",");
        }

        split[strlen(split)-1] = 0;
        int matches = 0;
        for (tmp2 = 0; tmp2 < outputs; tmp2++) {
            if (strcmp(split, class_name[tmp2]) == 0) {
                c[tmp2] = 1.0;
                matches ++;
                break;
            }
        }
        if (matches == 0) {
            pr_err("read_training_data error: unknown class %s.\n", split);
            ret = GENANN_TRAINING_DATA_ERROR;
            return ret;
        }
    }

    fclose(in);
}

static void
__wrapper_genann_hill_climb(genann const *ann, const double rate) {
    if (seeded == 0) {
        srand(time(0));
        seeded = 1;
    }
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        ann->weight[i] += ((double)rand()/RAND_MAX-rate);
    }
}

static double
__wrapper_pow(double x, double y) {
    return pow(x, y);
}

static genann *
__wrapper_genann_read_file(const char *file) {
    if (file == NULL) {
        pr_err("genann_read_file: received invalid file path\n");
        return NULL;
    }
    FILE *saved = fopen(file, "r");
    if (!saved) {
        pr_err("genann_read_file: Couldn't open file: %s\n", file);
        return NULL;
    }

    genann *ann = genann_read(saved);
    fclose(saved);
    if (!ann) {
        pr_err("genann_read_file: Error loading ANN from file: %s.\n", file);
        return NULL;
    }

    return ann;
}

/*******************************************************************************
 * end of genann util
 ******************************************************************************/


void __handle_command_genann(struct kava_chan *__chan,
                            const struct kava_cmd_base *__cmd) {
    __chan->cmd_print(__chan, __cmd);

    switch (__cmd->command_id) {
        case CALL_GENANN___GENANN_INIT:
            {
                struct genann_genann_init_call *__call = (struct genann_genann_init_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_genann_init_call) &&
                        WORKER_ASSERTION_MSG);

                /* Input: int inputs, int hidden_layers, int hidden, int outputs */
                int inputs, hidden_layers, hidden, outputs;
                {
                    inputs = (int)__call->inputs;
                    hidden_layers = (int)__call->hidden_layers;
                    hidden = (int)__call->hidden;
                    outputs = (int)__call->outputs;
                }

                /* Perform call */
                genann *ret;
                ret = __wrapper_genann_init(inputs, hidden_layers, hidden, outputs);

                size_t __total_buffer_size = 0;
                {
                }
                struct genann_genann_init_ret *__ret =
                    (struct genann_genann_init_ret *)__chan->cmd_new(__chan,
                            sizeof(struct genann_genann_init_ret), __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GENANN_INIT;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;

                {
                    /* Output: genann *ret */
                    __ret->ret = ret;
                }

                /* Send reply genann *ann */
                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
                break;
            }

        case CALL_GENANN___GENANN_TRAIN:
            {
                struct genann_genann_train_call *__call = (struct genann_genann_train_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_genann_train_call) &&
                        WORKER_ASSERTION_MSG);

                genann *ann;
                {
                    /* Input: genann *ann*/
                    ann = __call->ann;
                }

                double *inputs;
                {
                    /* Input: double const *inputs */
                    if ((__call->inputs) != NULL) {
                        if ((__call->__shm_inputs)) {
                            inputs = kava_shm_address((long)__call->inputs);
                        }
                        else {
                            inputs = ((double *)__chan->chan_get_buffer(__chan, __cmd, __call->inputs));
                        }
                    }
                    else {
                        inputs = NULL;
                    }
                }

                double *desired_outputs;
                {
                    /* Input: double const *desired_outputs */
                    if ((__call->desired_outputs) != NULL) {
                        if ((__call->__shm_desired_outputs)) {
                            desired_outputs = kava_shm_address((long)__call->desired_outputs);
                        }
                        else {
                            desired_outputs = ((double *)__chan->chan_get_buffer(__chan,__cmd, __call->desired_outputs));
                        }
                    }
                    else {
                        desired_outputs = NULL;
                    }
                }

                double learning_rate;
                {
                    learning_rate = __call->learning_rate;
                }

                __wrapper_genann_train(ann, inputs, desired_outputs, learning_rate);

                /*
                size_t __total_buffer_size = 0;
                {
                }
                struct genann_genann_train_ret *__ret =
                    (struct genann_genann_train_ret *)__chan->cmd_new(
                            __chan, sizeof(struct genann_genann_train_ret),
                            __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GENANN_TRAIN;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;
                {
                }

                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
                */

                break;
            }

        case CALL_GENANN___GENANN_COPY:
            {
                struct genann_genann_copy_call *__call = (struct genann_genann_copy_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_genann_copy_call) &&
                        WORKER_ASSERTION_MSG);

                genann *ann;
                {
                    ann = __call->ann;
                }

                /* Call wrapper function */
                genann *ret =__wrapper_genann_copy(ann);

                size_t __total_buffer_size = 0;
                {
                }
                struct genann_genann_copy_ret *__ret =
                    (struct genann_genann_copy_ret *)__chan->cmd_new(__chan,
                            sizeof(struct genann_genann_copy_ret), __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GENANN_COPY;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;

                {
                    /* Output: genann *ret */
                    __ret->ret = ret;
                }

                /* Send reply genann *ann */
                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);

                break;
            }

        case CALL_GENANN___GENANN_RUN:
            {
                struct genann_genann_run_call *__call = (struct genann_genann_run_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_genann_run_call) &&
                        WORKER_ASSERTION_MSG);

                genann *ann;
                {
                    ann = __call->ann;
                }

                double *inputs;
                {
                    if ((__call->inputs) != NULL) {
                        if ((__call->__shm_inputs)) {
                            inputs = kava_shm_address((long)__call->inputs);
                        }
                        else {
                            inputs = ((double *)__chan->chan_get_buffer(__chan, __cmd, __call->inputs));
                        }
                    }
                    else {
                        inputs = NULL;
                    }
                }

                double const *ret = __wrapper_genann_run(ann, inputs);

                size_t __total_buffer_size = 0;
                {
                    if (ret != NULL) {
                        __total_buffer_size += __chan->chan_buffer_size(__chan, (sizeof(double) * ann->outputs));
                    }
                }
                struct genann_genann_run_ret *__ret =
                    (struct genann_genann_run_ret *)__chan->cmd_new(__chan,
                            sizeof(struct genann_genann_run_ret), __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GENANN_RUN;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;

                {
                    if (ret != NULL) {
                        __ret->ret =
                            (double *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                                    ret, sizeof(double) * ann->outputs);
                    } else {
                        __ret->ret = NULL;
                    }
                }

                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
                break;
            }

        case CALL_GENANN___GENANN_RANDOMIZE:
            {
                struct genann_genann_randomize_call *__call = (struct genann_genann_randomize_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_genann_randomize_call) &&
                        WORKER_ASSERTION_MSG);

                genann *ann;
                {
                    ann = (genann *)__call->ann;
                }

                /* Call wrapper */
                __wrapper_genann_randomize(ann);

                /*
                size_t __total_buffer_size = 0;
                {}

                struct genann_genann_randomize_ret *__ret =
                    (struct genann_genann_randomize_ret *)__chan->cmd_new(
                            __chan, sizeof(struct genann_genann_randomize_ret),
                            __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GENANN_RANDOMIZE;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;
                {
                }

                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
                */

                break;
            }

        case CALL_GENANN___GENANN_FREE:
            {
                struct genann_genann_free_call *__call = (struct genann_genann_free_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_genann_free_call) &&
                        WORKER_ASSERTION_MSG);

                genann *ann;
                {
                    ann = __call->ann;
                }

                /* Call wrapper function */
                __wrapper_genann_free(ann);
                /*
                size_t __total_buffer_size = 0;
                {
                }

                struct genann_genann_free_ret *__ret =
                    (struct genann_genann_free_ret *)__chan->cmd_new(
                            __chan, sizeof(struct genann_genann_free_ret),
                            __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GENANN_FREE;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;
                {
                }

                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
                */

                break;
            }
    /***************************************************************************
     * genann utils
     **************************************************************************/

        case CALL_GENANN___GET_DATA_SAMPLE_SIZE:
            {
                struct genann_get_data_sample_size_call *__call =
                    (struct genann_get_data_sample_size_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_get_data_sample_size_call) &&
                        WORKER_ASSERTION_MSG);

                /* Inputs: char *file */
                const char *file;
                {
                    file = ((__call->file) != NULL) ?
                        ((char *)__chan->chan_get_buffer(__chan, __cmd, __call->file)) :
                        ((char *)__call->file);
                }

                /* Perform the call */
                int ret;
                ret = __wrapper_get_data_sample_size(file);

                size_t __total_buffer_size = 0;
                {
                }

                struct genann_get_data_sample_size_ret *__ret=
                    (struct genann_get_data_sample_size_ret *)__chan->cmd_new(
                            __chan, sizeof(struct genann_get_data_sample_size_ret),
                            __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GET_DATA_SAMPLE_SIZE;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;

                {
                    __ret->ret = ret;
                }

                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
                break;
            }

        case CALL_GENANN___READ_TRAINING_DATA:
            {
                GPtrArray *__kava_alloc_list_read_training_data =
                    g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
                struct genann_read_training_data_call *__call =
                    (struct genann_read_training_data_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_read_training_data_call) &&
                        WORKER_ASSERTION_MSG);

                const int inputs = (int)__call->inputs;
                const int outputs = (int)__call->outputs;
                const int samples = (int)__call->samples;

                size_t *lengths;
                {
                    lengths = (__call->lengths != NULL) ?
                        ((size_t *)__chan->chan_get_buffer(__chan, __cmd, __call->lengths)) :
                        ((size_t *)__call->lengths);
                }

                double *input;
                {
                    if ((__call->input) != (NULL)) {
                        if ((__call->__shm_input)) {
                            input = kava_shm_address((long)__call->input);
                        }
                        else {
                            const size_t __size = ((size_t) (samples * inputs));
                            input = (double *)malloc(__size * sizeof(double));
                            g_ptr_array_add(__kava_alloc_list_read_training_data,
                                    kava_buffer_with_deallocator_new(free, input));
                        }
                    }
                    else {
                        input = NULL;
                    }
                }

                double *label;
                {
                    if ((__call->label) != (NULL)) {
                        if ((__call->__shm_label)) {
                            label = kava_shm_address((long)__call->label);
                        }
                        else {
                            const size_t __size = ((size_t) (samples * outputs));
                            label = (double *)malloc(__size * sizeof(double));
                            g_ptr_array_add(__kava_alloc_list_read_training_data,
                                    kava_buffer_with_deallocator_new(free, label));
                        }
                    }
                    else {
                        label = NULL;
                    }
                }

                const char *file;
                {
                    file = (__call->file != NULL) ?
                        ((char *)__chan->chan_get_buffer(__chan, __cmd, __call->file)) :
                        ((char *)__call->file);
                }

                // const char *class_name[];
                const char **class_name;
                {
                    if (__call->class_name != NULL) {
                        char **__src_class_name = (char **)__chan->chan_get_buffer(__chan, __cmd, __call->class_name);
                        volatile size_t __buffer_size = (size_t)outputs;
                        class_name = (char **)malloc(__buffer_size * sizeof(char *));
                        g_ptr_array_add(__kava_alloc_list_read_training_data,
                                kava_buffer_with_deallocator_new(free, class_name));

                        const size_t __class_name_size_0 = __buffer_size;
                        for (size_t __class_name_index_0 = 0; __class_name_index_0 < __class_name_size_0;
                                __class_name_index_0++) {
                            char **__class_name_a_0 = (char **)class_name + __class_name_index_0;
                            char **__class_name_b_0 = (char **)__src_class_name + __class_name_index_0;
                            {
                                *__class_name_a_0 = ((*__class_name_b_0) != NULL) ?
                                    ((char *)__chan->chan_get_buffer(__chan, __cmd, *__class_name_b_0)) :
                                    ((char *)*__class_name_b_0);
                            }
                        }
                    } else {
                        __call->class_name = NULL;
                    }
                }

                /* Call the wrapper */
                int ret;
                ret = __wrapper_read_training_data(input, label, inputs,
                        outputs, file, class_name,
                        samples);

                size_t __total_buffer_size = 0;

                {
                    if (input != NULL && inputs > 0 && !__call->__shm_input) {
                        __total_buffer_size += __chan->chan_buffer_size(__chan,
                                (size_t)sizeof(double) * samples * inputs);
                    }

                    if (label != NULL && outputs > 0 && !__call->__shm_label) {
                        __total_buffer_size += __chan->chan_buffer_size(__chan,
                                (size_t)sizeof(double) * samples * outputs);
                    }
                }

                struct genann_read_training_data_ret *__ret =
                    (struct genann_read_training_data_ret *)__chan->cmd_new(__chan,
                            sizeof(struct genann_read_training_data_ret),
                            __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___READ_TRAINING_DATA;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;

                {
                    __ret->ret = ret;
                }

                {
                    /* return buffer input and lebl back to kernel */
                    if (input != NULL && inputs > 0) {
                        if (__call->__shm_input) {
                            __ret->input = __call->input;
                        }
                        else {
                            __ret->input = (double *)__chan->chan_attach_buffer(__chan,
                                    (struct kava_cmd_base *)__ret, input,
                                    (size_t)inputs * samples * sizeof(double));
                        }
                    } else {
                        __ret->input = NULL;
                    }

                    if (label != NULL && outputs > 0) {
                        if (__call->__shm_label) {
                           __ret->label = __call->label;
                        }
                        else {
                            __ret->label = (double *)__chan->chan_attach_buffer(__chan,
                                    (struct kava_cmd_base *)__ret, label,
                                    (size_t)outputs * samples * sizeof(double));
                        }
                    } else {
                        __ret->label = NULL;
                    }
                }
                /* Send reply message */
                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
                g_ptr_array_unref(__kava_alloc_list_read_training_data);

                break;
            }

        case CALL_GENANN___GENANN_HILL_CLIMB:
            {
                struct genann_genann_hill_climb_call *__call = (struct genann_genann_hill_climb_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_genann_hill_climb_call) &&
                        WORKER_ASSERTION_MSG);

                genann *ann;
                {
                    ann = (genann *)__call->ann;
                }

                double rate;
                {
                    rate = __call->rate;
                }

                /* Call wrapper function */
                __wrapper_genann_hill_climb(ann, rate);
                size_t __total_buffer_size = 0;
                {
                }

                struct genann_genann_hill_climb_ret *__ret =
                    (struct genann_genann_hill_climb_ret *)__chan->cmd_new(
                            __chan, sizeof(struct genann_genann_hill_climb_ret),
                            __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GENANN_HILL_CLIMB;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;
                {
                }

                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
                break;
            }

        case CALL_GENANN___POW:
            {
                struct genann_pow_call *__call =
                    (struct genann_pow_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_pow_call) &&
                        WORKER_ASSERTION_MSG);

                double x;
                {
                    x = __call->x;
                }

                double y;
                {
                    y = __call->y;
                }

                /* Call wrapper func */
                double ret = __wrapper_pow(x, y);

                size_t __total_buffer_size = 0;
                {}

                struct genann_pow_ret *__ret =
                    (struct genann_pow_ret *)__chan->cmd_new(__chan,
                            sizeof(struct genann_pow_ret), __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GENANN_HILL_CLIMB;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;
                {
                    __ret->ret = ret;
                }

                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);

                break;
            }

        case CALL_GENANN___GENANN_READ_FILE:
            {
                struct genann_genann_read_file_call *__call =
                    (struct genann_genann_read_file_call *)__cmd;
                assert(__call->base.mode == KAVA_CMD_MODE_API);
                assert(__call->base.command_size == sizeof(struct genann_genann_read_file_call) &&
                        WORKER_ASSERTION_MSG);

                /* Inputs: char *file */
                const char *file;
                {
                    file = ((__call->file) != NULL) ?
                        ((char *)__chan->chan_get_buffer(__chan, __cmd, __call->file)) :
                        ((char *)__call->file);
                }

                /* Perform the call */
                genann *ret;
                ret = __wrapper_genann_read_file(file);

                size_t __total_buffer_size = 0;
                {
                }

                struct genann_genann_read_file_ret *__ret=
                    (struct genann_genann_read_file_ret *)__chan->cmd_new(
                            __chan, sizeof(struct genann_genann_read_file_ret),
                            __total_buffer_size);
                __ret->base.mode = KAVA_CMD_MODE_API;
                __ret->base.command_id = RET_GENANN___GENANN_READ_FILE;
                __ret->base.thread_id = __call->base.thread_id;
                __ret->__call_id = __call->__call_id;

                {
                    /* Output: genann *ret */
                    __ret->ret = ret;
                    __ret->ann_inputs = ret->inputs;
                    __ret->ann_outputs = ret->outputs;
                }

                __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
                break;
            }

    /***************************************************************************
     * end of genann utils
     **************************************************************************/


        default:
            {
                pr_err("Unrecognized genann command: %lu\n", __cmd->command_id);
                break;
            }
    }
}

void __print_command_genann(FILE *file, const struct kava_chan *__chan,
        const struct kava_cmd_base *__cmd) {

    switch (__cmd->command_id) {
        case CALL_GENANN___GENANN_INIT:
            {
                pr_info("genann_init is invoked\n");
                break;
            }
        case CALL_GENANN___GENANN_RANDOMIZE:
            {
                pr_info("genann_randomize is invoked\n");
                break;
            }
        case CALL_GENANN___GENANN_COPY:
            {
                pr_info("genann_copy is invoked\n");
                break;
            }
        case CALL_GENANN___GENANN_TRAIN:
            {
                pr_info("genann_train is invoked\n");
                break;
            }
        case CALL_GENANN___GENANN_RUN:
            {
                pr_info("genann_run is invoked\n");
                break;
            }
        case CALL_GENANN___GENANN_FREE:
            {
                pr_info("genann_free is invoked\n");
                break;
            }
        case CALL_GENANN___GET_DATA_SAMPLE_SIZE:
            {
                pr_info("get_data_sample_size is invoked\n");
                break;
            }
        case CALL_GENANN___READ_TRAINING_DATA:
            {
                pr_info("read_training_data is invoked\n");
                break;
            }
        case CALL_GENANN___GENANN_HILL_CLIMB:
            {
                pr_info("genann_hill_climb is invoked\n");
                break;
            }
        case CALL_GENANN___POW:
            {
                pr_info("pow is invoked\n");
                break;
            }
        case CALL_GENANN___GENANN_READ_FILE:
            {
                pr_info("genann_read_file is invoked\n");
                break;
            }
        default:
            {
                pr_err("Unrecognized Genann command: %lu\n", __cmd->command_id);
                break;
            }
    }
}

