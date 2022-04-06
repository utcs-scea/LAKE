/*******************************************************************************

  Kernel-space genann API library.

*******************************************************************************/

#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/kernel.h>

#define kava_is_worker 0

#include "api.h"
#include "channel_kern.h"
#include "control.h"
#include "command.h"
#include "command_handler.h"
#include "endpoint.h"
#include "shadow_thread_pool.h"
#include "shared_memory.h"

#include "debug.h"

#include "genann_kava.h"
#include "genann_kava_utilities.h"

#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define unused          __attrbute__((unused))
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif

#define CMD_ERR_MSG "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s) + 1`)"

static char *chan_mode = "netlink_socket";
module_param(chan_mode, charp, 0000);
MODULE_PARM_DESC(chan_mode, "kCUDA channel mode. Default netlink_socket.");

static struct kava_chan *chan;
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

#define kava_metadata(p)        (&((struct genann_metadata *)kava_internal_metadata(&__kava_endpoint, p))->application)
#define kava_metadata_remove(p) (kava_internal_metadata_remove(&__kava_endpoint, p))

static void __handle_command_genann_init(void);
static void __handle_command_genann_destroy(void);
void __handle_command_genann(struct kava_chan *__chan, const struct kava_cmd_base* __cmd);
void __print_command_genann(FILE* file, const struct kava_chan *__chan,
                        const struct kava_cmd_base* __cmd);

void __attribute__((constructor)) init_genann_worker(void) {
    __handle_command_genann_init();
}

void __handle_command_genann_init()
{
    kava_endpoint_init(&__kava_endpoint, sizeof(struct genann_metadata));
    kava_register_cmd_handler(KAVA_CMD_MODE_API,
                            __handle_command_genann,
                            __print_command_genann);
}

void __handle_command_genann_destroy(void)
{
    kava_endpoint_destroy(&__kava_endpoint);
}

static struct kava_chan *__chan_create(void)
{
    return chan;
}


void __handle_command_genann(struct kava_chan* __chan,
                        const struct kava_cmd_base* __cmd)
{
    // TODO: impl this function
    __chan->cmd_print(__chan, __cmd);

    switch (__cmd->command_id) {
    case RET_GENANN___GENANN_ACT_HIDDEN_INDIRECT:
    {
        struct genann_genann_act_hidden_indirect_ret *__ret = (struct genann_genann_act_hidden_indirect_ret *)__cmd;
        struct genann_genann_act_hidden_indirect_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_genann_act_hidden_indirect_ret) &&
                CMD_ERR_MSG);
        __local = (struct genann_genann_act_hidden_indirect_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {
            /* Output: double ret */
            {
                __local->ret = __ret->ret;
            }
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    case RET_GENANN___GENANN_INIT:
    {
        struct genann_genann_init_ret *__ret = (struct genann_genann_init_ret *)__cmd;
        struct genann_genann_init_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_genann_init_ret) &&
                CMD_ERR_MSG);
        __local = (struct genann_genann_init_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {
            /* Output: genann *ret */
            {
                __local->ret = __ret->ret;
                BUG_ON(__local->ret == NULL);
                if (__local->ret) {
                    Metadata *metadata = kava_metadata(__ret->ret);
                    metadata->inputs = __local->inputs;
                    metadata->outputs = __local->outputs;
                }
            }
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    case RET_GENANN___GENANN_COPY:
    {
        struct genann_genann_copy_ret *__ret = (struct genann_genann_copy_ret *)__cmd;
        struct genann_genann_copy_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_genann_copy_ret) &&
                CMD_ERR_MSG);
        __local = (struct genann_genann_copy_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {
            /* Output: genann *ret */
            {
                __local->ret = __ret->ret;
                BUG_ON(__local->ret == NULL);
                if (__local->ret) {
                    Metadata *metadata = kava_metadata(__local->ret);
                    Metadata *src_metadata = kava_metadata(__local->ann);
                    metadata->inputs = src_metadata->inputs;
                    metadata->outputs = src_metadata->outputs;
                }
            }
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    case RET_GENANN___GENANN_RANDOMIZE:
    {
        struct genann_genann_randomize_ret *__ret = (struct genann_genann_randomize_ret *)__cmd;
        struct genann_genann_randomize_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_genann_randomize_ret) &&
                CMD_ERR_MSG);
        __local = (struct genann_genann_randomize_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {
            /* the content of ann is never accessed inside kernel space */
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    case RET_GENANN___GENANN_FREE:
    {
        struct genann_genann_free_ret *__ret = (struct genann_genann_free_ret *)__cmd;
        struct genann_genann_free_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_genann_free_ret) &&
                CMD_ERR_MSG);
        __local = (struct genann_genann_free_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {
            kava_metadata_remove(__local->ann);
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    case RET_GENANN___GENANN_RUN:
    {
        struct genann_genann_run_ret *__ret = (struct genann_genann_run_ret *)__cmd;
        struct genann_genann_run_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_genann_run_ret) &&
                CMD_ERR_MSG);
        __local = (struct genann_genann_run_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {
            {
                BUG_ON(__ret->ret == NULL);
                if (__ret->ret != NULL) {
                    volatile size_t __buffer_size = kava_metadata(__local->ann)->outputs * sizeof(double);
                    double *__src_ret_0 = (double *)__chan->chan_get_buffer(__chan, __cmd, __ret->ret);
                    __local->ret = (double *)vmalloc(__buffer_size);
                    BUG_ON(__local->ret == NULL);
                    if (__local->ret) {
                        memcpy(__local->ret, __src_ret_0, __buffer_size);
                    }
                }
            }
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    case RET_GENANN___GENANN_TRAIN:
    {
        struct genann_genann_train_ret *__ret = (struct genann_genann_train_ret *)__cmd;
        struct genann_genann_train_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_genann_train_ret) &&
                CMD_ERR_MSG);

        __local = (struct genann_genann_train_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    /***************************************************************************
     * genann utils
     **************************************************************************/

    case RET_GENANN___GET_DATA_SAMPLE_SIZE:
    {
        struct genann_get_data_sample_size_ret *__ret = (struct genann_get_data_sample_size_ret *)__cmd;
        struct genann_get_data_sample_size_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_get_data_sample_size_ret) &&
                CMD_ERR_MSG);

        __local = (struct genann_get_data_sample_size_call_record *)kava_remove_call(
                &__kava_endpoint, __ret->__call_id);
        {
            __local->ret = __ret->ret;
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    case RET_GENANN___READ_TRAINING_DATA:
    {
        struct genann_read_training_data_ret *__ret = (struct genann_read_training_data_ret *)__cmd;
        struct genann_read_training_data_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_read_training_data_ret) &&
                CMD_ERR_MSG);

        __local = (struct genann_read_training_data_call_record *)kava_remove_call(
                &__kava_endpoint, __ret->__call_id);
        {
            __local->ret = __ret->ret;
        }

        {
            /* input and label */
            if (__ret->input != NULL) {
                if (kava_shm_offset(__local->input) >= 0) {
                }
                else {
                    double *__src_input_0 = (double *)__chan->chan_get_buffer(__chan, __cmd, __ret->input);
                    BUG_ON(__ret->input == NULL);
                    if (__ret->input) {
                        memcpy(__local->input, __src_input_0, (size_t)(sizeof(double)
                                    * (__local->inputs) * (__local->samples)));
                    }
                }
            }

            if (__ret->label != NULL) {
                if (kava_shm_offset(__local->label) >= 0) {
                }
                else {
                    double *__src_label_0 = (double *)__chan->chan_get_buffer(__chan, __cmd, __ret->label);
                    BUG_ON(__ret->label == NULL);
                    if (__ret->label) {
                        memcpy(__local->label, __src_label_0, (size_t)(sizeof(double) *
                                    (__local->outputs) * (__local->samples)));
                    }
                }
            }
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }

        break;
    }

    case RET_GENANN___GENANN_HILL_CLIMB:
    {
        struct genann_genann_hill_climb_ret *__ret = (struct genann_genann_hill_climb_ret *)__cmd;
        struct genann_genann_hill_climb_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_genann_hill_climb_ret) &&
                CMD_ERR_MSG);

        __local = (struct genann_genann_hill_climb_call_record *)kava_remove_call(
                &__kava_endpoint, __ret->__call_id);
        {
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    case RET_GENANN___POW:
    {
        struct genann_pow_ret *__ret = (struct genann_pow_ret *)__cmd;
        struct genann_pow_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_pow_ret) &&
                CMD_ERR_MSG);

        __local = (struct genann_pow_call_record *)kava_remove_call(
                &__kava_endpoint, __ret->__call_id);
        {
            __local->ret = __ret->ret;
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    case RET_GENANN___GENANN_READ_FILE:
    {
        struct genann_genann_read_file_ret *__ret = (struct genann_genann_read_file_ret *)__cmd;
        struct genann_genann_read_file_call_record *__local;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct genann_genann_read_file_ret) &&
                CMD_ERR_MSG);
        __local = (struct genann_genann_read_file_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {
            /* Output: genann *ret */
            {
                __local->ret = __ret->ret;
                if (__local->ret) {
                    Metadata *metadata = kava_metadata(__local->ret);
                    metadata->inputs = __ret->ann_inputs;
                    metadata->outputs = __ret->ann_outputs;
                }
            }
        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }

    /***************************************************************************
     * end of genann utils
     **************************************************************************/


    default:
        break;
    }  // switch
}

void __print_command_genann(FILE* file, const struct kava_chan *__chan,
                        const struct kava_cmd_base *__cmd)
{
    switch (__cmd->command_id) {
    // TODO: needs later check of enum and corresponding func
    case RET_GENANN___GENANN_INIT:
        {
            pr_info("genann_init is responded\n");
            break;
        }
    case RET_GENANN___GENANN_COPY:
        {
            pr_info("genann_copy is responded\n");
            break;
        }
    case RET_GENANN___GENANN_TRAIN:
        {
            pr_info("genann_train is responded\n");
            break;
        }
    case RET_GENANN___GENANN_RUN:
        {
            pr_info("genann_run is responded\n");
            break;
        }
    case RET_GENANN___GENANN_RANDOMIZE:
        {
            pr_info("genann_randomize is responded\n");
        }
    case RET_GENANN___GENANN_FREE:
        {
            pr_info("genann_free is responded\n");
            break;
        }
    /***************************************************************************
     * genann utils
     **************************************************************************/

    case RET_GENANN___GET_DATA_SAMPLE_SIZE:
        {
            pr_info("get_data_sample_size is responded\n");
            break;
        }
    case RET_GENANN___READ_TRAINING_DATA:
        {
            pr_info("read_training_data is responded\n");
            break;
        }
    case RET_GENANN___GENANN_HILL_CLIMB:
        {
            pr_info("genann_hill_climb is responded\n");
            break;
        }
    case RET_GENANN___POW:
        {
            pr_info("pow is responded\n");
            break;
        }
    case RET_GENANN___GENANN_READ_FILE:
        {
            pr_info("genann_read_file responded\n");
            break;
        }

    /***************************************************************************
     * end of genann utils
     **************************************************************************/


    default:
        {
            pr_err("Unrecognized genann response: %lu\n", __cmd->command_id);
        }
    }
}

// TODO: might not need to call stub, only return the content inside the struct
SSE
double genann_act_hidden_indirect(const struct genann *ann, double a) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_act_hidden_indirect_call *__cmd;
    struct genann_genann_act_hidden_indirect_call_record *__call_record;
    int64_t __thread_id;
    double ret;

    size_t __total_buffer_size = 0;
    {
        // TODO: check whether to use struct directly
    } 

    __cmd = (struct genann_genann_act_hidden_indirect_call *)chan->cmd_new(chan,
            sizeof(struct genann_genann_act_hidden_indirect_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_ACT_HIDDEN_INDIRECT;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);
    
    __cmd->__call_id = __call_id;
    {
        /* Marshal parameters */
        __cmd->ann = ann;
        __cmd->a = a;
    }

    __call_record = (struct genann_genann_act_hidden_indirect_call_record *)vmalloc(
            sizeof(struct genann_genann_act_hidden_indirect_call_record));
    __call_record->ann = ann;
    __call_record->a = a;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(genann_act_hidden_indirect);

// TODO: possibly don't need to forward this call
SSE
double genann_act_output_indirect(const struct genann *ann, double a) {
    // TODO: forward to userspace to avoid pointer reference
    // return ann->activation_output(ann, a);
    
}
EXPORT_SYMBOL(genann_act_output_indirect);

// TODO: figure out whether stub forward is rly required here
SSE
double genann_act_sigmoid(const genann *ann __attribute__((unused)), double a) {
    // TODO; place holder, might not need forward op
    return a; 
}
EXPORT_SYMBOL(genann_act_sigmoid);

void genann_init_sigmoid_lookup(const genann *ann) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    struct genann_genann_init_sigmoid_lookup_call *__cmd;
    struct genann_genann_init_sigmoid_lookup_call_record *__call_record;
    int64_t __thread_id;
    /* no return val */

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct genann_genann_init_sigmoid_lookup_call *)chan->cmd_new(
            chan, sizeof(struct genann_genann_init_sigmoid_lookup_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_INIT_SIGMOID_LOOKUP;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;
    {
        /* Marshall parameters */
        __cmd->ann = ann;
    }

    __call_record = (struct genann_genann_init_sigmoid_lookup_call_record *) vmalloc(
            sizeof(struct genann_genann_init_sigmoid_lookup_call_record));
    __call_record->ann = ann;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    vfree(__call_record);
}
EXPORT_SYMBOL(genann_init_sigmoid_lookup);

// TODO: might only need in userspace
SSE
double genann_act_sigmoid_cached(const genann *ann, double a) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_act_sigmoid_cached_call *__cmd;
    struct genann_genann_act_sigmoid_cached_call_record *__call_record;
    int64_t __thread_id;
    double ret;

    size_t __total_buffer_size = 0;
    {
        // TODO
    }

    __cmd = (struct genann_genann_act_sigmoid_cached_call *)chan->cmd_new(
            chan, sizeof(struct genann_genann_act_sigmoid_cached_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_ACT_SIGMOID_CACHED;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Marshall arguments */
        __cmd->ann = ann;
        __cmd->a = a;
    }

    __call_record = (struct genann_genann_act_sigmoid_cached_call_record *)vmalloc(
            sizeof(struct genann_genann_act_sigmoid_cached_call_record));
    __call_record->ann = ann;
    __call_record->a = a;
    __call_record->__handler_deallocate = 0;
    __call_record->__call_complete = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(genann_act_sigmoid_cached);

SSE
double genann_act_linear(const struct genann *ann __attribute__((unused)), double a) {
    return a;
}
EXPORT_SYMBOL(genann_act_linear);

SSE
double genann_act_threshold(const struct genann *ann __attribute__((unused)), double a) {
    // TODO: check whether stub forwarding is necessary
    double result;
    kernel_fpu_begin();
    result = (double)(a > 0);
    kernel_fpu_end();
    return result;
}
EXPORT_SYMBOL(genann_act_threshold);

/*
 * genann_init() stub function
 */
genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_init_call *__cmd;
    struct genann_genann_init_call_record *__call_record;
    int64_t __thread_id;
    genann *ret;

    size_t __total_buffer_size = 0;
    {
    }

    __cmd = (struct genann_genann_init_call *)chan->cmd_new(chan, 
            sizeof(struct genann_genann_init_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_INIT;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* marshall inputs */
        {
            __cmd->inputs = inputs;
            __cmd->hidden_layers = hidden_layers;
            __cmd->hidden = hidden;
            __cmd->outputs = outputs;
        }
    }

    __call_record = (struct genann_genann_init_call_record *)vmalloc(
            sizeof(struct genann_genann_init_call_record));

    __call_record->inputs = inputs;
    __call_record->hidden_layers = hidden_layers;
    __call_record->hidden = hidden;
    __call_record->outputs = outputs;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(genann_init);

genann *genann_copy(genann const *ann) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_copy_call *__cmd;
    struct genann_genann_copy_call_record *__call_record;
    int64_t __thread_id;
    genann *ret;

    size_t __total_buffer_size = 0;
    {
    }

    __cmd = (struct genann_genann_copy_call *)chan->cmd_new(
            chan, sizeof(struct genann_genann_copy_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_COPY;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;
    {
        __cmd->ann = ann;
    }

    __call_record = (struct genann_genann_copy_call_record *)vmalloc(
            sizeof(struct genann_genann_copy_call_record));
    __call_record->ann = ann;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);

    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(genann_copy);

void genann_randomize(genann *ann) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_randomize_call *__cmd;
    struct genann_genann_randomize_call_record *__call_record;
    int64_t __thread_id;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct genann_genann_randomize_call *)chan->cmd_new(
            chan, sizeof(struct genann_genann_randomize_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_RANDOMIZE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;
    {
        /* Marshall parameters */
        __cmd->ann = ann;
    }

    /*
    __call_record = (struct genann_genann_randomize_call_record *)vmalloc(
            sizeof(struct genann_genann_randomize_call_record));
    __call_record->ann = ann;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);
    */

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
}
EXPORT_SYMBOL(genann_randomize);

void genann_free(genann *ann) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_free_call *__cmd;
    struct genann_genann_free_call_record *__call_record;
    int64_t __thread_id;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct genann_genann_free_call *)chan->cmd_new(
            chan, sizeof(struct genann_genann_free_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_FREE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: genann *ann */
        {
            __cmd->ann = ann;
        }
    }

    /*
    __call_record = (struct genann_genann_free_call_record *)vmalloc(
            sizeof(struct genann_genann_free_call_record));

    __call_record->ann = ann;
    __call_record->__call_complete  = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);
    */

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
}
EXPORT_SYMBOL(genann_free);

SSE
double const *genann_run(genann const *ann, double const *inputs) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_run_call *__cmd;
    struct genann_genann_run_call_record *__call_record;
    int64_t __thread_id;
    double *ret;

    size_t __total_buffer_size = 0;
    {
        const size_t __input_buf_size = (size_t)(kava_metadata(ann)->inputs) * sizeof(double);
        if ((inputs != NULL) && (__input_buf_size > 0)) {
            if (kava_shm_offset(inputs) >= 0) {
            }
            else {
                __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) __input_buf_size));
            }
        }
    }
    __cmd = (struct genann_genann_run_call *)chan->cmd_new(
            chan, sizeof(struct genann_genann_run_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_RUN;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;
    {
        /* Marshall parameters */
        {
            __cmd->ann = ann;
        }

        {
            const size_t __input_buf_size = (size_t)(kava_metadata(ann)->inputs) * sizeof(double);
            if (inputs && __input_buf_size > 0) {
                if (kava_shm_offset(inputs) >= 0) {
                    __cmd->__shm_inputs = 1;
                    __cmd->inputs = (void *)kava_shm_offset(inputs);
                }
                else {
                    __cmd->__shm_inputs = 0;
                    __cmd->inputs =
                        (double *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd,
                                inputs, __input_buf_size);
                }
            } else {
                __cmd->inputs = NULL;
            }
        }
    }

    __call_record = (struct genann_genann_run_call_record *)vmalloc(
            sizeof(struct genann_genann_run_call_record));
    __call_record->ann = ann;
    __call_record->inputs = inputs;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;

    kava_add_call(&__kava_endpoint, __call_id, __call_record);
    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(genann_run);

void genann_train(genann const *ann, double const *inputs,
        double const *desired_outputs,
        double learning_rate) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_train_call *__cmd;
    struct genann_genann_train_call_record *__call_record;
    int64_t __thread_id;

    size_t __total_buffer_size = 0;
    {
        size_t __buffer_size = sizeof(double) * kava_metadata(ann)->inputs;
        if (inputs && __buffer_size > 0) {
            if (kava_shm_offset(inputs) >= 0) {
            }
            else {
                __total_buffer_size +=
                    chan->chan_buffer_size(chan, __buffer_size);
            }
        }

        __buffer_size = sizeof(double) * kava_metadata(ann)->outputs;
        if (desired_outputs && __buffer_size > 0) {
            if (kava_shm_offset(desired_outputs) >= 0) {
            }
            else {
                __total_buffer_size +=
                    chan->chan_buffer_size(chan, __buffer_size);
            }
        }
    }
    __cmd = (struct genann_genann_train_call *)chan->cmd_new(
            chan, sizeof(struct genann_genann_train_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_TRAIN;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);
    __cmd->__call_id = __call_id;

    {
        {
            /* Input: genann const *ann */
            __cmd->ann = ann;
        }
        {
            /* Input: double const *inputs */
            const size_t __buffer_size = sizeof(double) * kava_metadata(ann)->inputs;
            if (inputs && __buffer_size > 0) {
                if (kava_shm_offset(inputs) >= 0) {
                    __cmd->__shm_inputs = 1;
                    __cmd->inputs = (void *)kava_shm_offset(inputs);
                }
                else {
                    __cmd->__shm_inputs = 0;
                    __cmd->inputs =
                        chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd,
                                inputs, __buffer_size);
                }
            } else {
                __cmd->inputs = NULL;
            }
        }
        {
            /* Input: double const *desired_outputs */
            const size_t __buffer_size = sizeof(double) * kava_metadata(ann)->outputs;
            if (desired_outputs && __buffer_size > 0) {
                if (kava_shm_offset(desired_outputs) >= 0) {
                    __cmd->__shm_desired_outputs = 1;
                    __cmd->desired_outputs = (void *)kava_shm_offset(desired_outputs);
                }
                else {
                    __cmd->__shm_desired_outputs = 0;
                    __cmd->desired_outputs =
                        chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd,
                                desired_outputs, __buffer_size);
                }
            } else {
                __cmd->desired_outputs = NULL;
            }
        }
        {
            __cmd->learning_rate = learning_rate;
        }
    }

    /*
    __call_record = (struct genann_genann_train_call_record *)vmalloc(
            sizeof(struct genann_genann_train_call_record));
    __call_record->ann = ann;
    __call_record->inputs = inputs;
    __call_record->desired_outputs = desired_outputs;
    __call_record->learning_rate = learning_rate;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);
    */

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
}
EXPORT_SYMBOL(genann_train);

/*******************************************************************************
 * genann utility functions
 ******************************************************************************/

int get_data_sample_size(const char *file) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_get_data_sample_size_call *__cmd;
    struct genann_get_data_sample_size_call_record *__call_record;
    int64_t __thread_id;
    int ret;

    size_t __total_buffer_size = 0;
    {
        BUG_ON(file == NULL);
        if (file) {
            __total_buffer_size +=
                chan->chan_buffer_size(chan, (size_t)(strlen(file) + 1) * (sizeof(const char)));
        }
    }

    __cmd = (struct genann_get_data_sample_size_call *)chan->cmd_new(
            chan, sizeof(struct genann_get_data_sample_size_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GET_DATA_SAMPLE_SIZE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);
    __cmd->__call_id = __call_id;

    {
        /* Attach file path to buffer */
        if (file) {
            __cmd->file =
                (char *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd,
                        file, (size_t)(strlen(file) + 1) * sizeof(const char));
        }
    }

    __call_record = (struct genann_get_data_sample_size_call_record *)vmalloc(
            sizeof(struct genann_get_data_sample_size_call_record));
    __call_record->file = file;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(get_data_sample_size);

int read_training_data(double *input, double *label, const int inputs,
        const int outputs, char *file, char **class_name, const size_t *lengths,
        const int samples) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_read_training_data_call *__cmd;
    struct genann_read_training_data_call_record *__call_record;
    int64_t __thread_id;
    int ret;

    struct kava_buffer_list *__kava_alloc_list_read_training_data =
        kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* size: char *file */
        BUG_ON(file == NULL);
        if (file) {
            __total_buffer_size +=
                chan->chan_buffer_size(chan, (size_t)(sizeof(const char) * 
                           (strlen(file) + 1)));
        }

        /* size: const size_t *lengths */
        {
            if (lengths != NULL && outputs > 0) {
                __total_buffer_size += chan->chan_buffer_size(chan,
                        (size_t)(outputs) * sizeof(const size_t));
            }
        }

        /* size: char *class_name */
        BUG_ON(class_name == NULL);
        if (class_name && outputs > 0) {
            const size_t __class_name_size_0 = (size_t)outputs;
            size_t __class_name_index_0;
            for (__class_name_index_0 = 0; __class_name_index_0 < __class_name_size_0;
                    __class_name_index_0++) {
                const size_t kava_index = __class_name_index_0;
                char **__class_name_a_0 = (char **)(class_name) + __class_name_index_0;
                size_t __buffer_size = ((lengths != NULL && lengths[kava_index]) ?
                    (lengths[kava_index]) : (strlen(class_name[kava_index]) + 1)) * sizeof(char);

                if ((*__class_name_a_0) != NULL && outputs > 0 && __buffer_size > (0)) {
                    __total_buffer_size += chan->chan_buffer_size(
                            chan, __buffer_size);
                }
            }
            __total_buffer_size += chan->chan_buffer_size(
                    chan, (size_t)(outputs * sizeof(char *)));
        }
    }
    __cmd = (struct genann_read_training_data_call *)chan->cmd_new(
            chan, sizeof(struct genann_read_training_data_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___READ_TRAINING_DATA;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);
    __cmd->__call_id = __call_id;

    {
        {
            /* Input: double *input */
            if (input) {
                if (kava_shm_offset(input) >= 0) {
                    __cmd->__shm_input = 1;
                    __cmd->input = (void *)kava_shm_offset(input);
                }
                else {
                    __cmd->__shm_input = 0;
                    __cmd->input = HAS_OUT_BUFFER_SENTINEL;
                }
            } else {
                __cmd->input = NULL;
            }
        }
        {
            /* Input: double *label */
            if (label) {
                if (kava_shm_offset(label) >= 0) {
                    __cmd->__shm_label = 1;
                    __cmd->label = (void *)kava_shm_offset(label);
                }
                else {
                    __cmd->__shm_label = 0;
                    __cmd->label = HAS_OUT_BUFFER_SENTINEL;
                }
            } else {
                __cmd->label = NULL;
            }
        }
        {
            /* Input: int inputs */
            __cmd->inputs = inputs;
        }
        {
            /* Input: int outputs */
            __cmd->outputs = outputs;
        }
        {
            /* Input: char *file */
            BUG_ON(file == NULL);
            if (file) {
                __cmd->file = (char *)chan->chan_attach_buffer(chan,
                        (struct kava_cmd_base *)__cmd, file,
                        (size_t)(strlen(file) + 1) * sizeof(const char));
            } else {
                __cmd->file = NULL;
            }
        }
        {
            /* Input: const size_t *lengths */
            if (lengths != NULL && outputs > 0) {
                __cmd->lengths = (size_t *)chan->chan_attach_buffer(chan,
                        (struct kava_cmd_base *)__cmd, lengths,
                        (size_t)(outputs) * sizeof(const size_t));
            } else {
                __cmd->lengths = NULL;
            }
        }
        {
            /* Input: char **class_name */
            if (class_name != NULL && outputs > 0) {
                const size_t __size_class_name_0 = (size_t)outputs;
                char **__tmp_class_name_0 = (char **)vmalloc(__size_class_name_0 * sizeof(char *));
                kava_endpoint_buffer_list_add(__kava_alloc_list_read_training_data,
                        kava_buffer_with_deallocator_new(vfree, (void *)__tmp_class_name_0));

                const size_t __class_name_size_0 = __size_class_name_0;
                size_t __class_name_index_0;
                for (__class_name_index_0 = 0; __class_name_index_0 < __class_name_size_0;
                        __class_name_index_0++) {
                    const size_t kava_index = __class_name_index_0;

                    char **__class_name_a_0 = (char **)class_name + __class_name_index_0;
                    char **__class_name_b_0 = (char **)(__tmp_class_name_0 + __class_name_index_0);

                    {
                        if ((*__class_name_a_0) != NULL && outputs > 0 &&
                                ((lengths != NULL && lengths[kava_index]) ? (lengths[kava_index]) : 
                                 (strlen(class_name[kava_index]) + 1)) > (0)) {
                            *__class_name_b_0 = (char *)chan->chan_attach_buffer(
                                    chan, (struct kava_cmd_base *)__cmd,
                                    *__class_name_a_0, (size_t)((lengths != NULL && lengths[kava_index]) ?
                                    (lengths[kava_index]) : (strlen(class_name[kava_index]) + 1)) *
                                    sizeof(char));
                        } else {
                            *__class_name_b_0 = NULL;
                        }
                    }
                }
                __cmd->class_name = (char **)chan->chan_attach_buffer(
                        chan, (struct kava_cmd_base *)__cmd, __tmp_class_name_0,
                        (size_t)outputs * sizeof(char *));
            } else {
                __cmd->class_name = NULL;
            }
        }
        {
            /* Input: samples */
            __cmd->samples = samples;
        }
    }

    __call_record = (struct genann_read_training_data_call_record *)vmalloc(
            sizeof(struct genann_read_training_data_call_record));
    __call_record->input = input;
    __call_record->label = label;
    __call_record->inputs = inputs;
    __call_record->outputs = outputs;
    __call_record->file = file;
    __call_record->class_name = class_name;
    __call_record->lengths= lengths;
    __call_record->samples = samples;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    kava_endpoint_buffer_list_free(__kava_alloc_list_read_training_data);
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(read_training_data);

void genann_hill_climb(genann const *ann, const double rate) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_hill_climb_call *__cmd;
    struct genann_genann_hill_climb_call_record *__call_record;
    int64_t __thread_id;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct genann_genann_hill_climb_call *)chan->cmd_new(
            chan, sizeof(struct genann_genann_hill_climb_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_HILL_CLIMB;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;
    {
        {
            __cmd->ann = ann;
        }

        {
            __cmd->rate = rate;
        }
    }

    __call_record = (struct genann_genann_hill_climb_call_record *)vmalloc(
            sizeof(struct genann_genann_hill_climb_call_record));
    __call_record->ann = ann;
    __call_record->rate = rate;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
}
EXPORT_SYMBOL(genann_hill_climb);

SSE
double pow(double x, double y) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_pow_call *__cmd;
    struct genann_pow_call_record *__call_record;
    int64_t __thread_id;
    double ret; 

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct genann_pow_call *)chan->cmd_new(
            chan, sizeof(struct genann_pow_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___POW;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;
    {
        __cmd->x = x;
        __cmd->y = y;
    }

    __call_record = (struct genann_pow_call_record *)vmalloc(
            sizeof(struct genann_pow_call_record));
    __call_record->x = x;
    __call_record->y = y;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(pow);

genann *genann_read_file(const char *file) {
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct genann_genann_read_file_call *__cmd;
    struct genann_genann_read_file_call_record *__call_record;
    int64_t __thread_id;
    genann *ret;

    size_t __total_buffer_size = 0;
    {
        BUG_ON(file == NULL);
        if (file) {
            __total_buffer_size +=
                chan->chan_buffer_size(chan, (size_t)(strlen(file) + 1) * (sizeof(const char)));
        }
    }

    __cmd = (struct genann_genann_read_file_call *)chan->cmd_new(
            chan, sizeof(struct genann_genann_read_file_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_GENANN___GENANN_READ_FILE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);
    __cmd->__call_id = __call_id;

    {
        /* Attach file path to buffer */
        if (file) {
            __cmd->file =
                (char *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd,
                        file, (size_t)(strlen(file) + 1) * sizeof(const char));
        }
    }

    __call_record = (struct genann_genann_read_file_call_record *)vmalloc(
            sizeof(struct genann_genann_read_file_call_record));
    __call_record->file = file;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(genann_read_file);

/*******************************************************************************
 * end of genann utility functions
 ******************************************************************************/


static int __init kgenann_init(void)
{
    kava_register_cmd_handler(KAVA_CMD_MODE_API,
                            NULL,
                            NULL);
    pr_info("Create control device\n");
    init_ctrl_if();
    pr_info("Load genann kernel library\n");
    init_global_kapi(KAVA_API_ID_GENANN, chan_mode);

    /* Initialize endpoint */
    init_endpoint_lib();
    __handle_command_genann_init();

    /* Create channel */
    switch (global_kapi->chan_mode) {
        case KAVA_CHAN_NL_SOCKET:
            chan = kava_chan_nl_socket_new();
            break;

        case KAVA_CHAN_FILE_POLL:
        default:
            chan = kava_chan_file_poll_new();
    }
    kava_init_cmd_handler(__chan_create);
    kava_init_internal_cmd_handler();

    pr_info("Register channel in control interface\n");
    return ctrl_if_register_chan(chan);
}

static void __exit kgenann_fini(void)
{
    pr_info("Stop running worker\n");
    stop_worker(chan);

    pr_info("Destroy endpoint\n");
    __handle_command_genann_destroy();

    pr_info("Unload genann kernel library\n");
    if (chan) {
        chan->chan_free(chan);
    }
    put_global_kapi();
    fini_ctrl_if();
}

module_init(kgenann_init);
module_exit(kgenann_fini);

MODULE_AUTHOR("Bodun Hu");
MODULE_DESCRIPTION("genann kernel library");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
