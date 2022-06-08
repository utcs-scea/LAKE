#define __KAVA__ 1

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#define kava_is_worker 0
#define kava_is_guest 1

#ifdef __KERNEL__
#include "api.h"
#include "channel_kern.h"
#include "control.h"
#include "command.h"
#include "command_handler.h"
#include "endpoint.h"
#include "shadow_thread_pool.h"
#include "shared_memory.h"
#else
#include <worker.h>
#include "worker.h"
#include "api.h"
#include "channel.h"
#include "command_handler.h"
#include "debug.h"
#include "endpoint.h"
#endif

// Must be included before lstm_tf_nw.h, so that API
// functions are declared properly.
#include <lstm_tf.h>
#include "../klib/lstm_tf/lstm_tf_nw.h"

#pragma GCC diagnostic ignored "-Wunused-function"

#ifdef __KERNEL__
static char *chan_mode = "netlink_socket";
module_param(chan_mode, charp, 0000);
MODULE_PARM_DESC(chan_mode, "kLSTM_TF channel mode. Default netlink_socket.");

static struct kava_chan *chan;
#endif

static struct kava_endpoint __kava_endpoint;

static void __handle_command_lstm_tf_init(void);
static void __handle_command_lstm_tf_destroy(void);
//void __replay_command_lstm_tf(struct kava_chan* __chan, struct kava_handle_pool* handle_pool,
//                                    struct kava_chan* __log, 
//                                    const struct kava_cmd_base* __call_cmd, const struct kava_cmd_base* __ret_cmd);
void __handle_command_lstm_tf(struct kava_chan *__chan, const struct kava_cmd_base *__cmd);
void __print_command_lstm_tf(FILE * file, const struct kava_chan *__chan, const struct kava_cmd_base *__cmd);

#define kava_metadata(p) (&((struct lstm_tf_metadata*)kava_internal_metadata(&__kava_endpoint, p))->application)

void
enable_constructor(void)
{
}

#include "lstm_tf_nw_utilities.h"

#ifndef NWCC_DEBUG
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
#pragma GCC diagnostic ignored "-Wdiscarded-qualifiers"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wdiscarded-array-qualifiers"
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
#pragma GCC diagnostic ignored "-Waddress"
#endif
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"

void __attribute__ ((constructor(1))) init_lstm_tf_worker(void)
{
    __handle_command_lstm_tf_init();
}

void __attribute__ ((destructor)) destroy_lstm_tf_worker(void)
{
    __handle_command_lstm_tf_destroy();
}

static struct kava_chan *
__chan_create(void)
{
    return chan;
}

void
__handle_command_lstm_tf_init(void)
{
    kava_endpoint_init(&__kava_endpoint, sizeof(struct lstm_tf_metadata));
    kava_register_cmd_handler(KAVA_CMD_MODE_API, __handle_command_lstm_tf, __print_command_lstm_tf);
}

void
__handle_command_lstm_tf_destroy(void)
{
    kava_endpoint_destroy(&__kava_endpoint);
}

void
__handle_command_lstm_tf(struct kava_chan *__chan, const struct kava_cmd_base *__cmd)
{
    __chan->cmd_print(__chan, __cmd);
    switch (__cmd->command_id) {
    case RET_LSTM_TF_LOAD_MODEL:{
        struct lstm_tf_load_model_ret *__ret = (struct lstm_tf_load_model_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct lstm_tf_load_model_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct lstm_tf_load_model_call_record *__local =
            (struct lstm_tf_load_model_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            char *file;
            file = __local->file;

            int ret;
            ret = (int)__ret->ret;

            /* Output: int ret */
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
    case RET_LSTM_TF_CLOSE_CTX:{
        struct lstm_tf_close_ctx_ret *__ret = (struct lstm_tf_close_ctx_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct lstm_tf_close_ctx_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct lstm_tf_close_ctx_call_record *__local =
            (struct lstm_tf_close_ctx_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

        }

        __local->__call_complete = 1;
        if (__local->__handler_deallocate) {
            vfree(__local);
        }
        break;
    }
    case RET_LSTM_TF_STANDARD_INFERENCE:{
        struct lstm_tf_standard_inference_ret *__ret = (struct lstm_tf_standard_inference_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct lstm_tf_standard_inference_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct lstm_tf_standard_inference_call_record *__local =
            (struct lstm_tf_standard_inference_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            unsigned int num_syscall;
            num_syscall = __local->num_syscall;

            unsigned int sliding_window;
            sliding_window = __local->sliding_window;

            void *syscalls;
            syscalls = __local->syscalls;

            int ret;
            ret = (int)__ret->ret;

            /* Output: int ret */
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

    default:
        pr_err("Received unsupported command");
    }                                            // switch
}

void
__print_command_lstm_tf(FILE * file, const struct kava_chan *__chan, const struct kava_cmd_base *__cmd)
{
    switch (__cmd->command_id) {
    case CALL_LSTM_TF_LOAD_MODEL:{
        pr_info("load_model is called \n");
        break;
    }
    case RET_LSTM_TF_LOAD_MODEL:{
        pr_info("load_model is responded\n");
        break;
    }
    case CALL_LSTM_TF_CLOSE_CTX:{
        pr_info("close_ctx is called \n");
        break;
    }
    case RET_LSTM_TF_CLOSE_CTX:{
        pr_info("close_ctx is responded\n");
        break;
    }
    case CALL_LSTM_TF_STANDARD_INFERENCE:{
        pr_info("standard_inference is called \n");
        break;
    }
    case RET_LSTM_TF_STANDARD_INFERENCE:{
        pr_info("standard_inference is responded\n");
        break;
    }
    default:
        pr_err("Received unsupported command");
    }                                            // switch
}

////// API function stub implementations

#define __chan nw_global_command_channel

EXPORTED int
load_model(const char *file)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_load_model = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const char * file */
        if ((file) != (NULL) && (strlen(file) + 1) > (0)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (strlen(file) + 1)) * sizeof(const char));
        }
    }
    struct lstm_tf_load_model_call *__cmd =
        (struct lstm_tf_load_model_call *)chan->cmd_new(chan, sizeof(struct lstm_tf_load_model_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_LSTM_TF_LOAD_MODEL;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: const char * file */
        {
            if ((file) != (NULL) && (strlen(file) + 1) > (0)) {
                __cmd->file =
                    (char *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, file,
                    ((size_t) (strlen(file) + 1)) * sizeof(const char));
            } else {
                __cmd->file = NULL;
            }
        }
    }

    struct lstm_tf_load_model_call_record *__call_record =
        (struct lstm_tf_load_model_call_record *)vmalloc(sizeof(struct lstm_tf_load_model_call_record));

    __call_record->file = file;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_load_model);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    int ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(load_model);

EXPORTED void
close_ctx()
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_close_ctx = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
    }
    struct lstm_tf_close_ctx_call *__cmd =
        (struct lstm_tf_close_ctx_call *)chan->cmd_new(chan, sizeof(struct lstm_tf_close_ctx_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_LSTM_TF_CLOSE_CTX;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

    }

    struct lstm_tf_close_ctx_call_record *__call_record =
        (struct lstm_tf_close_ctx_call_record *)vmalloc(sizeof(struct lstm_tf_close_ctx_call_record));

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_close_ctx);

    return;
}

EXPORT_SYMBOL(close_ctx);

EXPORTED int
standard_inference(const void *syscalls, unsigned int num_syscall, unsigned int sliding_window)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_standard_inference = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * syscalls */
        if ((syscalls) != (NULL) && (num_syscall) > (0)) {
            // if (kava_shm_offset(syscalls) >= 0) {
            // } else {
            //     __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (num_syscall)) * sizeof(const void));
            // }
             __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (num_syscall)) * sizeof(int));
        }
    }
    struct lstm_tf_standard_inference_call *__cmd =
        (struct lstm_tf_standard_inference_call *)chan->cmd_new(chan, sizeof(struct lstm_tf_standard_inference_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_LSTM_TF_STANDARD_INFERENCE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: const void * syscalls */
        {
            if ((syscalls) != (NULL) && (num_syscall) > (0)) {
            //     if (kava_shm_offset(syscalls) >= 0) {
            //         __cmd->syscalls = (void *)kava_shm_offset(syscalls);
            //         __cmd->syscalls = 1;
            //     } else {
            //         __cmd->syscalls = 0;

            //         __cmd->syscalls =
            //             (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, syscalls,
            //             ((size_t) (num_syscall)) * sizeof(const void));
            //     }
            // } else {
                __cmd->syscalls =
                    (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, syscalls,
                    ((size_t) (num_syscall)) * sizeof(int));
            } else {
                __cmd->syscalls = NULL;
            }
        }
        /* Input: unsigned int num_syscall */
        {
            __cmd->num_syscall = num_syscall;
        }
        /* Input: unsigned int sliding_window */
        {
            __cmd->sliding_window = sliding_window;
        }
    }

    struct lstm_tf_standard_inference_call_record *__call_record =
        (struct lstm_tf_standard_inference_call_record *)vmalloc(sizeof(struct lstm_tf_standard_inference_call_record));

    __call_record->num_syscall = num_syscall;

    __call_record->sliding_window = sliding_window;

    __call_record->syscalls = syscalls;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_standard_inference);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    int ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(standard_inference);

/// Kernel initialization

static int __init
lstm_tf_init(void)
{
    kava_register_cmd_handler(KAVA_CMD_MODE_API, NULL, NULL);
    pr_info("Create control device\n");
    init_ctrl_if();
    pr_info("Load lstm_tf kernel library\n");
    init_global_kapi(KAVA_API_ID_LSTM_TF, chan_mode);

    /* Initialize endpoint */
    init_endpoint_lib();
    __handle_command_lstm_tf_init();

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

static void __exit
lstm_tf_fini(void)
{
    pr_info("Stop running worker\n");
    stop_worker(chan);

    pr_info("Destroy endpoint\n");
    __handle_command_lstm_tf_destroy();

    pr_info("Unload lstm_tf kernel library\n");
    if (chan)
        chan->chan_free(chan);
    put_global_kapi();
    fini_ctrl_if();
}

module_init(lstm_tf_init);
module_exit(lstm_tf_fini);

////// Necessary for some kernel functions to be exported

MODULE_AUTHOR("Guess Who");
MODULE_DESCRIPTION("LSTM_TF kernel library");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "." __stringify(0) "." __stringify(0) "." "0");

////// Replacement declarations

#define ava_begin_replacement
#define ava_end_replacement
