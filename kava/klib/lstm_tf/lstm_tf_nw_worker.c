
#define __KAVA__ 1
#define kava_is_worker 1
#define kava_is_guest 0

#include <worker.h>

#undef AVA_BENCHMARKING_MIGRATE

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
    worker_common_init();
    __handle_command_lstm_tf_init();
    ;
}

////// API function stub implementations

#define __chan nw_global_command_channel

static int
__wrapper_load_model(const char *file)
{
    {
        int ret;
        ret = load_model(file);

        return ret;
    }
}

static int
__wrapper_kleio_load_model(const char *file)
{
    {
        int ret;
        ret = kleio_load_model(file);

        return ret;
    }
}


static void
__wrapper_close_ctx()
{
    {

        close_ctx();

        return;
    }
}

static int
__wrapper_standard_inference(unsigned int num_syscall, unsigned int sliding_window, const void *syscalls)
{
    {
        int ret;
        ret = standard_inference(syscalls, num_syscall, sliding_window);

        return ret;
    }
}

static int
__wrapper_kleio_inference(unsigned int num_syscall, unsigned int sliding_window, const void *syscalls)
{
    {
        int ret;
        ret = kleio_inference(syscalls, num_syscall, sliding_window);

        return ret;
    }
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
    //__chan->cmd_print(__chan, __cmd);
    switch (__cmd->command_id) {

    case CALL_LSTM_TF_LOAD_MODEL:{
        GPtrArray *__kava_alloc_list_load_model =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct lstm_tf_load_model_call *__call = (struct lstm_tf_load_model_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct lstm_tf_load_model_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: const char * file */
        char *file; {
            file =
                ((__call->file) != (NULL)) ? ((const char *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->file)) : ((const char *)__call->file);
            if ((__call->file) != (NULL)) {
                char *__src_file_0;
                __src_file_0 = file;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (strlen(file) + 1));
                file = (const char *)__chan->chan_get_buffer(__chan, __cmd, __call->file);

                if ((file) != (__src_file_0)) {
                    memcpy(file, __src_file_0, __buffer_size * sizeof(const char));
                }
            } else {
                file =
                    ((__call->file) != (NULL)) ? ((const char *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->file)) : ((const char *)__call->file);
        }}

        /* Perform Call */

        int ret;
        ret = __wrapper_load_model(file);

        size_t __total_buffer_size = 0;
        {
        }
        struct lstm_tf_load_model_ret *__ret =
            (struct lstm_tf_load_model_ret *)__chan->cmd_new(__chan, sizeof(struct lstm_tf_load_model_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_LSTM_TF_LOAD_MODEL;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: int ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_load_model);        /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_LSTM_TF_KLEIO_LOAD_MODEL:{
        GPtrArray *__kava_alloc_list_kleio_load_model =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct lstm_tf_kleio_load_model_call *__call = (struct lstm_tf_kleio_load_model_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct lstm_tf_kleio_load_model_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: const char * file */
        char *file; {
            file =
                ((__call->file) != (NULL)) ? ((const char *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->file)) : ((const char *)__call->file);
            if ((__call->file) != (NULL)) {
                char *__src_file_0;
                __src_file_0 = file;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (strlen(file) + 1));
                file = (const char *)__chan->chan_get_buffer(__chan, __cmd, __call->file);

                if ((file) != (__src_file_0)) {
                    memcpy(file, __src_file_0, __buffer_size * sizeof(const char));
                }
            } else {
                file =
                    ((__call->file) != (NULL)) ? ((const char *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->file)) : ((const char *)__call->file);
        }}

        /* Perform Call */

        int ret;
        ret = __wrapper_kleio_load_model(file);

        size_t __total_buffer_size = 0;
        {
        }
        struct lstm_tf_kleio_load_model_ret *__ret =
            (struct lstm_tf_kleio_load_model_ret *)__chan->cmd_new(__chan, sizeof(struct lstm_tf_kleio_load_model_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_LSTM_TF_KLEIO_LOAD_MODEL;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: int ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_kleio_load_model);        /* Deallocate all memory in the alloc list */

        break;
    }

    case CALL_LSTM_TF_CLOSE_CTX:{
        GPtrArray *__kava_alloc_list_close_ctx =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct lstm_tf_close_ctx_call *__call = (struct lstm_tf_close_ctx_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct lstm_tf_close_ctx_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Perform Call */

        __wrapper_close_ctx();

        size_t __total_buffer_size = 0;
        {
        }
        struct lstm_tf_close_ctx_ret *__ret =
            (struct lstm_tf_close_ctx_ret *)__chan->cmd_new(__chan, sizeof(struct lstm_tf_close_ctx_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_LSTM_TF_CLOSE_CTX;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_close_ctx); /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_LSTM_TF_STANDARD_INFERENCE:{
        GPtrArray *__kava_alloc_list_standard_inference =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct lstm_tf_standard_inference_call *__call = (struct lstm_tf_standard_inference_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct lstm_tf_standard_inference_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: unsigned int num_syscall */
        unsigned int num_syscall; {
            num_syscall = (unsigned int)__call->num_syscall;
            num_syscall = __call->num_syscall;
        }

        /* Input: unsigned int sliding_window */
        unsigned int sliding_window; {
            sliding_window = (unsigned int)__call->sliding_window;
            sliding_window = __call->sliding_window;
        }

        /* Input: const void * syscalls */
        void *syscalls; 
        {
            syscalls =
                ((__call->syscalls) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->syscalls)) : ((const void *)__call->syscalls);
            if ((__call->syscalls) != (NULL)) {
                void *__src_syscalls_0;
                __src_syscalls_0 = syscalls;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (num_syscall*sizeof(int)));
                syscalls = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->syscalls);

                if ((syscalls) != (__src_syscalls_0)) {
                    memcpy(syscalls, __src_syscalls_0, __buffer_size * sizeof(int));
                }
            } else {
                syscalls =
                    ((__call->syscalls) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->syscalls)) : ((const void *)__call->syscalls);
            }
        }

        /* Perform Call */

        int ret;
        ret = __wrapper_standard_inference(num_syscall, sliding_window, syscalls);

        size_t __total_buffer_size = 0;
        {
            
        }
        struct lstm_tf_standard_inference_ret *__ret =
            (struct lstm_tf_standard_inference_ret *)__chan->cmd_new(__chan,
            sizeof(struct lstm_tf_standard_inference_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_LSTM_TF_STANDARD_INFERENCE;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: int ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_standard_inference);        /* Deallocate all memory in the alloc list */

        break;
    }

    case CALL_LSTM_TF_KLEIO_INFERENCE:{
        GPtrArray *__kava_alloc_list_kleio_inference =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct lstm_tf_kleio_inference_call *__call = (struct lstm_tf_kleio_inference_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct lstm_tf_kleio_inference_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: unsigned int num_syscall */
        unsigned int num_syscall; {
            num_syscall = (unsigned int)__call->num_syscall;
            num_syscall = __call->num_syscall;
        }

        /* Input: unsigned int sliding_window */
        unsigned int sliding_window; {
            sliding_window = (unsigned int)__call->sliding_window;
            sliding_window = __call->sliding_window;
        }

        /* Input: const void * syscalls */
        void *syscalls; 
        {
            syscalls =
                ((__call->syscalls) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->syscalls)) : ((const void *)__call->syscalls);
            if ((__call->syscalls) != (NULL)) {
                void *__src_syscalls_0;
                __src_syscalls_0 = syscalls;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (num_syscall*sizeof(int)));
                syscalls = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->syscalls);

                if ((syscalls) != (__src_syscalls_0)) {
                    memcpy(syscalls, __src_syscalls_0, __buffer_size * sizeof(int));
                }
            } else {
                syscalls =
                    ((__call->syscalls) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->syscalls)) : ((const void *)__call->syscalls);
            }
        }

        /* Perform Call */

        int ret;
        ret = __wrapper_kleio_inference(num_syscall, sliding_window, syscalls);

        size_t __total_buffer_size = 0;
        {
            
        }
        struct lstm_tf_kleio_inference_ret *__ret =
            (struct lstm_tf_kleio_inference_ret *)__chan->cmd_new(__chan,
            sizeof(struct lstm_tf_kleio_inference_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_LSTM_TF_KLEIO_INFERENCE;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: int ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_kleio_inference);        /* Deallocate all memory in the alloc list */

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
