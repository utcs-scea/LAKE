/*******************************************************************************

  Kernel-space CUDA API library.

*******************************************************************************/

#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#define kava_is_worker 0

#include "api.h"
#include "channel_kern.h"
#include "command.h"
#include "command_handler.h"
#include "control.h"
#include "endpoint.h"
#include "shadow_thread_pool.h"
#include "shared_memory.h"

#include "cuda_kava.h"
#include "cuda_kava_utilities.h"

static char *chan_mode = "netlink_socket";
module_param(chan_mode, charp, 0000);
MODULE_PARM_DESC(chan_mode, "kCUDA channel mode. Default netlink_socket.");

static struct kava_chan *chan;
static struct kava_endpoint __kava_endpoint;

typedef struct {
    /* argument types */
    int func_argc;
    char func_arg_is_handle[64];
    size_t func_arg_size[64];
} Metadata;

struct cu_metadata {
    struct kava_metadata_base base;
    Metadata application;
};

static void __handle_command_cuda_init(void);
static void __handle_command_cuda_destroy(void);
void __handle_command_cuda(struct kava_chan *__chan, const struct kava_cmd_base* __cmd);
void __print_command_cuda(FILE* file, const struct kava_chan *__chan,
                        const struct kava_cmd_base* __cmd);

#define kava_metadata(p) (&((struct cu_metadata *)kava_internal_metadata(&__kava_endpoint, p))->application)

void __attribute__((constructor)) init_cuda_worker(void) {
    __handle_command_cuda_init();
}

void __handle_command_cuda_init()
{
    kava_endpoint_init(&__kava_endpoint, sizeof(struct cu_metadata));
    kava_register_cmd_handler(KAVA_CMD_MODE_API,
                            __handle_command_cuda,
                            __print_command_cuda);
}

void __handle_command_cuda_destroy(void)
{
    kava_endpoint_destroy(&__kava_endpoint);
}

static struct kava_chan *__chan_create(void)
{
    return chan;
}

void __handle_command_cuda(struct kava_chan* __chan,
                        const struct kava_cmd_base* __cmd)
{
    __chan->cmd_print(__chan, __cmd);

    switch (__cmd->command_id) {
        case RET_CUDA___CUDA_GET_GPU_UTILIZATION_RATES:
        {
            struct cu_cu_get_gpu_utilization_rates_ret *__ret = (struct cu_cu_get_gpu_utilization_rates_ret *)__cmd;
            struct cu_cu_get_gpu_utilization_rates_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_get_gpu_utilization_rates_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_get_gpu_utilization_rates_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CUDA_TEST_CHANNEL:
        {
            struct cu_cu_test_channel_ret *__ret = (struct cu_cu_test_channel_ret *)__cmd;
            struct cu_cu_test_channel_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_test_channel_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_test_channel_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            __local->__call_complete = 1;
            if (__local->__handler_deallocate) {
                vfree(__local);
            }
            break;
        }

        case RET_CUDA___CU_INIT:
        {
            struct cu_cu_init_ret *__ret = (struct cu_cu_init_ret *)__cmd;
            struct cu_cu_init_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_init_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_init_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_DEVICE_GET:
        {
            struct cu_cu_device_get_ret *__ret = (struct cu_cu_device_get_ret *)__cmd;
            struct cu_cu_device_get_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_device_get_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_device_get_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUdevice * device */
                {
                    if ((__ret->device) != (NULL)) {
                        CUdevice *__src_device_0;
                        __src_device_0 = (CUdevice *)__chan->chan_get_buffer(__chan, __cmd, __ret->device);
                        BUG_ON(__local->device == NULL);
                        if (__local->device != NULL) {
                            memcpy(__local->device, __src_device_0, sizeof(CUdevice));
                        }
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_CTX_CREATE_V2:
        {
            struct cu_cu_ctx_create_v2_ret *__ret = (struct cu_cu_ctx_create_v2_ret *)__cmd;
            struct cu_cu_ctx_create_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_ctx_create_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_ctx_create_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUcontext * pctx */
                {
                    if ((__ret->pctx) != (NULL)) {
                        volatile size_t __buffer_size = ((size_t) (1));
                        CUcontext *__src_pctx_0 = (CUcontext *)__chan->chan_get_buffer(__chan, __cmd, __ret->pctx);
                        const size_t __pctx_size_0 = __buffer_size;
                        size_t __pctx_index_0;
                        for (__pctx_index_0 = 0; __pctx_index_0 < __pctx_size_0; __pctx_index_0++) {
                            CUcontext *__pctx_a_0 = (CUcontext *) (__local->pctx) + __pctx_index_0;
                            CUcontext *__pctx_b_0 = (CUcontext *) (__src_pctx_0) + __pctx_index_0;

                            {
                                BUG_ON(__pctx_a_0 == NULL || __pctx_b_0 == NULL);
                                if (__pctx_a_0 && __pctx_b_0)
                                    *__pctx_a_0 = *__pctx_b_0;
                            }
                        }
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MODULE_LOAD:
        {
            struct cu_cu_module_load_ret *__ret = (struct cu_cu_module_load_ret *)__cmd;
            struct cu_cu_module_load_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_module_load_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_module_load_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUmodule * module */
                {
                    if ((__ret->module) != (NULL)) {
                        volatile size_t __buffer_size = ((size_t) (1));
                        CUmodule *__src_module_0 = (CUmodule *)__chan->chan_get_buffer(__chan, __cmd, __ret->module);
                        const size_t __module_size_0 = __buffer_size;
                        size_t __module_index_0;
                        for (__module_index_0 = 0; __module_index_0 < __module_size_0; __module_index_0++) {
                            CUmodule *__module_a_0 = (CUmodule *) (__local->module) + __module_index_0;
                            CUmodule *__module_b_0 = (CUmodule *) (__src_module_0) + __module_index_0;

                            {
                                BUG_ON(__module_a_0 == NULL || __module_b_0 == NULL);
                                if (__module_a_0 && __module_b_0)
                                    *__module_a_0 = *__module_b_0;
                            }
                        }
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MODULE_UNLOAD:
        {
            struct cu_cu_module_unload_ret *__ret = (struct cu_cu_module_unload_ret *)__cmd;
            struct cu_cu_module_unload_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_module_unload_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_module_unload_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                CUmodule hmod;
                CUresult ret;

                hmod = __local->hmod;
                ret = (CUresult) __ret->ret;

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MODULE_GET_FUNCTION:
        {
            struct cu_cu_module_get_function_ret *__ret = (struct cu_cu_module_get_function_ret *)__cmd;
            struct cu_cu_module_get_function_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_module_get_function_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_module_get_function_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                CUfunction *hfunc;
                char *name;

                hfunc = __local->hfunc;
                name = __local->name;

                /* Output: CUfunction * hfunc */
                {
                    if ((__ret->hfunc) != (NULL)) {
                        volatile size_t __buffer_size = ((size_t) (1));
                        CUfunction *__src_hfunc_0 = (CUfunction *)__chan->chan_get_buffer(__chan, __cmd, __ret->hfunc);
                        const size_t __hfunc_size_0 = __buffer_size;
                        size_t __hfunc_index_0;
                        for (__hfunc_index_0 = 0; __hfunc_index_0 < __hfunc_size_0; __hfunc_index_0++) {
                            CUfunction *__hfunc_a_0 = (CUfunction *) (__local->hfunc) + __hfunc_index_0;
                            CUfunction *__hfunc_b_0 = (CUfunction *) (__src_hfunc_0) + __hfunc_index_0;

                            {
                                BUG_ON(__hfunc_a_0 == NULL || __hfunc_b_0 == NULL);
                                if (__hfunc_a_0 && __hfunc_b_0)
                                    *__hfunc_a_0 = *__hfunc_b_0;
                            }
                        }
                    }
                }

                /* Output: CUresult ret */
                {
                    __local->ret = __ret->ret;
                }
                kava_parse_function_args(name, &kava_metadata(*hfunc)->func_argc,
                                         kava_metadata(*hfunc)->func_arg_is_handle,
                                         kava_metadata(*hfunc)->func_arg_size);
            }

            __local->__call_complete = 1;
            if (__local->__handler_deallocate) {
                vfree(__local);
            }
            break;
        }

        case RET_CUDA___CU_LAUNCH_KERNEL:
        {
            print_timestamp("internal_start");
            struct cu_cu_launch_kernel_ret *__ret = (struct cu_cu_launch_kernel_ret *)__cmd;
            struct cu_cu_launch_kernel_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_launch_kernel_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_launch_kernel_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
                {
                    __local->ret = __ret->ret;
                }
            }

            __local->__call_complete = 1;
            if (__local->__handler_deallocate) {
                vfree(__local);
            }
            print_timestamp("internal_end");
            break;
        }

        case RET_CUDA___CU_CTX_DESTROY_V2:
        {
            struct cu_cu_ctx_destroy_v2_ret *__ret = (struct cu_cu_ctx_destroy_v2_ret *)__cmd;
            struct cu_cu_ctx_destroy_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_ctx_destroy_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_ctx_destroy_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MEM_ALLOC_V2:
        {
            struct cu_cu_mem_alloc_v2_ret *__ret = (struct cu_cu_mem_alloc_v2_ret *)__cmd;
            struct cu_cu_mem_alloc_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_mem_alloc_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_mem_alloc_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUdeviceptr * dptr */
                {
                    if ((__ret->dptr) != (NULL)) {
                        CUdeviceptr *__src_dptr_0 = (CUdeviceptr *)__chan->chan_get_buffer(__chan, __cmd, __ret->dptr);
                        BUG_ON(__local->dptr == NULL);
                        if (__local->dptr)
                            memcpy(__local->dptr, __src_dptr_0, sizeof(CUdeviceptr));
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MEMCPY_HTO_D_V2:
        {
            struct cu_cu_memcpy_hto_d_v2_ret *__ret = (struct cu_cu_memcpy_hto_d_v2_ret *)__cmd;
            struct cu_cu_memcpy_hto_d_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_memcpy_hto_d_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_memcpy_hto_d_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MEMCPY_DTO_H_V2:
        {
            struct cu_cu_memcpy_dto_h_v2_ret *__ret = (struct cu_cu_memcpy_dto_h_v2_ret *)__cmd;
            struct cu_cu_memcpy_dto_h_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_memcpy_dto_h_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_memcpy_dto_h_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                size_t ByteCount;
                ByteCount = __local->ByteCount;

                /* Output: void * dstHost */
                {
                    if ((__ret->dstHost) != (NULL)) {
                        if (kava_shm_offset(__local->dstHost) >= 0) {
                        }
                        else {
                            volatile size_t __buffer_size = ((size_t) (ByteCount));
                            void *__src_dstHost_0 = (void *)__chan->chan_get_buffer(__chan, __cmd, __ret->dstHost);
                            BUG_ON(__local->dstHost == NULL);
                            if (__local->dstHost)
                                memcpy(__local->dstHost, __src_dstHost_0, __buffer_size * sizeof(void));
                        }
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_CTX_SYNCHRONIZE:
        {
            struct cu_cu_ctx_synchronize_ret *__ret = (struct cu_cu_ctx_synchronize_ret *)__cmd;
            struct cu_cu_ctx_synchronize_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_ctx_synchronize_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_ctx_synchronize_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_CTX_SET_CURRENT:
        {
            struct cu_cu_ctx_set_current_ret *__ret = (struct cu_cu_ctx_set_current_ret *)__cmd;
            struct cu_cu_ctx_set_current_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_ctx_set_current_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_ctx_set_current_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_DRIVER_GET_VERSION:
        {
            struct cu_cu_driver_get_version_ret *__ret = (struct cu_cu_driver_get_version_ret *)__cmd;
            struct cu_cu_driver_get_version_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_driver_get_version_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_driver_get_version_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: int * driverVersion */
                {
                    if ((__ret->driverVersion) != (NULL)) {
                        volatile size_t __buffer_size = ((size_t) (1));
                        int *__src_driverVersion_0;
                        __src_driverVersion_0 = (int *)__chan->chan_get_buffer(__chan, __cmd, __ret->driverVersion);
                        BUG_ON(__local->driverVersion == NULL);
                        if (__local->driverVersion)
                            memcpy(__local->driverVersion, __src_driverVersion_0, __buffer_size * sizeof(int));
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MEM_FREE_V2:
        {
            struct cu_cu_mem_free_v2_ret *__ret = (struct cu_cu_mem_free_v2_ret *)__cmd;
            struct cu_cu_mem_free_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_mem_free_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_mem_free_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MODULE_GET_GLOBAL_V2:
        {
            struct cu_cu_module_get_global_v2_ret *__ret = (struct cu_cu_module_get_global_v2_ret *)__cmd;
            struct cu_cu_module_get_global_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_module_get_global_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_module_get_global_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUdeviceptr * dptr */
                {
                    if ((__ret->dptr) != (NULL)) {
                        volatile size_t __buffer_size = ((size_t) (1));
                        CUdeviceptr *__src_dptr_0 = (CUdeviceptr *)__chan->chan_get_buffer(__chan, __cmd, __ret->dptr);
                        BUG_ON(__local->dptr == NULL);
                        if (__local->dptr)
                            memcpy(__local->dptr, __src_dptr_0, __buffer_size * sizeof(CUdeviceptr));
                    }
                }

                /* Output: size_t * bytes */
                {
                    if ((__ret->bytes) != (NULL)) {
                        volatile size_t __buffer_size = ((size_t) (1));
                        size_t *__src_bytes_0 = (size_t *)__chan->chan_get_buffer(__chan, __cmd, __ret->bytes);
                        BUG_ON(__local->bytes == NULL);
                        if (__local->bytes)
                            memcpy(__local->bytes, __src_bytes_0, __buffer_size * sizeof(size_t));
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_DEVICE_GET_COUNT:
        {
            struct cu_cu_device_get_count_ret *__ret = (struct cu_cu_device_get_count_ret *)__cmd;
            struct cu_cu_device_get_count_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_device_get_count_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_device_get_count_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: int * count */
                {
                    if ((__ret->count) != (NULL)) {
                        volatile size_t __buffer_size = ((size_t) (1));
                        int *__src_count_0 = (int *)__chan->chan_get_buffer(__chan, __cmd, __ret->count);
                        BUG_ON(__local->count == NULL);
                        if (__local->count)
                            memcpy(__local->count, __src_count_0, __buffer_size * sizeof(int));
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_FUNC_SET_CACHE_CONFIG:
        {
            struct cu_cu_func_set_cache_config_ret *__ret = (struct cu_cu_func_set_cache_config_ret *)__cmd;
            struct cu_cu_func_set_cache_config_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_func_set_cache_config_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_func_set_cache_config_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_STREAM_CREATE:
        {
            struct cu_cu_stream_create_ret *__ret = (struct cu_cu_stream_create_ret *)__cmd;
            struct cu_cu_stream_create_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_stream_create_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_stream_create_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUstream * phStream */
                {
                    if ((__ret->phStream) != (NULL)) {
                        CUstream *__src_phStream_0 = (CUstream *)__chan->chan_get_buffer(__chan, __cmd, __ret->phStream);
                        BUG_ON(__local->phStream == NULL);
                        if (__local->phStream)
                            memcpy(__local->phStream, __src_phStream_0, sizeof(CUstream));
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_STREAM_SYNCHRONIZE:
        {
            struct cu_cu_stream_synchronize_ret *__ret = (struct cu_cu_stream_synchronize_ret *)__cmd;
            struct cu_cu_stream_synchronize_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_stream_synchronize_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_stream_synchronize_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_STREAM_DESTROY_V2:
        {
            struct cu_cu_stream_destroy_v2_ret *__ret = (struct cu_cu_stream_destroy_v2_ret *)__cmd;
            struct cu_cu_stream_destroy_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_stream_destroy_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_stream_destroy_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MEMCPY_HTO_D_ASYNC_V2:
        {
            struct cu_cu_memcpy_hto_d_async_v2_ret *__ret = (struct cu_cu_memcpy_hto_d_async_v2_ret *)__cmd;
            struct cu_cu_memcpy_hto_d_async_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_memcpy_hto_d_async_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_memcpy_hto_d_async_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MEMCPY_DTO_H_ASYNC_V2:
        {
            struct cu_cu_memcpy_dto_h_async_v2_ret *__ret = (struct cu_cu_memcpy_dto_h_async_v2_ret *)__cmd;
            struct cu_cu_memcpy_dto_h_async_v2_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_memcpy_dto_h_async_v2_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_memcpy_dto_h_async_v2_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: void * dstHost */
                {
                    if ((__ret->dstHost) != (NULL)) {
                        if (kava_shm_offset(__local->dstHost) >= 0) {
                        }
                        else {
                            /* Size is in bytes until the division below. */
                            volatile size_t __buffer_size;
                            void *__src_dstHost_0 = (void *)__chan->chan_get_buffer(__chan, __cmd, __ret->dstHost);
                            BUG_ON(__buffer_size != __local->ByteCount);
                            BUG_ON(__buffer_size % sizeof(void) != 0);
                            BUG_ON(__local->dstHost == NULL);
#warning The memory synchronization should be fixed by mmap.
                            if (__local->dstHost)
                                memcpy(__local->dstHost, __src_dstHost_0, __buffer_size);
                        }
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_GET_ERROR_STRING:
        {
            struct cu_cu_get_error_string_ret *__ret = (struct cu_cu_get_error_string_ret *)__cmd;
            struct cu_cu_get_error_string_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_get_error_string_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_get_error_string_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: const char ** pStr */
                {
                    if ((__ret->pStr) != (NULL)) {
                        const char **__src_pStr_0 = (const char **)__chan->chan_get_buffer(__chan, __cmd, __ret->pStr);
                        const size_t __pStr_size_0 = ((size_t) (1));
                        size_t __pStr_index_0;

                        if (__local->pStr != NULL) {
                            for (__pStr_index_0 = 0; __pStr_index_0 < __pStr_size_0; __pStr_index_0++) {
                                char **__pStr_a_0 = (char **)(__local->pStr) + __pStr_index_0;
                                char **__pStr_b_0 = (char **)(__src_pStr_0) + __pStr_index_0;

                                const char *__src_pStr_1 = (const char *)__chan->chan_get_buffer(
                                        __chan, __cmd, *__pStr_b_0);
                                if (__src_pStr_1 != NULL) {
                                    volatile size_t __buffer_size = ((size_t)(strlen(__src_pStr_1) + 1));
                                    *__pStr_a_0 = (char *)kmalloc(__buffer_size, GFP_KERNEL);
                                    memcpy(*__pStr_a_0, __src_pStr_1, __buffer_size);
                                }
                            }
                        }
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_DEVICE_GET_ATTRIBUTE:
        {
            struct cu_cu_device_get_attribute_ret *__ret = (struct cu_cu_device_get_attribute_ret *)__cmd;
            struct cu_cu_device_get_attribute_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_device_get_attribute_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_device_get_attribute_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: int * pi */
                {
                    if ((__ret->pi) != (NULL)) {
                        volatile size_t __buffer_size = ((size_t) (1));
                        int *__src_pi_0 = (int *)__chan->chan_get_buffer(__chan, __cmd, __ret->pi);
                        if (__local->pi != NULL)
                            memcpy(__local->pi, __src_pi_0, __buffer_size * sizeof(int));
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_DEVICE_GET_NAME:
        {
            struct cu_cu_device_get_name_ret *__ret = (struct cu_cu_device_get_name_ret *)__cmd;
            struct cu_cu_device_get_name_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_device_get_name_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_device_get_name_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: char * name */
                {
                    if ((__ret->name) != (NULL)) {
                        volatile size_t __buffer_size = ((size_t) (__local->len));
                        char *__src_name_0 = (char *)__chan->chan_get_buffer(__chan, __cmd, __ret->name);
                        if (__local->name != NULL)
                            memcpy(__local->name, __src_name_0, __buffer_size * sizeof(char));
                    }
                }

                /* Output: CUresult ret */
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

        case RET_CUDA___CU_MEM_ALLOC_PITCH:
        {
            struct cu_cu_mem_alloc_pitch_ret *__ret = (struct cu_cu_mem_alloc_pitch_ret *)__cmd;
            struct cu_cu_mem_alloc_pitch_call_record *__local;
            BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
            BUG_ON(__ret->base.command_size != sizeof(struct cu_cu_mem_alloc_pitch_ret) &&
                   "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
            __local = (struct cu_cu_mem_alloc_pitch_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

            {
                /* Output: CUdeviceptr * dptr */
                {
                    if ((__ret->dptr) != (NULL)) {
                        CUdeviceptr *__src_dptr_0 = (CUdeviceptr *)__chan->chan_get_buffer(__chan, __cmd, __ret->dptr);
                        BUG_ON(__local->dptr == NULL);
                        if (__local->dptr)
                            memcpy(__local->dptr, __src_dptr_0, sizeof(CUdeviceptr));
                    }
                }

                /* Output: size_t * pPitch*/
                {
                    if ((__ret->pPitch) != (NULL)) {
                        size_t *__src_pPitch_0 = (size_t *)__chan->chan_get_buffer(__chan, __cmd, __ret->pPitch);
                        BUG_ON(__local->pPitch == NULL);
                        if (__local->pPitch)
                            memcpy(__local->pPitch, __src_pPitch_0, sizeof(size_t));
                    }
                }

                /* Output: CUresult ret */
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
            pr_err("Unrecognized CUDA command: %lu\n", __cmd->command_id);
    }
}

void __print_command_cuda(FILE* file, const struct kava_chan *__chan,
                        const struct kava_cmd_base *__cmd)
{
    switch (__cmd->command_id) {
        case RET_CUDA___CU_INIT:
            pr_info("cuInit is responded\n");
            break;

        //default:
        //    pr_err("Unrecognized CUDA response: %lu\n", __cmd->command_id);
    }
}

void __kava_stop_shadow_thread(void)
{
    struct kava_stop_shadow_thread_call *__cmd;
    int64_t __thread_id;

    __cmd = (struct kava_stop_shadow_thread_call *)chan->cmd_new(chan, sizeof(struct kava_stop_shadow_thread_call), 0);
    __cmd->base.mode = KAVA_CMD_MODE_INTERNAL;
    __cmd->base.command_id = KAVA_CMD_ID_HANDLER_THREAD_EXIT;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
}
EXPORT_SYMBOL(__kava_stop_shadow_thread);

int CUDAAPI cudaGetGPUUtilizationRates(void)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_get_gpu_utilization_rates_call *__cmd;
    struct cu_cu_get_gpu_utilization_rates_call_record *__call_record;
    int64_t __thread_id;
    int ret;

    size_t __total_buffer_size = 0;
    {
    }

    __cmd = (struct cu_cu_get_gpu_utilization_rates_call *)chan->cmd_new(chan, sizeof(struct cu_cu_get_gpu_utilization_rates_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CUDA_GET_GPU_UTILIZATION_RATES;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    __call_record = (struct cu_cu_get_gpu_utilization_rates_call_record *)vmalloc(sizeof(struct cu_cu_get_gpu_utilization_rates_call_record));

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(cudaGetGPUUtilizationRates);

CUresult CUDAAPI cuInit(unsigned int Flags)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_init_call *__cmd;
    struct cu_cu_init_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;

    __cmd = (struct cu_cu_init_call *)chan->cmd_new(chan, sizeof(struct cu_cu_init_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_INIT;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: unsigned int Flags */
        {
            __cmd->Flags = Flags;
        }
    }

    __call_record = (struct cu_cu_init_call_record *)vmalloc(sizeof(struct cu_cu_init_call_record));

    __call_record->Flags = Flags;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}
EXPORT_SYMBOL(cuInit);

EXPORTED CUresult
cuDeviceGet(CUdevice * device, int ordinal)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_device_get_call *__cmd;
    struct cu_cu_device_get_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_device_get_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_device_get_call), __total_buffer_size);

    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_DEVICE_GET;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUdevice * device */
        {
            if ((device) != (NULL)) {
                __cmd->device = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->device = NULL;
            }
        }
        /* Input: int ordinal */
        {
            __cmd->ordinal = ordinal;
        }
    }

    __call_record = (struct cu_cu_device_get_call_record *)vmalloc(
            sizeof(struct cu_cu_device_get_call_record));

    __call_record->device = device;
    __call_record->ordinal = ordinal;
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
EXPORT_SYMBOL(cuDeviceGet);

EXPORTED CUresult
cuCtxCreate_v2(CUcontext * pctx, unsigned int flags, CUdevice dev)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_ctx_create_v2_call *__cmd;
    struct cu_cu_ctx_create_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_ctx_create_v2_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_ctx_create_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_CTX_CREATE_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUcontext * pctx */
        {
            if ((pctx) != (NULL)) {
                __cmd->pctx = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->pctx = NULL;
            }
        }
        /* Input: unsigned int flags */
        {
            __cmd->flags = flags;
        }
        /* Input: CUdevice dev */
        {
            __cmd->dev = dev;
        }
    }

    __call_record = (struct cu_cu_ctx_create_v2_call_record *)vmalloc(
            sizeof(struct cu_cu_ctx_create_v2_call_record));
    __call_record->pctx = pctx;
    __call_record->flags = flags;
    __call_record->dev = dev;

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
EXPORT_SYMBOL(cuCtxCreate_v2);

EXPORTED CUresult
cuModuleLoad(CUmodule *module, const char *fname)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_module_load_call *__cmd;
    struct cu_cu_module_load_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
        /* Size: const char * fname */
        if ((fname) != (NULL) && (strlen(fname) + 1) > (0)) {
            __total_buffer_size +=
                chan->chan_buffer_size(chan, ((size_t) (strlen(fname) + 1)) * sizeof(const char));
        }
    }
    __cmd = (struct cu_cu_module_load_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_module_load_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MODULE_LOAD;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUmodule * module */
        {
            if ((module) != (NULL)) {
                __cmd->module = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->module = NULL;
            }
        }
        /* Input: const char * fname */
        {
            if ((fname) != (NULL) && (strlen(fname) + 1) > (0)) {
                __cmd->fname =
                    (char *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, fname,
                    ((size_t) (strlen(fname) + 1)) * sizeof(const char));
            } else {
                __cmd->fname = NULL;
            }
        }
    }

    __call_record = (struct cu_cu_module_load_call_record *)vmalloc(
            sizeof(struct cu_cu_module_load_call_record));
    __call_record->module = module;
    __call_record->fname = (char *)fname;
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
EXPORT_SYMBOL(cuModuleLoad);

EXPORTED CUresult
cuModuleUnload(CUmodule hmod)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_module_unload_call *__cmd;
    struct cu_cu_module_unload_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_module_unload_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_module_unload_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MODULE_UNLOAD;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUmodule hmod */
        {
            __cmd->hmod = hmod;
        }
    }

    __call_record = (struct cu_cu_module_unload_call_record *)vmalloc(
            sizeof(struct cu_cu_module_unload_call_record));

    __call_record->hmod = hmod;
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
EXPORT_SYMBOL(cuModuleUnload);

EXPORTED CUresult
cuModuleGetFunction(CUfunction * hfunc, CUmodule hmod, const char *name)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_module_get_function_call *__cmd;
    struct cu_cu_module_get_function_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
        /* Size: const char * name */
        if ((name) != (NULL) && (strlen(name) + 1) > (0)) {
            __total_buffer_size += chan->chan_buffer_size(
                    chan, ((size_t) (strlen(name) + 1)) * sizeof(const char));
        }
    }
    __cmd = (struct cu_cu_module_get_function_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_module_get_function_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MODULE_GET_FUNCTION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUfunction * hfunc */
        {
            if ((hfunc) != (NULL)) {
                __cmd->hfunc = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->hfunc = NULL;
            }
        }
        /* Input: CUmodule hmod */
        {
            __cmd->hmod = hmod;
        }
        /* Input: const char * name */
        {
            if ((name) != (NULL) && (strlen(name) + 1) > (0)) {
                __cmd->name =
                    (char *)chan->chan_attach_buffer(chan,
                            (struct kava_cmd_base *)__cmd,
                            name,
                            ((size_t) (strlen(name) + 1)) * sizeof(const char));
            } else {
                __cmd->name = NULL;
            }
        }
    }

    __call_record = (struct cu_cu_module_get_function_call_record *)vmalloc(
            sizeof(struct cu_cu_module_get_function_call_record));

    __call_record->hfunc = hfunc;
    __call_record->hmod = hmod;
    __call_record->name = (char *)name;
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
EXPORT_SYMBOL(cuModuleGetFunction);

EXPORTED CUresult
cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes,
    CUstream hStream, void **kernelParams, void **extra)
{
    print_timestamp("start");
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_launch_kernel_call *__cmd;
    struct cu_cu_launch_kernel_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    struct kava_buffer_list *__kava_alloc_list_cuLaunchKernel =
        kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: void ** extra */
        if ((extra) != (NULL) && (cuLaunchKernel_extra_size(extra)) > (0)) {
            const size_t __extra_size_0 = ((size_t) (cuLaunchKernel_extra_size(extra)));
            size_t __extra_index_0;
            for (__extra_index_0 = 0; __extra_index_0 < __extra_size_0; __extra_index_0++) {
                void **__extra_a_0;
                __extra_a_0 = (void **)(extra) + __extra_index_0;

                if ((*__extra_a_0) != (NULL)) {
                    __total_buffer_size += chan->chan_buffer_size(
                            chan, ((size_t) (1)) * sizeof(void));
                }
            }
            __total_buffer_size += chan->chan_buffer_size(
                    chan, ((size_t) (cuLaunchKernel_extra_size(extra))) * sizeof(void *));
        }

        /* Size: void ** kernelParams */
        if ((kernelParams) != (NULL) && (kava_metadata(f)->func_argc) > (0)) {
            const size_t __kernelParams_size_0 = ((size_t) (kava_metadata(f)->func_argc));
            size_t __kernelParams_index_0;
            for (__kernelParams_index_0 = 0; __kernelParams_index_0 < __kernelParams_size_0;
                __kernelParams_index_0++) {
                const size_t ava_index = __kernelParams_index_0;

                void **__kernelParams_a_0;
                __kernelParams_a_0 = (void **)(kernelParams) + ava_index;

                if (kava_metadata(f)->func_arg_is_handle[ava_index]) {
                    if ((*__kernelParams_a_0) != (NULL)) {
                        __total_buffer_size += chan->chan_buffer_size(chan, sizeof(CUdeviceptr));
                    }
                } else {
                    if ((*__kernelParams_a_0) != (NULL)) {
                        __total_buffer_size += chan->chan_buffer_size(chan,
                                kava_metadata(f)->func_arg_size[ava_index]);
                    }
                }
            }
            __total_buffer_size += chan->chan_buffer_size(
                    chan, ((size_t) (kava_metadata(f)->func_argc)) * sizeof(void *));
        }
    }
    print_timestamp("void**1st");
    __cmd = (struct cu_cu_launch_kernel_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_launch_kernel_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_LAUNCH_KERNEL;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;
    print_timestamp("cmd_new");
    {
        /* Input: CUfunction f */
        {
            __cmd->f = f;
        }
        /* Input: unsigned int gridDimX */
        {
            __cmd->gridDimX = gridDimX;
        }
        /* Input: unsigned int gridDimY */
        {
            __cmd->gridDimY = gridDimY;
        }
        /* Input: unsigned int gridDimZ */
        {
            __cmd->gridDimZ = gridDimZ;
        }
        /* Input: unsigned int blockDimX */
        {
            __cmd->blockDimX = blockDimX;
        }
        /* Input: unsigned int blockDimY */
        {
            __cmd->blockDimY = blockDimY;
        }
        /* Input: unsigned int blockDimZ */
        {
            __cmd->blockDimZ = blockDimZ;
        }
        /* Input: unsigned int sharedMemBytes */
        {
            __cmd->sharedMemBytes = sharedMemBytes;
        }
        /* Input: CUstream hStream */
        {
            __cmd->hStream = hStream;
        }
        /* Input: void ** kernelParams */
        {
            if ((kernelParams) != (NULL) && (kava_metadata(f)->func_argc) > (0)) {
                const size_t __size_kernelParams_0 = ((size_t) (kava_metadata(f)->func_argc));
                void **__tmp_kernelParams_0 = (void **)vmalloc(__size_kernelParams_0 * sizeof(void *));
                const size_t __kernelParams_size_0 = __size_kernelParams_0;
                size_t __kernelParams_index_0;
                kava_endpoint_buffer_list_add(__kava_alloc_list_cuLaunchKernel,
                        kava_buffer_with_deallocator_new(vfree, (void *)__tmp_kernelParams_0));

                for (__kernelParams_index_0 = 0; __kernelParams_index_0 < __kernelParams_size_0;
                    __kernelParams_index_0++) {
                    const size_t ava_index = __kernelParams_index_0;
                    void **__kernelParams_a_0;
                    void **__kernelParams_b_0;

                    __kernelParams_a_0 = (void **)(kernelParams) + ava_index;
                    __kernelParams_b_0 = (void **)(__tmp_kernelParams_0) + ava_index;

                    if (kava_metadata(f)->func_arg_is_handle[ava_index]) { {
                            if ((*__kernelParams_a_0) != (NULL)) {
                                *__kernelParams_b_0 = (CUdeviceptr *)chan->chan_attach_buffer(
                                        chan, (struct kava_cmd_base *)__cmd,
                                    *__kernelParams_a_0, sizeof(CUdeviceptr));
                            } else {
                                *__kernelParams_b_0 = NULL;
                            }
                    }
                    } else { {
                            if ((*__kernelParams_a_0) != (NULL)) {
                                *__kernelParams_b_0 = (int *)chan->chan_attach_buffer(
                                        chan, (struct kava_cmd_base *)__cmd,
                                        *__kernelParams_a_0,
                                        (size_t) kava_metadata(f)->func_arg_size[ava_index]);
                            } else {
                                *__kernelParams_b_0 = NULL;
                            }
                    }
                    }
                }
                __cmd->kernelParams = (void **)chan->chan_attach_buffer(
                        chan, (struct kava_cmd_base *)__cmd, __tmp_kernelParams_0,
                        ((size_t) (kava_metadata(f)->func_argc)) * sizeof(void *));
            } else {
                __cmd->kernelParams = NULL;
            }
            print_timestamp("params");
        }
        /* Input: void ** extra */
        {
            if ((extra) != (NULL) && (cuLaunchKernel_extra_size(extra)) > (0)) {
                const size_t __size_extra_0 = ((size_t) (cuLaunchKernel_extra_size(extra)));
                void **__tmp_extra_0 = (void **)vmalloc(__size_extra_0 * sizeof(void *));
                const size_t __extra_size_0 = __size_extra_0;
                size_t __extra_index_0;
                kava_endpoint_buffer_list_add(__kava_alloc_list_cuLaunchKernel,
                        kava_buffer_with_deallocator_new(vfree, __tmp_extra_0));

                for (__extra_index_0 = 0; __extra_index_0 < __extra_size_0; __extra_index_0++) {
                    void **__extra_a_0;
                    void **__extra_b_0;

                    __extra_a_0 = (void **)(extra) + __extra_index_0;
                    __extra_b_0 = (void **)(__tmp_extra_0) + __extra_index_0;

                    {
                        if ((*__extra_a_0) != (NULL)) {
                            *__extra_b_0 = (void *)chan->chan_attach_buffer(
                                    chan, (struct kava_cmd_base *)__cmd,
                                    *__extra_a_0, ((size_t) (1)) * sizeof(void));
                        } else {
                            *__extra_b_0 = NULL;
                        }
                    }
                }
                __cmd->extra = (void **)chan->chan_attach_buffer(
                        chan, (struct kava_cmd_base *)__cmd, __tmp_extra_0,
                        ((size_t) (cuLaunchKernel_extra_size(extra))) * sizeof(void *));
            } else {
                __cmd->extra = NULL;
            }
        }
        print_timestamp("void**2nd");
    }

    __call_record = (struct cu_cu_launch_kernel_call_record *)vmalloc(
            sizeof(struct cu_cu_launch_kernel_call_record));
    __call_record->f = f;
    __call_record->gridDimX = gridDimX;
    __call_record->gridDimY = gridDimY;
    __call_record->gridDimZ = gridDimZ;
    __call_record->blockDimX = blockDimX;
    __call_record->blockDimY = blockDimY;
    __call_record->blockDimZ = blockDimZ;
    __call_record->sharedMemBytes = sharedMemBytes;
    __call_record->hStream = hStream;
    __call_record->extra = extra;
    __call_record->kernelParams = kernelParams;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);
    print_timestamp("add_call");

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    kava_endpoint_buffer_list_free(__kava_alloc_list_cuLaunchKernel); /* Deallocate all memory in the alloc list */
    print_timestamp("cmd_send");
    shadow_thread_handle_command_until(kava_shadow_thread_pool,
            __thread_id, __call_record->__call_complete);
    ret = __call_record->ret;
    print_timestamp("vfree");
    vfree(__call_record);
    print_timestamp("end");
    return ret;
}
EXPORT_SYMBOL(cuLaunchKernel);

EXPORTED CUresult
cuCtxDestroy_v2(CUcontext ctx)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_ctx_destroy_v2_call *__cmd;
    struct cu_cu_ctx_destroy_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_ctx_destroy_v2_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_ctx_destroy_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_CTX_DESTROY_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUcontext ctx */
        {
            __cmd->ctx = ctx;
        }
    }

    __call_record = (struct cu_cu_ctx_destroy_v2_call_record *)vmalloc(
            sizeof(struct cu_cu_ctx_destroy_v2_call_record));
    __call_record->ctx = ctx;
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
EXPORT_SYMBOL(cuCtxDestroy_v2);

EXPORTED CUresult
cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_mem_alloc_v2_call *__cmd;
    struct cu_cu_mem_alloc_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_mem_alloc_v2_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_mem_alloc_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MEM_ALLOC_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUdeviceptr * dptr */
        {
            if ((dptr) != (NULL)) {
                __cmd->dptr = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->dptr = NULL;
            }
        }
        /* Input: size_t bytesize */
        {
            __cmd->bytesize = bytesize;
        }
    }

    __call_record = (struct cu_cu_mem_alloc_v2_call_record *)vmalloc(
            sizeof(struct cu_cu_mem_alloc_v2_call_record));
    __call_record->dptr = dptr;
    __call_record->bytesize = bytesize;
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
EXPORT_SYMBOL(cuMemAlloc_v2);

EXPORTED CUresult
cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_memcpy_hto_d_v2_call *__cmd;
    struct cu_cu_memcpy_hto_d_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * srcHost */
        if ((srcHost) != (NULL) && (ByteCount) > (0)) {
            if (kava_shm_offset(srcHost) >= 0) {
            }
            else {
                __total_buffer_size += chan->chan_buffer_size(
                        chan, ((size_t) (ByteCount)) * sizeof(const void));
            }
        }
    }
    __cmd = (struct cu_cu_memcpy_hto_d_v2_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_memcpy_hto_d_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MEMCPY_HTO_D_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUdeviceptr dstDevice */
        {
            __cmd->dstDevice = dstDevice;
        }
        /* Input: const void * srcHost */
        {
            if ((srcHost) != (NULL) && (ByteCount) > (0)) {
                if (kava_shm_offset(srcHost) >= 0) {
                    __cmd->srcHost = (void *)kava_shm_offset(srcHost);
                    __cmd->__shm_srcHost = 1;
                }
                else {
                    __cmd->srcHost =
                        (void *)chan->chan_attach_buffer(chan,
                                (struct kava_cmd_base *)__cmd, srcHost,
                                ((size_t) (ByteCount)) * sizeof(const void));
                    __cmd->__shm_srcHost = 0;
                }
            } else {
                __cmd->srcHost = NULL;
            }
        }
        /* Input: size_t ByteCount */
        {
            __cmd->ByteCount = ByteCount;
        }
    }

    __call_record = (struct cu_cu_memcpy_hto_d_v2_call_record *)vmalloc(
            sizeof(struct cu_cu_memcpy_hto_d_v2_call_record));
    __call_record->dstDevice = dstDevice;
    __call_record->ByteCount = ByteCount;
    __call_record->srcHost = (void *)srcHost;
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
EXPORT_SYMBOL(cuMemcpyHtoD_v2);

EXPORTED CUresult
cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_memcpy_dto_h_v2_call *__cmd;
    struct cu_cu_memcpy_dto_h_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_memcpy_dto_h_v2_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_memcpy_dto_h_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MEMCPY_DTO_H_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: void * dstHost */
        {
            if ((dstHost) != (NULL) && (ByteCount) > (0)) {
                if (kava_shm_offset(dstHost) >= 0) {
                    __cmd->__shm_dstHost = 1;
                    __cmd->dstHost = (void *)kava_shm_offset(dstHost);
                }
                else {
                    __cmd->dstHost = HAS_OUT_BUFFER_SENTINEL;
                    __cmd->__shm_dstHost = 0;
                }
            } else {
                __cmd->dstHost = NULL;
            }
        }
        /* Input: CUdeviceptr srcDevice */
        {
            __cmd->srcDevice = srcDevice;
        }
        /* Input: size_t ByteCount */
        {
            __cmd->ByteCount = ByteCount;
        }
    }

    __call_record = (struct cu_cu_memcpy_dto_h_v2_call_record *)vmalloc(
            sizeof(struct cu_cu_memcpy_dto_h_v2_call_record));
    __call_record->srcDevice = srcDevice;
    __call_record->ByteCount = ByteCount;
    __call_record->dstHost = dstHost;
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
EXPORT_SYMBOL(cuMemcpyDtoH_v2);

EXPORTED CUresult
cuCtxSynchronize()
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_ctx_synchronize_call *__cmd;
    struct cu_cu_ctx_synchronize_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_ctx_synchronize_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_ctx_synchronize_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_CTX_SYNCHRONIZE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
    }

    __call_record = (struct cu_cu_ctx_synchronize_call_record *)vmalloc(
            sizeof(struct cu_cu_ctx_synchronize_call_record));
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
EXPORT_SYMBOL(cuCtxSynchronize);

CUresult CUDAAPI cuCtxSetCurrent(CUcontext ctx)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    struct cu_cu_ctx_set_current_call *__cmd;
    struct cu_cu_ctx_set_current_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_ctx_set_current_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_ctx_set_current_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_CTX_SET_CURRENT;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUcontext ctx */
        {
            __cmd->ctx = ctx;
        }
    }

    __call_record = (struct cu_cu_ctx_set_current_call_record *)vmalloc(
            sizeof(struct cu_cu_ctx_set_current_call_record));
    __call_record->ctx = ctx;
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
EXPORT_SYMBOL(cuCtxSetCurrent);

EXPORTED CUresult
cuDriverGetVersion(int *driverVersion)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_driver_get_version_call *__cmd;
    struct cu_cu_driver_get_version_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_driver_get_version_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_driver_get_version_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_DRIVER_GET_VERSION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: int * driverVersion */
        {
            if ((driverVersion) != (NULL)) {
                __cmd->driverVersion = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->driverVersion = NULL;
            }
        }
    }

    __call_record = (struct cu_cu_driver_get_version_call_record *)vmalloc(
            sizeof(struct cu_cu_driver_get_version_call_record));
    __call_record->driverVersion = driverVersion;
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
EXPORT_SYMBOL(cuDriverGetVersion);

EXPORTED CUresult
cuMemFree_v2(CUdeviceptr dptr)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_mem_free_v2_call *__cmd;
    struct cu_cu_mem_free_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_mem_free_v2_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_mem_free_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MEM_FREE_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUdeviceptr dptr */
        {
            __cmd->dptr = dptr;
        }
    }

    __call_record = (struct cu_cu_mem_free_v2_call_record *)vmalloc(
            sizeof(struct cu_cu_mem_free_v2_call_record));
    __call_record->dptr = dptr;
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
EXPORT_SYMBOL(cuMemFree_v2);

EXPORTED CUresult
cuModuleGetGlobal_v2(CUdeviceptr * dptr, size_t * bytes, CUmodule hmod, const char *name)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_module_get_global_v2_call *__cmd;
    struct cu_cu_module_get_global_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
        /* Size: const char * name */
        if ((name) != (NULL) && (strlen(name) + 1) > (0)) {
            __total_buffer_size += chan->chan_buffer_size(
                    chan, ((size_t) (strlen(name) + 1)) * sizeof(const char));
        }
    }
    __cmd = (struct cu_cu_module_get_global_v2_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_module_get_global_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MODULE_GET_GLOBAL_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUdeviceptr * dptr */
        {
            if ((dptr) != (NULL)) {
                __cmd->dptr = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->dptr = NULL;
            }
        }
        /* Input: size_t * bytes */
        {
            if ((bytes) != (NULL)) {
                __cmd->bytes = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->bytes = NULL;
            }
        }
        /* Input: CUmodule hmod */
        {
            __cmd->hmod = hmod;
        }
        /* Input: const char * name */
        {
            if ((name) != (NULL) && (strlen(name) + 1) > (0)) {
                __cmd->name = (char *)chan->chan_attach_buffer(
                        chan, (struct kava_cmd_base *)__cmd, name,
                        ((size_t) (strlen(name) + 1)) * sizeof(const char));
            } else {
                __cmd->name = NULL;
            }
        }
    }

    __call_record = (struct cu_cu_module_get_global_v2_call_record *)vmalloc(
            sizeof(struct cu_cu_module_get_global_v2_call_record));
    __call_record->dptr = dptr;
    __call_record->bytes = bytes;
    __call_record->hmod = hmod;
    __call_record->name = (char *)name;
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
EXPORT_SYMBOL(cuModuleGetGlobal_v2);

EXPORTED CUresult
cuDeviceGetCount(int *count)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_device_get_count_call *__cmd;
    struct cu_cu_device_get_count_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_device_get_count_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_device_get_count_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_DEVICE_GET_COUNT;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: int * count */
        {
            if ((count) != (NULL)) {
                __cmd->count = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->count = NULL;
            }
        }
    }

    __call_record = (struct cu_cu_device_get_count_call_record *)vmalloc(
            sizeof(struct cu_cu_device_get_count_call_record));
    __call_record->count = count;
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
EXPORT_SYMBOL(cuDeviceGetCount);

EXPORTED CUresult
cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_func_set_cache_config_call *__cmd;
    struct cu_cu_func_set_cache_config_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_func_set_cache_config_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_func_set_cache_config_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_FUNC_SET_CACHE_CONFIG;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        {
            __cmd->hfunc = hfunc;
            __cmd->config = config;
        }
    }

    __call_record = (struct cu_cu_func_set_cache_config_call_record *)vmalloc(
            sizeof(struct cu_cu_func_set_cache_config_call_record));
    __call_record->hfunc = hfunc;
    __call_record->config = config;
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
EXPORT_SYMBOL(cuFuncSetCacheConfig);

CUresult CUDAAPI cuStreamCreate(CUstream *phStream, unsigned int Flags)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_stream_create_call *__cmd;
    struct cu_cu_stream_create_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_stream_create_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_stream_create_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_STREAM_CREATE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUstream * phStream */
        {
            if ((phStream) != (NULL)) {
                __cmd->phStream = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->phStream = NULL;
            }
        }
        /* Input: unsigned int Flags */
        {
            __cmd->Flags = Flags;
        }
    }

    __call_record = (struct cu_cu_stream_create_call_record *)vmalloc(
            sizeof(struct cu_cu_stream_create_call_record));
    __call_record->phStream = phStream;
    __call_record->Flags = Flags;
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
EXPORT_SYMBOL(cuStreamCreate);

CUresult CUDAAPI cuStreamSynchronize(CUstream hStream)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    struct cu_cu_stream_synchronize_call *__cmd;
    struct cu_cu_stream_synchronize_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_stream_synchronize_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_stream_synchronize_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_STREAM_SYNCHRONIZE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUstream hStream */
        {
            __cmd->hStream = hStream;
        }
    }

    __call_record = (struct cu_cu_stream_synchronize_call_record *)vmalloc(
        sizeof(struct cu_cu_stream_synchronize_call_record));
    __call_record->hStream = hStream;
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
EXPORT_SYMBOL(cuStreamSynchronize);


CUresult CUDAAPI cuStreamDestroy_v2(CUstream hStream)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    struct cu_cu_stream_destroy_v2_call *__cmd;
    struct cu_cu_stream_destroy_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_stream_destroy_v2_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_stream_destroy_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_STREAM_DESTROY_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUstream hStream */
        {
            __cmd->hStream = hStream;
        }
    }

    __call_record = (struct cu_cu_stream_destroy_v2_call_record *)vmalloc(
        sizeof(struct cu_cu_stream_destroy_v2_call_record));
    __call_record->hStream = hStream;
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
EXPORT_SYMBOL(cuStreamDestroy_v2);

CUresult CUDAAPI cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                    const void *srcHost,
                                    size_t ByteCount,
                                    CUstream hStream)
{
#ifdef REPLY_ASYNC_API
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
#endif

    struct cu_cu_memcpy_hto_d_async_v2_call *__cmd;
    struct cu_cu_memcpy_hto_d_async_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * srcHost */
        if ((srcHost) != (NULL) && (ByteCount) > (0)) {
            if (kava_shm_offset(srcHost) >= 0) {
            }
            else {
                __total_buffer_size += ByteCount * sizeof(const void);
            }
        }
    }
    __cmd = (struct cu_cu_memcpy_hto_d_async_v2_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_memcpy_hto_d_async_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MEMCPY_HTO_D_ASYNC_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

#ifdef REPLY_ASYNC_API
    __cmd->__call_id = __call_id;
#endif

    {
        /* Input: CUdeviceptr dstDevice */
        {
            __cmd->dstDevice = dstDevice;
        }
        /* Input: const void * srcHost */
        {
            if ((srcHost) != (NULL) && (ByteCount) > (0)) {
                if (kava_shm_offset(srcHost) >= 0) {
                    __cmd->__shm_srcHost = 1;
                    __cmd->srcHost = (void *)kava_shm_offset(srcHost);
                }
                else {
                    __cmd->srcHost = (void *)chan->chan_attach_buffer(
                            chan, (struct kava_cmd_base *)__cmd,
                            srcHost, (ByteCount) * sizeof(const void));

                    __cmd->__shm_srcHost = 0;
                }
            } else {
                __cmd->srcHost = NULL;
            }
        }
        /* Input: size_t ByteCount */
        {
            __cmd->ByteCount = ByteCount;
        }
        /* Input: CUstream hStream */
        {
            __cmd->hStream = hStream;
        }
    }

#ifdef REPLY_ASYNC_API
    __call_record = (struct cu_cu_memcpy_hto_d_async_v2_call_record *)vmalloc(
        sizeof(struct cu_cu_memcpy_hto_d_async_v2_call_record));
    __call_record->dstDevice = dstDevice;
    __call_record->ByteCount = ByteCount;
    __call_record->hStream = hStream;
    __call_record->srcHost = (void *)srcHost;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);
#endif

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuMemcpyHtoDAsync_v2);

CUresult CUDAAPI cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                                size_t ByteCount, CUstream hStream)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_memcpy_dto_h_async_v2_call *__cmd;
    struct cu_cu_memcpy_dto_h_async_v2_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
        /* Size: void * dstHost */
        if ((dstHost) != (NULL) && (ByteCount) > (0)) {
            if (kava_shm_offset(dstHost) >= 0) {
            }
            else {
            }
        }
    }
    __cmd = (struct cu_cu_memcpy_dto_h_async_v2_call *)chan->cmd_new(chan,
        sizeof(struct cu_cu_memcpy_dto_h_async_v2_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MEMCPY_DTO_H_ASYNC_V2;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: void * dstHost */
        {
            if ((dstHost) != (NULL) && (ByteCount) > (0)) {
                if (kava_shm_offset(dstHost) >= 0) {
                    __cmd->__shm_dstHost = 1;
                    __cmd->dstHost = (void *)kava_shm_offset(dstHost);
                }
                else {
                    __cmd->dstHost = dstHost;

                    __cmd->__shm_dstHost = 0;
                }
            } else {
                __cmd->dstHost = NULL;
            }
        }
        /* Input: CUdeviceptr srcDevice */
        {
            __cmd->srcDevice = srcDevice;
        }
        /* Input: size_t ByteCount */
        {
            __cmd->ByteCount = ByteCount;
        }
        /* Input: CUstream hStream */
        {
            __cmd->hStream = hStream;
        }
    }

    __call_record = (struct cu_cu_memcpy_dto_h_async_v2_call_record *)vmalloc(
        sizeof(struct cu_cu_memcpy_dto_h_async_v2_call_record));
    __call_record->srcDevice = srcDevice;
    __call_record->ByteCount = ByteCount;
    __call_record->hStream = hStream;
    __call_record->dstHost = dstHost;
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);
    return CUDA_SUCCESS;
    //shadow_thread_handle_command_until(kava_shadow_thread_pool,
    //        __thread_id, __call_record->__call_complete);
    //ret = __call_record->ret;
    //vfree(__call_record);
    //return ret;

}
EXPORT_SYMBOL(cuMemcpyDtoHAsync_v2);

CUresult cuGetErrorString(CUresult error, const char **pStr)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_get_error_string_call *__cmd;
    struct cu_cu_get_error_string_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_get_error_string_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_get_error_string_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_GET_ERROR_STRING;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUresult error */
        {
            __cmd->error = error;
        }
        /* Input: const char ** pStr */
        {
            if ((pStr) != (NULL)) {
                __cmd->pStr = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->pStr = NULL;
            }
        }
    }

    __call_record = (struct cu_cu_get_error_string_call_record *)vmalloc(
            sizeof(struct cu_cu_get_error_string_call_record));
    __call_record->error = error;
    __call_record->pStr = (char **)pStr;
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
EXPORT_SYMBOL(cuGetErrorString);

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_device_get_attribute_call *__cmd;
    struct cu_cu_device_get_attribute_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_device_get_attribute_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_device_get_attribute_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_DEVICE_GET_ATTRIBUTE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: int * pi */
        {
            if ((pi) != (NULL)) {
                __cmd->pi = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->pi = NULL;
            }
        }
        /* Input: CUdevice_attribute attrib */
        {
            __cmd->attrib = attrib;
        }
        /* Input: CUdevice dev */
        {
            __cmd->dev = dev;
        }
    }

    __call_record = (struct cu_cu_device_get_attribute_call_record *)vmalloc(
        sizeof(struct cu_cu_device_get_attribute_call_record));
    __call_record->pi = pi;
    __call_record->attrib = attrib;
    __call_record->dev = dev;
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
EXPORT_SYMBOL(cuDeviceGetAttribute);

CUresult cuDeviceGetName(char *name, int len, CUdevice dev)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_device_get_name_call *__cmd;
    struct cu_cu_device_get_name_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_device_get_name_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_device_get_name_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_DEVICE_GET_NAME;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: char * name */
        {
            if ((name) != (NULL) && (len) > (0)) {
                __cmd->name = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->name = NULL;
            }
        }
        /* Input: int len */
        {
            __cmd->len = len;
        }
        /* Input: CUdevice dev */
        {
            __cmd->dev = dev;
        }
    }

    __call_record = (struct cu_cu_device_get_name_call_record *)vmalloc(
            sizeof(struct cu_cu_device_get_name_call_record));
    __call_record->len = len;
    __call_record->dev = dev;
    __call_record->name = name;
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
EXPORT_SYMBOL(cuDeviceGetName);

EXPORTED CUresult CUDAAPI
cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch, size_t WidthInBytes,
                size_t Height, unsigned int ElementSizeBytes)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_mem_alloc_pitch_call *__cmd;
    struct cu_cu_mem_alloc_pitch_call_record *__call_record;
    int64_t __thread_id;
    CUresult ret;

    size_t __total_buffer_size = 0;
    {
    }
    __cmd = (struct cu_cu_mem_alloc_pitch_call *)chan->cmd_new(
            chan, sizeof(struct cu_cu_mem_alloc_pitch_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CU_MEM_ALLOC_PITCH;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {
        /* Input: CUdeviceptr * dptr */
        {
            if ((dptr) != (NULL)) {
                __cmd->dptr = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->dptr = NULL;
            }
        }
        /* Input: size_t * pPitch*/
        {
            if ((pPitch) != (NULL)) {
                __cmd->pPitch = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->pPitch = NULL;
            }
        }
        /* Input: size_t bytesize */
        {
            __cmd->WidthInBytes = WidthInBytes;
        }
        /* Input: size_t bytesize */
        {
            __cmd->Height = Height;
        }
        /* Input: size_t bytesize */
        {
            __cmd->ElementSizeBytes = ElementSizeBytes;
        }
    }

    __call_record = (struct cu_cu_mem_alloc_pitch_call_record *)vmalloc(
            sizeof(struct cu_cu_mem_alloc_pitch_call_record));
    __call_record->dptr = dptr;
    __call_record->pPitch = pPitch;
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
EXPORT_SYMBOL(cuMemAllocPitch_v2);

CUresult CUDAAPI cuTestMmul(unsigned int Flags)
{
    struct kava_cmd_base *cmd;
    struct timespec ts;

    pr_info("cuMmul test API called\n");
    getnstimeofday(&ts);

    cmd = chan->cmd_new(chan, sizeof(struct kava_cmd_base), 0);
    cmd->mode = KAVA_CMD_MODE_API;
    cmd->command_id = CALL_CUDA___CUDA_TEST_MMUL;
    chan->cmd_send(chan, cmd);

    pr_info("cuMmul called: sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);

    // TODO: add the full logic.

    return 0;
}
EXPORT_SYMBOL(cuTestMmul);

CUresult CUDAAPI cuTestInit(void)
{
    struct kava_cmd_base *cmd;
    cmd = chan->cmd_new(chan, sizeof(struct kava_cmd_base), 0);
    cmd->mode = KAVA_CMD_MODE_API;
    cmd->command_id = CALL_CUDA___CUDA_TEST_INIT;
    chan->cmd_send(chan, cmd);
    return 0;
}
EXPORT_SYMBOL(cuTestInit);

CUresult CUDAAPI cuTestFree(void)
{
    struct kava_cmd_base *cmd;
    cmd = chan->cmd_new(chan, sizeof(struct kava_cmd_base), 0);
    cmd->mode = KAVA_CMD_MODE_API;
    cmd->command_id = CALL_CUDA___CUDA_TEST_FREE;
    chan->cmd_send(chan, cmd);
    return 0;
}
EXPORT_SYMBOL(cuTestFree);

CUresult CUDAAPI cuTestKtoU(size_t size)
{
    struct kava_cmd_base *cmd;
    struct timespec ts, ts2;
    size_t mat_size;

    /* Evaluate matrix multiplication in kernel on CPU */
    int i, j, k;
    int *a, *b, *c;
    a = vmalloc(size * size * sizeof(int));
    b = vmalloc(size * size * sizeof(int));
    c = vmalloc(size * size * sizeof(int));
    memset(a, 0, size * size * sizeof(int));
    memset(b, 0, size * size * sizeof(int));
    for (i = 0; i < size * size; i++) {
        get_random_bytes(&a[i], sizeof(int) - 1);
        get_random_bytes(&b[i], sizeof(int) - 1);
        a[i] %= 100;
        b[i] %= 100;
    }
    memset(c, 0, size * size * sizeof(int));

    getnstimeofday(&ts);
    for (i = 0; i < size; i++)
        for (j = 0; j < size; j++)
            for (k = 0; k < size; k++) {
                c[i*size+j] += a[i*size+k] * b[k*size+j];
            }
    getnstimeofday(&ts2);
    pr_info("mmul-cpu spent: size=%lu, usec=%lu\n",
            size, (ts2.tv_sec - ts.tv_sec) * 1000000 + (ts2.tv_nsec - ts.tv_nsec) / 1000);

    vfree(a);
    vfree(b);
    vfree(c);

    mat_size = size * size * 2 * sizeof(int);

    cmd = chan->cmd_new(chan, sizeof(struct kava_cmd_base), mat_size);
    cmd->mode = KAVA_CMD_MODE_API;
    cmd->command_id = CALL_CUDA___CUDA_TEST_K_TO_U;

    getnstimeofday(&ts);
    chan->cmd_send(chan, cmd);
    pr_info("matrix sent: size=%lu KB, sec=%lu, usec=%lu\n",
            mat_size >> 10, ts.tv_sec, ts.tv_nsec / 1000);

    return 0;
}
EXPORT_SYMBOL(cuTestKtoU);

void CUDAAPI cuTestChannel(size_t size)
{
    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);

    struct cu_cu_test_channel_call *__cmd;
    struct cu_cu_test_channel_call_record *__call_record;
    int64_t __thread_id;

    size_t __total_buffer_size = 0;

    if (size < sizeof(struct cu_cu_test_channel_call)) {
        pr_err("Cannot transfer command smaller than %ld bytes\n", sizeof(struct cu_cu_test_channel_call));
        return;
    }
    else {
        __total_buffer_size = size - sizeof(struct cu_cu_test_channel_call);
    }

    __cmd = (struct cu_cu_test_channel_call *)chan->cmd_new(chan, sizeof(struct cu_cu_test_channel_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_CUDA___CUDA_TEST_CHANNEL;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    __call_record = (struct cu_cu_test_channel_call_record *)vmalloc(sizeof(struct cu_cu_test_channel_call_record));
    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    vfree(__call_record);
}
EXPORT_SYMBOL(cuTestChannel);

static int __init cuda_init(void)
{
    kava_register_cmd_handler(KAVA_CMD_MODE_API,
                            NULL,
                            NULL);

    pr_info("Create control device\n");
    init_ctrl_if();
    pr_info("Load cuda kernel library\n");
    init_global_kapi(KAVA_API_ID_CUDA, chan_mode);

    /* Initialize endpoint */
    init_endpoint_lib();
    __handle_command_cuda_init();

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

static void __exit cuda_fini(void)
{
    pr_info("Stop running worker\n");
    stop_worker(chan);

    pr_info("Destroy endpoint\n");
    __handle_command_cuda_destroy();

    pr_info("Unload cuda kernel library\n");
    if (chan)
        chan->chan_free(chan);
    put_global_kapi();
    fini_ctrl_if();
}

module_init(cuda_init);
module_exit(cuda_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("CUDA kernel library");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
