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

#include "api.h"
#include "channel_kern.h"
#include "control.h"
#include "command.h"
#include "command_handler.h"
#include "endpoint.h"
#include "shadow_thread_pool.h"
#include "shared_memory.h"

// Must be included before mvnc_nw.h, so that API
// functions are declared properly.
#include "mvnc_nw.h"
#include "mvnc_nw_utilities.h"

#pragma GCC diagnostic ignored "-Wunused-function"

static char *chan_mode = "netlink_socket";
module_param(chan_mode, charp, 0000);
MODULE_PARM_DESC(chan_mode, "kMVNC channel mode. Default netlink.");

static struct kava_chan *chan;
static struct kava_endpoint __kava_endpoint;

static void __handle_command_mvnc_init(void);
static void __handle_command_mvnc_destroy(void);
void __handle_command_mvnc(struct kava_chan *__chan, const struct kava_cmd_base *__cmd);
void __print_command_mvnc(FILE * file, const struct kava_chan *__chan, const struct kava_cmd_base *__cmd);

#define kava_metadata(p) (&((struct mvnc_metadata*)kava_internal_metadata(&__kava_endpoint, p))->application)

#include "mvnc_nw_utilities.h"

struct mvnc_metadata {
    struct kava_metadata_base base;
    struct metadata application;
};

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
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"

void __attribute__ ((constructor(1))) init_mvnc_worker(void)
{
    __handle_command_mvnc_init();
}

void __attribute__ ((destructor)) destroy_mvnc_worker(void)
{
    __handle_command_mvnc_destroy();
}

static struct kava_chan *
__chan_create(void)
{
    return chan;
}

void
__handle_command_mvnc_init(void)
{
    kava_endpoint_init(&__kava_endpoint, sizeof(struct mvnc_metadata));
    kava_register_cmd_handler(KAVA_CMD_MODE_API, __handle_command_mvnc, __print_command_mvnc);
}

void
__handle_command_mvnc_destroy(void)
{
    kava_endpoint_destroy(&__kava_endpoint);
}

void
__handle_command_mvnc(struct kava_chan *__chan, const struct kava_cmd_base *__cmd)
{
    __chan->cmd_print(__chan, __cmd);
    switch (__cmd->command_id) {
    case RET_MVNC_NC_GLOBAL_SET_OPTION:{
        struct mvnc_nc_global_set_option_ret *__ret = (struct mvnc_nc_global_set_option_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_global_set_option_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_global_set_option_call_record *__local =
            (struct mvnc_nc_global_set_option_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            int option;
            option = __local->option;

            unsigned int dataLength;
            dataLength = __local->dataLength;

            void *data;
            data = __local->data;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GLOBAL_GET_OPTION:{
        struct mvnc_nc_global_get_option_ret *__ret = (struct mvnc_nc_global_get_option_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_global_get_option_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_global_get_option_call_record *__local =
            (struct mvnc_nc_global_get_option_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            int option;
            option = __local->option;

            unsigned int *dataLength;
            dataLength = __local->dataLength;

            void *data;
            data = __local->data;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: unsigned int * dataLength */
            {
                if ((__ret->dataLength) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    unsigned int *__src_dataLength_0;
                    __src_dataLength_0 = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __ret->dataLength);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->dataLength == NULL);
                    memcpy(__local->dataLength, __src_dataLength_0, __buffer_size * sizeof(unsigned int));
                }
            }

            /* Output: void * data */
            {
                if ((__ret->data) != (NULL)) {
                    if (kava_shm_offset(__local->data) >= 0) {
                    } else {
                        volatile size_t __buffer_size = 0;
                        void *__src_data_0;
                        __src_data_0 = (void *)__chan->chan_get_buffer(__chan, __cmd, __ret->data);
                        __buffer_size = ((size_t) (*dataLength));
                        BUG_ON(__local->data == NULL);
                        memcpy(__local->data, __src_data_0, __buffer_size * sizeof(void));
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_DEVICE_SET_OPTION:{
        struct mvnc_nc_device_set_option_ret *__ret = (struct mvnc_nc_device_set_option_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_device_set_option_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_device_set_option_call_record *__local =
            (struct mvnc_nc_device_set_option_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncDeviceHandle_t *deviceHandle;
            deviceHandle = __local->deviceHandle;

            int option;
            option = __local->option;

            unsigned int dataLength;
            dataLength = __local->dataLength;

            void *data;
            data = __local->data;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_DEVICE_GET_OPTION:{
        struct mvnc_nc_device_get_option_ret *__ret = (struct mvnc_nc_device_get_option_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_device_get_option_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_device_get_option_call_record *__local =
            (struct mvnc_nc_device_get_option_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncDeviceHandle_t *deviceHandle;
            deviceHandle = __local->deviceHandle;

            int option;
            option = __local->option;

            unsigned int *dataLength;
            dataLength = __local->dataLength;

            void *data;
            data = __local->data;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: unsigned int * dataLength */
            {
                if ((__ret->dataLength) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    unsigned int *__src_dataLength_0;
                    __src_dataLength_0 = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __ret->dataLength);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->dataLength == NULL);
                    memcpy(__local->dataLength, __src_dataLength_0, __buffer_size * sizeof(unsigned int));
                }
            }

            /* Output: void * data */
            {
                if ((__ret->data) != (NULL)) {
                    if (kava_shm_offset(__local->data) >= 0) {
                    } else {
                        volatile size_t __buffer_size = 0;
                        void *__src_data_0;
                        __src_data_0 = (void *)__chan->chan_get_buffer(__chan, __cmd, __ret->data);
                        __buffer_size = ((size_t) (*dataLength));
                        BUG_ON(__local->data == NULL);
                        memcpy(__local->data, __src_data_0, __buffer_size * sizeof(void));
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_DEVICE_CREATE:{
        struct mvnc_nc_device_create_ret *__ret = (struct mvnc_nc_device_create_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_device_create_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_device_create_call_record *__local =
            (struct mvnc_nc_device_create_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            int index;
            index = __local->index;

            struct ncDeviceHandle_t **deviceHandle;
            deviceHandle = __local->deviceHandle;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: struct ncDeviceHandle_t ** deviceHandle */
            {
                if ((__ret->deviceHandle) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncDeviceHandle_t **__src_deviceHandle_0;
                    __src_deviceHandle_0 =
                        (struct ncDeviceHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->deviceHandle);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->deviceHandle == NULL);
                    const size_t __deviceHandle_size_0 = __buffer_size;
                    size_t __deviceHandle_index_0;
                    for (__deviceHandle_index_0 = 0; __deviceHandle_index_0 < __deviceHandle_size_0;
                        __deviceHandle_index_0++) {
                        const size_t ava_index = __deviceHandle_index_0;

                        struct ncDeviceHandle_t **__deviceHandle_a_0;
                        __deviceHandle_a_0 =
                            (struct ncDeviceHandle_t **)(__local->deviceHandle) + __deviceHandle_index_0;

                        struct ncDeviceHandle_t **__deviceHandle_b_0;
                        __deviceHandle_b_0 =
                            (struct ncDeviceHandle_t **)(__src_deviceHandle_0) + __deviceHandle_index_0;

                        {
                            *__deviceHandle_a_0 = *__deviceHandle_b_0;
                        }
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_DEVICE_OPEN:{
        struct mvnc_nc_device_open_ret *__ret = (struct mvnc_nc_device_open_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_device_open_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_device_open_call_record *__local =
            (struct mvnc_nc_device_open_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncDeviceHandle_t *deviceHandle;
            deviceHandle = __local->deviceHandle;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_DEVICE_CLOSE:{
        struct mvnc_nc_device_close_ret *__ret = (struct mvnc_nc_device_close_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_device_close_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_device_close_call_record *__local =
            (struct mvnc_nc_device_close_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncDeviceHandle_t *deviceHandle;
            deviceHandle = __local->deviceHandle;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_DEVICE_DESTROY:{
        struct mvnc_nc_device_destroy_ret *__ret = (struct mvnc_nc_device_destroy_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_device_destroy_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_device_destroy_call_record *__local =
            (struct mvnc_nc_device_destroy_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncDeviceHandle_t **deviceHandle;
            deviceHandle = __local->deviceHandle;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: struct ncDeviceHandle_t ** deviceHandle */
            {
                if ((__ret->deviceHandle) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncDeviceHandle_t **__src_deviceHandle_0;
                    __src_deviceHandle_0 =
                        (struct ncDeviceHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->deviceHandle);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->deviceHandle == NULL);
                    const size_t __deviceHandle_size_0 = __buffer_size;
                    size_t __deviceHandle_index_0;
                    for (__deviceHandle_index_0 = 0; __deviceHandle_index_0 < __deviceHandle_size_0;
                        __deviceHandle_index_0++) {
                        const size_t ava_index = __deviceHandle_index_0;

                        struct ncDeviceHandle_t **__deviceHandle_a_0;
                        __deviceHandle_a_0 =
                            (struct ncDeviceHandle_t **)(__local->deviceHandle) + __deviceHandle_index_0;

                        struct ncDeviceHandle_t **__deviceHandle_b_0;
                        __deviceHandle_b_0 =
                            (struct ncDeviceHandle_t **)(__src_deviceHandle_0) + __deviceHandle_index_0;

                        {
                            *__deviceHandle_a_0 = *__deviceHandle_b_0;
                        }
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GRAPH_CREATE:{
        struct mvnc_nc_graph_create_ret *__ret = (struct mvnc_nc_graph_create_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_graph_create_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_graph_create_call_record *__local =
            (struct mvnc_nc_graph_create_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            char *ava_name;
            ava_name = __local->ava_name;

            struct ncGraphHandle_t **graphHandle;
            graphHandle = __local->graphHandle;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: struct ncGraphHandle_t ** graphHandle */
            {
                if ((__ret->graphHandle) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncGraphHandle_t **__src_graphHandle_0;
                    __src_graphHandle_0 =
                        (struct ncGraphHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->graphHandle);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->graphHandle == NULL);
                    const size_t __graphHandle_size_0 = __buffer_size;
                    size_t __graphHandle_index_0;
                    for (__graphHandle_index_0 = 0; __graphHandle_index_0 < __graphHandle_size_0;
                        __graphHandle_index_0++) {
                        const size_t ava_index = __graphHandle_index_0;

                        struct ncGraphHandle_t **__graphHandle_a_0;
                        __graphHandle_a_0 = (struct ncGraphHandle_t **)(__local->graphHandle) + __graphHandle_index_0;

                        struct ncGraphHandle_t **__graphHandle_b_0;
                        __graphHandle_b_0 = (struct ncGraphHandle_t **)(__src_graphHandle_0) + __graphHandle_index_0;

                        {
                            *__graphHandle_a_0 = *__graphHandle_b_0;
                        }
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GRAPH_ALLOCATE:{
        struct mvnc_nc_graph_allocate_ret *__ret = (struct mvnc_nc_graph_allocate_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_graph_allocate_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_graph_allocate_call_record *__local =
            (struct mvnc_nc_graph_allocate_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncDeviceHandle_t *deviceHandle;
            deviceHandle = __local->deviceHandle;

            struct ncGraphHandle_t *graphHandle;
            graphHandle = __local->graphHandle;

            unsigned int graphBufferLength;
            graphBufferLength = __local->graphBufferLength;

            void *graphBuffer;
            graphBuffer = __local->graphBuffer;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GRAPH_DESTROY:{
        struct mvnc_nc_graph_destroy_ret *__ret = (struct mvnc_nc_graph_destroy_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_graph_destroy_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_graph_destroy_call_record *__local =
            (struct mvnc_nc_graph_destroy_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncGraphHandle_t **graphHandle;
            graphHandle = __local->graphHandle;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: struct ncGraphHandle_t ** graphHandle */
            {
                if ((__ret->graphHandle) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncGraphHandle_t **__src_graphHandle_0;
                    __src_graphHandle_0 =
                        (struct ncGraphHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->graphHandle);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->graphHandle == NULL);
                    const size_t __graphHandle_size_0 = __buffer_size;
                    size_t __graphHandle_index_0;
                    for (__graphHandle_index_0 = 0; __graphHandle_index_0 < __graphHandle_size_0;
                        __graphHandle_index_0++) {
                        const size_t ava_index = __graphHandle_index_0;

                        struct ncGraphHandle_t **__graphHandle_a_0;
                        __graphHandle_a_0 = (struct ncGraphHandle_t **)(__local->graphHandle) + __graphHandle_index_0;

                        struct ncGraphHandle_t **__graphHandle_b_0;
                        __graphHandle_b_0 = (struct ncGraphHandle_t **)(__src_graphHandle_0) + __graphHandle_index_0;

                        {
                            *__graphHandle_a_0 = *__graphHandle_b_0;
                        }
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GRAPH_SET_OPTION:{
        struct mvnc_nc_graph_set_option_ret *__ret = (struct mvnc_nc_graph_set_option_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_graph_set_option_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_graph_set_option_call_record *__local =
            (struct mvnc_nc_graph_set_option_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncGraphHandle_t *graphHandle;
            graphHandle = __local->graphHandle;

            int option;
            option = __local->option;

            unsigned int dataLength;
            dataLength = __local->dataLength;

            void *data;
            data = __local->data;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GRAPH_GET_OPTION:{
        struct mvnc_nc_graph_get_option_ret *__ret = (struct mvnc_nc_graph_get_option_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_graph_get_option_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_graph_get_option_call_record *__local =
            (struct mvnc_nc_graph_get_option_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncGraphHandle_t *graphHandle;
            graphHandle = __local->graphHandle;

            int option;
            option = __local->option;

            unsigned int *dataLength;
            dataLength = __local->dataLength;

            void *data;
            data = __local->data;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: unsigned int * dataLength */
            {
                if ((__ret->dataLength) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    unsigned int *__src_dataLength_0;
                    __src_dataLength_0 = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __ret->dataLength);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->dataLength == NULL);
                    memcpy(__local->dataLength, __src_dataLength_0, __buffer_size * sizeof(unsigned int));
                }
            }

            /* Output: void * data */
            {
                if ((__ret->data) != (NULL)) {
                    if (kava_shm_offset(__local->data) >= 0) {
                    } else {
                        volatile size_t __buffer_size = 0;
                        void *__src_data_0;
                        __src_data_0 = (void *)__chan->chan_get_buffer(__chan, __cmd, __ret->data);
                        __buffer_size = ((size_t) (*dataLength));
                        BUG_ON(__local->data == NULL);
                        memcpy(__local->data, __src_data_0, __buffer_size * sizeof(void));
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GRAPH_QUEUE_INFERENCE:{
        struct mvnc_nc_graph_queue_inference_ret *__ret = (struct mvnc_nc_graph_queue_inference_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_graph_queue_inference_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_graph_queue_inference_call_record *__local =
            (struct mvnc_nc_graph_queue_inference_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncGraphHandle_t *graphHandle;
            graphHandle = __local->graphHandle;

            unsigned int inFifoCount;
            inFifoCount = __local->inFifoCount;

            unsigned int outFifoCount;
            outFifoCount = __local->outFifoCount;

            struct ncFifoHandle_t **fifoIn;
            fifoIn = __local->fifoIn;

            struct ncFifoHandle_t **fifoOut;
            fifoOut = __local->fifoOut;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: struct ncFifoHandle_t ** fifoIn */
            {
                if ((__ret->fifoIn) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncFifoHandle_t **__src_fifoIn_0;
                    __src_fifoIn_0 = (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->fifoIn);
                    __buffer_size = ((size_t) (inFifoCount));
                    BUG_ON(__local->fifoIn == NULL);
                    const size_t __fifoIn_size_0 = __buffer_size;
                    size_t __fifoIn_index_0;
                    for (__fifoIn_index_0 = 0; __fifoIn_index_0 < __fifoIn_size_0; __fifoIn_index_0++) {
                        const size_t ava_index = __fifoIn_index_0;

                        struct ncFifoHandle_t **__fifoIn_a_0;
                        __fifoIn_a_0 = (struct ncFifoHandle_t **)(__local->fifoIn) + __fifoIn_index_0;

                        struct ncFifoHandle_t **__fifoIn_b_0;
                        __fifoIn_b_0 = (struct ncFifoHandle_t **)(__src_fifoIn_0) + __fifoIn_index_0;

                        {
                            *__fifoIn_a_0 = *__fifoIn_b_0;
                        }
                }}
            }

            /* Output: struct ncFifoHandle_t ** fifoOut */
            {
                if ((__ret->fifoOut) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncFifoHandle_t **__src_fifoOut_0;
                    __src_fifoOut_0 = (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->fifoOut);
                    __buffer_size = ((size_t) (outFifoCount));
                    BUG_ON(__local->fifoOut == NULL);
                    const size_t __fifoOut_size_0 = __buffer_size;
                    size_t __fifoOut_index_0;
                    for (__fifoOut_index_0 = 0; __fifoOut_index_0 < __fifoOut_size_0; __fifoOut_index_0++) {
                        const size_t ava_index = __fifoOut_index_0;

                        struct ncFifoHandle_t **__fifoOut_a_0;
                        __fifoOut_a_0 = (struct ncFifoHandle_t **)(__local->fifoOut) + __fifoOut_index_0;

                        struct ncFifoHandle_t **__fifoOut_b_0;
                        __fifoOut_b_0 = (struct ncFifoHandle_t **)(__src_fifoOut_0) + __fifoOut_index_0;

                        {
                            *__fifoOut_a_0 = *__fifoOut_b_0;
                        }
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GRAPH_QUEUE_INFERENCE_WITH_FIFO_ELEM:{
        struct mvnc_nc_graph_queue_inference_with_fifo_elem_ret *__ret =
            (struct mvnc_nc_graph_queue_inference_with_fifo_elem_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_graph_queue_inference_with_fifo_elem_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_graph_queue_inference_with_fifo_elem_call_record *__local =
            (struct mvnc_nc_graph_queue_inference_with_fifo_elem_call_record *)kava_remove_call(&__kava_endpoint,
            __ret->__call_id);

        {

            struct ncGraphHandle_t *graphHandle;
            graphHandle = __local->graphHandle;

            struct ncFifoHandle_t *fifoIn;
            fifoIn = __local->fifoIn;

            struct ncFifoHandle_t *fifoOut;
            fifoOut = __local->fifoOut;

            unsigned int *inputTensorLength;
            inputTensorLength = __local->inputTensorLength;

            void *userParam;
            userParam = __local->userParam;

            void *inputTensor;
            inputTensor = __local->inputTensor;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: unsigned int * inputTensorLength */
            {
                if ((__ret->inputTensorLength) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    unsigned int *__src_inputTensorLength_0;
                    __src_inputTensorLength_0 =
                        (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __ret->inputTensorLength);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->inputTensorLength == NULL);
                    memcpy(__local->inputTensorLength, __src_inputTensorLength_0, __buffer_size * sizeof(unsigned int));
                }
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS:{
        struct mvnc_nc_graph_allocate_with_fifos_ret *__ret = (struct mvnc_nc_graph_allocate_with_fifos_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_graph_allocate_with_fifos_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_graph_allocate_with_fifos_call_record *__local =
            (struct mvnc_nc_graph_allocate_with_fifos_call_record *)kava_remove_call(&__kava_endpoint,
            __ret->__call_id);

        {

            struct ncDeviceHandle_t *deviceHandle;
            deviceHandle = __local->deviceHandle;

            struct ncGraphHandle_t *graphHandle;
            graphHandle = __local->graphHandle;

            unsigned int graphBufferLength;
            graphBufferLength = __local->graphBufferLength;

            struct ncFifoHandle_t **inFifoHandle;
            inFifoHandle = __local->inFifoHandle;

            struct ncFifoHandle_t **outFifoHandle;
            outFifoHandle = __local->outFifoHandle;

            void *graphBuffer;
            graphBuffer = __local->graphBuffer;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: struct ncFifoHandle_t ** inFifoHandle */
            {
                if ((__ret->inFifoHandle) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncFifoHandle_t **__src_inFifoHandle_0;
                    __src_inFifoHandle_0 =
                        (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->inFifoHandle);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->inFifoHandle == NULL);
                    const size_t __inFifoHandle_size_0 = __buffer_size;
                    size_t __inFifoHandle_index_0;
                    for (__inFifoHandle_index_0 = 0; __inFifoHandle_index_0 < __inFifoHandle_size_0;
                        __inFifoHandle_index_0++) {
                        const size_t ava_index = __inFifoHandle_index_0;

                        struct ncFifoHandle_t **__inFifoHandle_a_0;
                        __inFifoHandle_a_0 = (struct ncFifoHandle_t **)(__local->inFifoHandle) + __inFifoHandle_index_0;

                        struct ncFifoHandle_t **__inFifoHandle_b_0;
                        __inFifoHandle_b_0 = (struct ncFifoHandle_t **)(__src_inFifoHandle_0) + __inFifoHandle_index_0;

                        {
                            *__inFifoHandle_a_0 = *__inFifoHandle_b_0;
                        }
                }}
            }

            /* Output: struct ncFifoHandle_t ** outFifoHandle */
            {
                if ((__ret->outFifoHandle) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncFifoHandle_t **__src_outFifoHandle_0;
                    __src_outFifoHandle_0 =
                        (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->outFifoHandle);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->outFifoHandle == NULL);
                    const size_t __outFifoHandle_size_0 = __buffer_size;
                    size_t __outFifoHandle_index_0;
                    for (__outFifoHandle_index_0 = 0; __outFifoHandle_index_0 < __outFifoHandle_size_0;
                        __outFifoHandle_index_0++) {
                        const size_t ava_index = __outFifoHandle_index_0;

                        struct ncFifoHandle_t **__outFifoHandle_a_0;
                        __outFifoHandle_a_0 =
                            (struct ncFifoHandle_t **)(__local->outFifoHandle) + __outFifoHandle_index_0;

                        struct ncFifoHandle_t **__outFifoHandle_b_0;
                        __outFifoHandle_b_0 =
                            (struct ncFifoHandle_t **)(__src_outFifoHandle_0) + __outFifoHandle_index_0;

                        {
                            *__outFifoHandle_a_0 = *__outFifoHandle_b_0;
                        }
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS_EX:{
        struct mvnc_nc_graph_allocate_with_fifos_ex_ret *__ret =
            (struct mvnc_nc_graph_allocate_with_fifos_ex_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_graph_allocate_with_fifos_ex_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_graph_allocate_with_fifos_ex_call_record *__local =
            (struct mvnc_nc_graph_allocate_with_fifos_ex_call_record *)kava_remove_call(&__kava_endpoint,
            __ret->__call_id);

        {

            struct ncDeviceHandle_t *deviceHandle;
            deviceHandle = __local->deviceHandle;

            struct ncGraphHandle_t *graphHandle;
            graphHandle = __local->graphHandle;

            unsigned int graphBufferLength;
            graphBufferLength = __local->graphBufferLength;

            struct ncFifoHandle_t **inFifoHandle;
            inFifoHandle = __local->inFifoHandle;

            ncFifoType_t inFifoType;
            inFifoType = __local->inFifoType;

            int inNumElem;
            inNumElem = __local->inNumElem;

            ncFifoDataType_t inDataType;
            inDataType = __local->inDataType;

            struct ncFifoHandle_t **outFifoHandle;
            outFifoHandle = __local->outFifoHandle;

            ncFifoType_t outFifoType;
            outFifoType = __local->outFifoType;

            int outNumElem;
            outNumElem = __local->outNumElem;

            ncFifoDataType_t outDataType;
            outDataType = __local->outDataType;

            void *graphBuffer;
            graphBuffer = __local->graphBuffer;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: struct ncFifoHandle_t ** inFifoHandle */
            {
                if ((__ret->inFifoHandle) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncFifoHandle_t **__src_inFifoHandle_0;
                    __src_inFifoHandle_0 =
                        (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->inFifoHandle);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->inFifoHandle == NULL);
                    const size_t __inFifoHandle_size_0 = __buffer_size;
                    size_t __inFifoHandle_index_0;
                    for (__inFifoHandle_index_0 = 0; __inFifoHandle_index_0 < __inFifoHandle_size_0;
                        __inFifoHandle_index_0++) {
                        const size_t ava_index = __inFifoHandle_index_0;

                        struct ncFifoHandle_t **__inFifoHandle_a_0;
                        __inFifoHandle_a_0 = (struct ncFifoHandle_t **)(__local->inFifoHandle) + __inFifoHandle_index_0;

                        struct ncFifoHandle_t **__inFifoHandle_b_0;
                        __inFifoHandle_b_0 = (struct ncFifoHandle_t **)(__src_inFifoHandle_0) + __inFifoHandle_index_0;

                        {
                            *__inFifoHandle_a_0 = *__inFifoHandle_b_0;
                        }
                }}
            }

            /* Output: struct ncFifoHandle_t ** outFifoHandle */
            {
                if ((__ret->outFifoHandle) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncFifoHandle_t **__src_outFifoHandle_0;
                    __src_outFifoHandle_0 =
                        (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->outFifoHandle);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->outFifoHandle == NULL);
                    const size_t __outFifoHandle_size_0 = __buffer_size;
                    size_t __outFifoHandle_index_0;
                    for (__outFifoHandle_index_0 = 0; __outFifoHandle_index_0 < __outFifoHandle_size_0;
                        __outFifoHandle_index_0++) {
                        const size_t ava_index = __outFifoHandle_index_0;

                        struct ncFifoHandle_t **__outFifoHandle_a_0;
                        __outFifoHandle_a_0 =
                            (struct ncFifoHandle_t **)(__local->outFifoHandle) + __outFifoHandle_index_0;

                        struct ncFifoHandle_t **__outFifoHandle_b_0;
                        __outFifoHandle_b_0 =
                            (struct ncFifoHandle_t **)(__src_outFifoHandle_0) + __outFifoHandle_index_0;

                        {
                            *__outFifoHandle_a_0 = *__outFifoHandle_b_0;
                        }
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_FIFO_CREATE:{
        struct mvnc_nc_fifo_create_ret *__ret = (struct mvnc_nc_fifo_create_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_fifo_create_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_fifo_create_call_record *__local =
            (struct mvnc_nc_fifo_create_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            char *ava_name;
            ava_name = __local->ava_name;

            ncFifoType_t ava_type;
            ava_type = __local->ava_type;

            struct ncFifoHandle_t **fifoHandle;
            fifoHandle = __local->fifoHandle;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: struct ncFifoHandle_t ** fifoHandle */
            {
                if ((__ret->fifoHandle) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    struct ncFifoHandle_t **__src_fifoHandle_0;
                    __src_fifoHandle_0 =
                        (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __ret->fifoHandle);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->fifoHandle == NULL);
                    const size_t __fifoHandle_size_0 = __buffer_size;
                    size_t __fifoHandle_index_0;
                    for (__fifoHandle_index_0 = 0; __fifoHandle_index_0 < __fifoHandle_size_0; __fifoHandle_index_0++) {
                        const size_t ava_index = __fifoHandle_index_0;

                        struct ncFifoHandle_t **__fifoHandle_a_0;
                        __fifoHandle_a_0 = (struct ncFifoHandle_t **)(__local->fifoHandle) + __fifoHandle_index_0;

                        struct ncFifoHandle_t **__fifoHandle_b_0;
                        __fifoHandle_b_0 = (struct ncFifoHandle_t **)(__src_fifoHandle_0) + __fifoHandle_index_0;

                        {
                            *__fifoHandle_a_0 = *__fifoHandle_b_0;
                        }
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_FIFO_ALLOCATE:{
        struct mvnc_nc_fifo_allocate_ret *__ret = (struct mvnc_nc_fifo_allocate_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_fifo_allocate_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_fifo_allocate_call_record *__local =
            (struct mvnc_nc_fifo_allocate_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncFifoHandle_t *fifoHandle;
            fifoHandle = __local->fifoHandle;

            struct ncDeviceHandle_t *device;
            device = __local->device;

            struct ncTensorDescriptor_t *tensorDesc;
            tensorDesc = __local->tensorDesc;

            unsigned int numElem;
            numElem = __local->numElem;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_FIFO_SET_OPTION:{
        struct mvnc_nc_fifo_set_option_ret *__ret = (struct mvnc_nc_fifo_set_option_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_fifo_set_option_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_fifo_set_option_call_record *__local =
            (struct mvnc_nc_fifo_set_option_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncFifoHandle_t *fifoHandle;
            fifoHandle = __local->fifoHandle;

            int option;
            option = __local->option;

            unsigned int dataLength;
            dataLength = __local->dataLength;

            void *data;
            data = __local->data;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_FIFO_GET_OPTION:{
        struct mvnc_nc_fifo_get_option_ret *__ret = (struct mvnc_nc_fifo_get_option_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_fifo_get_option_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_fifo_get_option_call_record *__local =
            (struct mvnc_nc_fifo_get_option_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncFifoHandle_t *fifoHandle;
            fifoHandle = __local->fifoHandle;

            int option;
            option = __local->option;

            unsigned int *dataLength;
            dataLength = __local->dataLength;

            void *data;
            data = __local->data;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: unsigned int * dataLength */
            {
                if ((__ret->dataLength) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    unsigned int *__src_dataLength_0;
                    __src_dataLength_0 = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __ret->dataLength);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->dataLength == NULL);
                    memcpy(__local->dataLength, __src_dataLength_0, __buffer_size * sizeof(unsigned int));
                }
            }

            /* Output: void * data */
            {
                if ((__ret->data) != (NULL)) {
                    if (kava_shm_offset(__local->data) >= 0) {
                    } else {
                        volatile size_t __buffer_size = 0;
                        void *__src_data_0;
                        __src_data_0 = (void *)__chan->chan_get_buffer(__chan, __cmd, __ret->data);
                        __buffer_size = ((size_t) (*dataLength));
                        BUG_ON(__local->data == NULL);
                        memcpy(__local->data, __src_data_0, __buffer_size * sizeof(void));
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_FIFO_DESTROY:{
        struct mvnc_nc_fifo_destroy_ret *__ret = (struct mvnc_nc_fifo_destroy_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_fifo_destroy_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_fifo_destroy_call_record *__local =
            (struct mvnc_nc_fifo_destroy_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncFifoHandle_t **fifoHandle;
            fifoHandle = __local->fifoHandle;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_FIFO_WRITE_ELEM:{
        struct mvnc_nc_fifo_write_elem_ret *__ret = (struct mvnc_nc_fifo_write_elem_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_fifo_write_elem_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_fifo_write_elem_call_record *__local =
            (struct mvnc_nc_fifo_write_elem_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncFifoHandle_t *fifoHandle;
            fifoHandle = __local->fifoHandle;

            unsigned int *inputTensorLength;
            inputTensorLength = __local->inputTensorLength;

            void *userParam;
            userParam = __local->userParam;

            void *inputTensor;
            inputTensor = __local->inputTensor;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: unsigned int * inputTensorLength */
            {
                if ((__ret->inputTensorLength) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    unsigned int *__src_inputTensorLength_0;
                    __src_inputTensorLength_0 =
                        (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __ret->inputTensorLength);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->inputTensorLength == NULL);
                    memcpy(__local->inputTensorLength, __src_inputTensorLength_0, __buffer_size * sizeof(unsigned int));
                }
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_FIFO_READ_ELEM:{
        struct mvnc_nc_fifo_read_elem_ret *__ret = (struct mvnc_nc_fifo_read_elem_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_fifo_read_elem_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_fifo_read_elem_call_record *__local =
            (struct mvnc_nc_fifo_read_elem_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncFifoHandle_t *fifoHandle;
            fifoHandle = __local->fifoHandle;

            unsigned int *outputDataLen;
            outputDataLen = __local->outputDataLen;

            void **userParam;
            userParam = __local->userParam;

            void *outputData;
            outputData = __local->outputData;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: unsigned int * outputDataLen */
            {
                if ((__ret->outputDataLen) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    unsigned int *__src_outputDataLen_0;
                    __src_outputDataLen_0 =
                        (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __ret->outputDataLen);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->outputDataLen == NULL);
                    memcpy(__local->outputDataLen, __src_outputDataLen_0, __buffer_size * sizeof(unsigned int));
                }
            }

            /* Output: void ** userParam */
            {
                if ((__ret->userParam) != (NULL)) {
                    volatile size_t __buffer_size = 0;
                    void **__src_userParam_0;
                    __src_userParam_0 = (void **)__chan->chan_get_buffer(__chan, __cmd, __ret->userParam);
                    __buffer_size = ((size_t) (1));
                    BUG_ON(__local->userParam == NULL);
                    memcpy(__local->userParam, __src_userParam_0, __buffer_size * sizeof(void *));
                }
            }

            /* Output: void * outputData */
            {
                if ((__ret->outputData) != (NULL)) {
                    if (kava_shm_offset(__local->outputData) >= 0) {
                    } else {
                        volatile size_t __buffer_size = 0;
                        void *__src_outputData_0;
                        __src_outputData_0 = (void *)__chan->chan_get_buffer(__chan, __cmd, __ret->outputData);
                        __buffer_size = ((size_t) (*outputDataLen));
                        BUG_ON(__local->outputData == NULL);
                        memcpy(__local->outputData, __src_outputData_0, __buffer_size * sizeof(void));
                }}
            }

            /* Output: ncStatus_t ret */
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
    case RET_MVNC_NC_FIFO_REMOVE_ELEM:{
        struct mvnc_nc_fifo_remove_elem_ret *__ret = (struct mvnc_nc_fifo_remove_elem_ret *)__cmd;
        BUG_ON(__ret->base.mode != KAVA_CMD_MODE_API);
        BUG_ON(__ret->base.command_size != sizeof(struct mvnc_nc_fifo_remove_elem_ret)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");
        struct mvnc_nc_fifo_remove_elem_call_record *__local =
            (struct mvnc_nc_fifo_remove_elem_call_record *)kava_remove_call(&__kava_endpoint, __ret->__call_id);

        {

            struct ncFifoHandle_t *fifoHandle;
            fifoHandle = __local->fifoHandle;

            ncStatus_t ret;
            ret = (ncStatus_t) __ret->ret;

            /* Output: ncStatus_t ret */
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
__print_command_mvnc(FILE * file, const struct kava_chan *__chan, const struct kava_cmd_base *__cmd)
{
    switch (__cmd->command_id) {
    case CALL_MVNC_NC_GLOBAL_SET_OPTION:{
        pr_info("ncGlobalSetOption is called \n");
        break;
    }
    case RET_MVNC_NC_GLOBAL_SET_OPTION:{
        pr_info("ncGlobalSetOption is responded\n");
        break;
    }
    case CALL_MVNC_NC_GLOBAL_GET_OPTION:{
        pr_info("ncGlobalGetOption is called \n");
        break;
    }
    case RET_MVNC_NC_GLOBAL_GET_OPTION:{
        pr_info("ncGlobalGetOption is responded\n");
        break;
    }
    case CALL_MVNC_NC_DEVICE_SET_OPTION:{
        pr_info("ncDeviceSetOption is called \n");
        break;
    }
    case RET_MVNC_NC_DEVICE_SET_OPTION:{
        pr_info("ncDeviceSetOption is responded\n");
        break;
    }
    case CALL_MVNC_NC_DEVICE_GET_OPTION:{
        pr_info("ncDeviceGetOption is called \n");
        break;
    }
    case RET_MVNC_NC_DEVICE_GET_OPTION:{
        pr_info("ncDeviceGetOption is responded\n");
        break;
    }
    case CALL_MVNC_NC_DEVICE_CREATE:{
        pr_info("ncDeviceCreate is called \n");
        break;
    }
    case RET_MVNC_NC_DEVICE_CREATE:{
        pr_info("ncDeviceCreate is responded\n");
        break;
    }
    case CALL_MVNC_NC_DEVICE_OPEN:{
        pr_info("ncDeviceOpen is called \n");
        break;
    }
    case RET_MVNC_NC_DEVICE_OPEN:{
        pr_info("ncDeviceOpen is responded\n");
        break;
    }
    case CALL_MVNC_NC_DEVICE_CLOSE:{
        pr_info("ncDeviceClose is called \n");
        break;
    }
    case RET_MVNC_NC_DEVICE_CLOSE:{
        pr_info("ncDeviceClose is responded\n");
        break;
    }
    case CALL_MVNC_NC_DEVICE_DESTROY:{
        pr_info("ncDeviceDestroy is called \n");
        break;
    }
    case RET_MVNC_NC_DEVICE_DESTROY:{
        pr_info("ncDeviceDestroy is responded\n");
        break;
    }
    case CALL_MVNC_NC_GRAPH_CREATE:{
        pr_info("ncGraphCreate is called \n");
        break;
    }
    case RET_MVNC_NC_GRAPH_CREATE:{
        pr_info("ncGraphCreate is responded\n");
        break;
    }
    case CALL_MVNC_NC_GRAPH_ALLOCATE:{
        pr_info("ncGraphAllocate is called \n");
        break;
    }
    case RET_MVNC_NC_GRAPH_ALLOCATE:{
        pr_info("ncGraphAllocate is responded\n");
        break;
    }
    case CALL_MVNC_NC_GRAPH_DESTROY:{
        pr_info("ncGraphDestroy is called \n");
        break;
    }
    case RET_MVNC_NC_GRAPH_DESTROY:{
        pr_info("ncGraphDestroy is responded\n");
        break;
    }
    case CALL_MVNC_NC_GRAPH_SET_OPTION:{
        pr_info("ncGraphSetOption is called \n");
        break;
    }
    case RET_MVNC_NC_GRAPH_SET_OPTION:{
        pr_info("ncGraphSetOption is responded\n");
        break;
    }
    case CALL_MVNC_NC_GRAPH_GET_OPTION:{
        pr_info("ncGraphGetOption is called \n");
        break;
    }
    case RET_MVNC_NC_GRAPH_GET_OPTION:{
        pr_info("ncGraphGetOption is responded\n");
        break;
    }
    case CALL_MVNC_NC_GRAPH_QUEUE_INFERENCE:{
        pr_info("ncGraphQueueInference is called \n");
        break;
    }
    case RET_MVNC_NC_GRAPH_QUEUE_INFERENCE:{
        pr_info("ncGraphQueueInference is responded\n");
        break;
    }
    case CALL_MVNC_NC_GRAPH_QUEUE_INFERENCE_WITH_FIFO_ELEM:{
        pr_info("ncGraphQueueInferenceWithFifoElem is called \n");
        break;
    }
    case RET_MVNC_NC_GRAPH_QUEUE_INFERENCE_WITH_FIFO_ELEM:{
        pr_info("ncGraphQueueInferenceWithFifoElem is responded\n");
        break;
    }
    case CALL_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS:{
        pr_info("ncGraphAllocateWithFifos is called \n");
        break;
    }
    case RET_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS:{
        pr_info("ncGraphAllocateWithFifos is responded\n");
        break;
    }
    case CALL_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS_EX:{
        pr_info("ncGraphAllocateWithFifosEx is called \n");
        break;
    }
    case RET_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS_EX:{
        pr_info("ncGraphAllocateWithFifosEx is responded\n");
        break;
    }
    case CALL_MVNC_NC_FIFO_CREATE:{
        pr_info("ncFifoCreate is called \n");
        break;
    }
    case RET_MVNC_NC_FIFO_CREATE:{
        pr_info("ncFifoCreate is responded\n");
        break;
    }
    case CALL_MVNC_NC_FIFO_ALLOCATE:{
        pr_info("ncFifoAllocate is called \n");
        break;
    }
    case RET_MVNC_NC_FIFO_ALLOCATE:{
        pr_info("ncFifoAllocate is responded\n");
        break;
    }
    case CALL_MVNC_NC_FIFO_SET_OPTION:{
        pr_info("ncFifoSetOption is called \n");
        break;
    }
    case RET_MVNC_NC_FIFO_SET_OPTION:{
        pr_info("ncFifoSetOption is responded\n");
        break;
    }
    case CALL_MVNC_NC_FIFO_GET_OPTION:{
        pr_info("ncFifoGetOption is called \n");
        break;
    }
    case RET_MVNC_NC_FIFO_GET_OPTION:{
        pr_info("ncFifoGetOption is responded\n");
        break;
    }
    case CALL_MVNC_NC_FIFO_DESTROY:{
        pr_info("ncFifoDestroy is called \n");
        break;
    }
    case RET_MVNC_NC_FIFO_DESTROY:{
        pr_info("ncFifoDestroy is responded\n");
        break;
    }
    case CALL_MVNC_NC_FIFO_WRITE_ELEM:{
        pr_info("ncFifoWriteElem is called \n");
        break;
    }
    case RET_MVNC_NC_FIFO_WRITE_ELEM:{
        pr_info("ncFifoWriteElem is responded\n");
        break;
    }
    case CALL_MVNC_NC_FIFO_READ_ELEM:{
        pr_info("ncFifoReadElem is called \n");
        break;
    }
    case RET_MVNC_NC_FIFO_READ_ELEM:{
        pr_info("ncFifoReadElem is responded\n");
        break;
    }
    case CALL_MVNC_NC_FIFO_REMOVE_ELEM:{
        pr_info("ncFifoRemoveElem is called \n");
        break;
    }
    case RET_MVNC_NC_FIFO_REMOVE_ELEM:{
        pr_info("ncFifoRemoveElem is responded\n");
        break;
    }
    default:
        pr_err("Received unsupported command");
    }                                            // switch
}

////// API function stub implementations

#define __chan nw_global_command_channel

EXPORTED dllexport ncStatus_t
ncGlobalSetOption(int option, const void *data, unsigned int dataLength)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGlobalSetOption = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * data */
        if ((data) != (NULL) && (dataLength) > (0)) {
            if (kava_shm_offset(data) >= 0) {
            } else {
                __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (dataLength)) * sizeof(const void));
        }}
    }
    struct mvnc_nc_global_set_option_call *__cmd =
        (struct mvnc_nc_global_set_option_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_global_set_option_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GLOBAL_SET_OPTION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: int option */
        {
            __cmd->option = option;
        }
        /* Input: const void * data */
        {
            if ((data) != (NULL) && (dataLength) > (0)) {
                if (kava_shm_offset(data) >= 0) {
                    __cmd->data = (void *)kava_shm_offset(data);
                    __cmd->__shm_data = 1;
                } else {
                    __cmd->__shm_data = 0;

                    __cmd->data =
                        (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, data,
                        ((size_t) (dataLength)) * sizeof(const void));
            }} else {
                __cmd->data = NULL;
            }
        }
        /* Input: unsigned int dataLength */
        {
            __cmd->dataLength = dataLength;
        }
    }

    struct mvnc_nc_global_set_option_call_record *__call_record =
        (struct mvnc_nc_global_set_option_call_record *)vmalloc(sizeof(struct mvnc_nc_global_set_option_call_record));

    __call_record->option = option;

    __call_record->dataLength = dataLength;

    __call_record->data = data;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGlobalSetOption);

    return NC_OK;
}

EXPORT_SYMBOL(ncGlobalSetOption);

EXPORTED dllexport ncStatus_t
ncGlobalGetOption(int option, void *data, unsigned int *dataLength)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGlobalGetOption = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: unsigned int * dataLength */
        if ((dataLength) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(unsigned int));
        }

        /* Size: void * data */
        if ((data) != (NULL) && (*dataLength) > (0)) {
            if (kava_shm_offset(data) >= 0) {
            } else {

            }
        }
    }
    struct mvnc_nc_global_get_option_call *__cmd =
        (struct mvnc_nc_global_get_option_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_global_get_option_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GLOBAL_GET_OPTION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: int option */
        {
            __cmd->option = option;
        }
        /* Input: void * data */
        {
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (kava_shm_offset(data) >= 0) {
                    __cmd->data = (void *)kava_shm_offset(data);
                    __cmd->__shm_data = 1;
                } else {
                    __cmd->__shm_data = 0;

                    __cmd->data = HAS_OUT_BUFFER_SENTINEL;
                }
            } else {
                __cmd->data = NULL;
            }
        }
        /* Input: unsigned int * dataLength */
        {
            if ((dataLength) != (NULL)) {
                __cmd->dataLength =
                    (unsigned int *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, dataLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __cmd->dataLength = NULL;
            }
        }
    }

    struct mvnc_nc_global_get_option_call_record *__call_record =
        (struct mvnc_nc_global_get_option_call_record *)vmalloc(sizeof(struct mvnc_nc_global_get_option_call_record));

    __call_record->option = option;

    __call_record->dataLength = dataLength;

    __call_record->data = data;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGlobalGetOption);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncGlobalGetOption);

EXPORTED dllexport ncStatus_t
ncDeviceSetOption(struct ncDeviceHandle_t * deviceHandle, int option, const void *data, unsigned int dataLength)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncDeviceSetOption = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * data */
        if ((data) != (NULL) && (dataLength) > (0)) {
            if (kava_shm_offset(data) >= 0) {
            } else {
                __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (dataLength)) * sizeof(const void));
        }}
    }
    struct mvnc_nc_device_set_option_call *__cmd =
        (struct mvnc_nc_device_set_option_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_device_set_option_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_DEVICE_SET_OPTION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        {
            __cmd->deviceHandle = deviceHandle;
        }
        /* Input: int option */
        {
            __cmd->option = option;
        }
        /* Input: const void * data */
        {
            if ((data) != (NULL) && (dataLength) > (0)) {
                if (kava_shm_offset(data) >= 0) {
                    __cmd->data = (void *)kava_shm_offset(data);
                    __cmd->__shm_data = 1;
                } else {
                    __cmd->__shm_data = 0;

                    __cmd->data =
                        (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, data,
                        ((size_t) (dataLength)) * sizeof(const void));
            }} else {
                __cmd->data = NULL;
            }
        }
        /* Input: unsigned int dataLength */
        {
            __cmd->dataLength = dataLength;
        }
    }

    struct mvnc_nc_device_set_option_call_record *__call_record =
        (struct mvnc_nc_device_set_option_call_record *)vmalloc(sizeof(struct mvnc_nc_device_set_option_call_record));

    __call_record->deviceHandle = deviceHandle;

    __call_record->option = option;

    __call_record->dataLength = dataLength;

    __call_record->data = data;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncDeviceSetOption);

    return NC_OK;
}

EXPORT_SYMBOL(ncDeviceSetOption);

EXPORTED dllexport ncStatus_t
ncDeviceGetOption(struct ncDeviceHandle_t * deviceHandle, int option, void *data, unsigned int *dataLength)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncDeviceGetOption = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: unsigned int * dataLength */
        if ((dataLength) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(unsigned int));
        }

        /* Size: void * data */
        if ((data) != (NULL) && (*dataLength) > (0)) {
            if (kava_shm_offset(data) >= 0) {
            } else {

            }
        }
    }
    struct mvnc_nc_device_get_option_call *__cmd =
        (struct mvnc_nc_device_get_option_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_device_get_option_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_DEVICE_GET_OPTION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        {
            __cmd->deviceHandle = deviceHandle;
        }
        /* Input: int option */
        {
            __cmd->option = option;
        }
        /* Input: void * data */
        {
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (kava_shm_offset(data) >= 0) {
                    __cmd->data = (void *)kava_shm_offset(data);
                    __cmd->__shm_data = 1;
                } else {
                    __cmd->__shm_data = 0;

                    __cmd->data = HAS_OUT_BUFFER_SENTINEL;
                }
            } else {
                __cmd->data = NULL;
            }
        }
        /* Input: unsigned int * dataLength */
        {
            if ((dataLength) != (NULL)) {
                __cmd->dataLength =
                    (unsigned int *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, dataLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __cmd->dataLength = NULL;
            }
        }
    }

    struct mvnc_nc_device_get_option_call_record *__call_record =
        (struct mvnc_nc_device_get_option_call_record *)vmalloc(sizeof(struct mvnc_nc_device_get_option_call_record));

    __call_record->deviceHandle = deviceHandle;

    __call_record->option = option;

    __call_record->dataLength = dataLength;

    __call_record->data = data;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncDeviceGetOption);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncDeviceGetOption);

EXPORTED dllexport ncStatus_t
ncDeviceCreate(int index, struct ncDeviceHandle_t ** deviceHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncDeviceCreate = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
    }
    struct mvnc_nc_device_create_call *__cmd =
        (struct mvnc_nc_device_create_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_device_create_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_DEVICE_CREATE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: int index */
        {
            __cmd->index = index;
        }
        /* Input: struct ncDeviceHandle_t ** deviceHandle */
        {
            if ((deviceHandle) != (NULL)) {
                __cmd->deviceHandle = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->deviceHandle = NULL;
            }
        }
    }

    struct mvnc_nc_device_create_call_record *__call_record =
        (struct mvnc_nc_device_create_call_record *)vmalloc(sizeof(struct mvnc_nc_device_create_call_record));

    __call_record->index = index;

    __call_record->deviceHandle = deviceHandle;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncDeviceCreate);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncDeviceCreate);

EXPORTED dllexport ncStatus_t
ncDeviceOpen(struct ncDeviceHandle_t * deviceHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncDeviceOpen = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
    }
    struct mvnc_nc_device_open_call *__cmd =
        (struct mvnc_nc_device_open_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_device_open_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_DEVICE_OPEN;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        {
            __cmd->deviceHandle = deviceHandle;
        }
    }

    struct mvnc_nc_device_open_call_record *__call_record =
        (struct mvnc_nc_device_open_call_record *)vmalloc(sizeof(struct mvnc_nc_device_open_call_record));

    __call_record->deviceHandle = deviceHandle;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncDeviceOpen);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncDeviceOpen);

EXPORTED dllexport ncStatus_t
ncDeviceClose(struct ncDeviceHandle_t * deviceHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncDeviceClose = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
    }
    struct mvnc_nc_device_close_call *__cmd =
        (struct mvnc_nc_device_close_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_device_close_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_DEVICE_CLOSE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        {
            __cmd->deviceHandle = deviceHandle;
        }
    }

    struct mvnc_nc_device_close_call_record *__call_record =
        (struct mvnc_nc_device_close_call_record *)vmalloc(sizeof(struct mvnc_nc_device_close_call_record));

    __call_record->deviceHandle = deviceHandle;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncDeviceClose);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncDeviceClose);

EXPORTED dllexport ncStatus_t
ncDeviceDestroy(struct ncDeviceHandle_t ** deviceHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncDeviceDestroy = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: struct ncDeviceHandle_t ** deviceHandle */
        if ((deviceHandle) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(struct ncDeviceHandle_t *));
        }
    }
    struct mvnc_nc_device_destroy_call *__cmd =
        (struct mvnc_nc_device_destroy_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_device_destroy_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_DEVICE_DESTROY;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncDeviceHandle_t ** deviceHandle */
        {
            if ((deviceHandle) != (NULL)) {
                __cmd->deviceHandle =
                    (struct ncDeviceHandle_t **)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd,
                    deviceHandle, ((size_t) (1)) * sizeof(struct ncDeviceHandle_t *));
            } else {
                __cmd->deviceHandle = NULL;
            }
        }
    }

    struct mvnc_nc_device_destroy_call_record *__call_record =
        (struct mvnc_nc_device_destroy_call_record *)vmalloc(sizeof(struct mvnc_nc_device_destroy_call_record));

    __call_record->deviceHandle = deviceHandle;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncDeviceDestroy);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncDeviceDestroy);

EXPORTED dllexport ncStatus_t
ncGraphCreate(const char *ava_name, struct ncGraphHandle_t ** graphHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGraphCreate = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const char * ava_name */
        if ((ava_name) != (NULL) && (strlen(ava_name) + 1) > (0)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (strlen(ava_name) + 1)) * sizeof(const char));
        }
    }
    struct mvnc_nc_graph_create_call *__cmd =
        (struct mvnc_nc_graph_create_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_graph_create_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GRAPH_CREATE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: const char * ava_name */
        {
            if ((ava_name) != (NULL) && (strlen(ava_name) + 1) > (0)) {
                __cmd->ava_name =
                    (char *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, ava_name,
                    ((size_t) (strlen(ava_name) + 1)) * sizeof(const char));
            } else {
                __cmd->ava_name = NULL;
            }
        }
        /* Input: struct ncGraphHandle_t ** graphHandle */
        {
            if ((graphHandle) != (NULL)) {
                __cmd->graphHandle = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->graphHandle = NULL;
            }
        }
    }

    struct mvnc_nc_graph_create_call_record *__call_record =
        (struct mvnc_nc_graph_create_call_record *)vmalloc(sizeof(struct mvnc_nc_graph_create_call_record));

    __call_record->ava_name = ava_name;

    __call_record->graphHandle = graphHandle;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGraphCreate);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncGraphCreate);

EXPORTED dllexport ncStatus_t
ncGraphAllocate(struct ncDeviceHandle_t * deviceHandle, struct ncGraphHandle_t * graphHandle, const void *graphBuffer,
    unsigned int graphBufferLength)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGraphAllocate = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * graphBuffer */
        if ((graphBuffer) != (NULL) && (graphBufferLength) > (0)) {
            if (kava_shm_offset(graphBuffer) >= 0) {
            } else {
                __total_buffer_size +=
                    chan->chan_buffer_size(chan, ((size_t) (graphBufferLength)) * sizeof(const void));
        }}
    }
    struct mvnc_nc_graph_allocate_call *__cmd =
        (struct mvnc_nc_graph_allocate_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_graph_allocate_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GRAPH_ALLOCATE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        {
            __cmd->deviceHandle = deviceHandle;
        }
        /* Input: struct ncGraphHandle_t * graphHandle */
        {
            __cmd->graphHandle = graphHandle;
        }
        /* Input: const void * graphBuffer */
        {
            if ((graphBuffer) != (NULL) && (graphBufferLength) > (0)) {
                if (kava_shm_offset(graphBuffer) >= 0) {
                    __cmd->graphBuffer = (void *)kava_shm_offset(graphBuffer);
                    __cmd->__shm_graphBuffer = 1;
                } else {
                    __cmd->__shm_graphBuffer = 0;

                    __cmd->graphBuffer =
                        (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, graphBuffer,
                        ((size_t) (graphBufferLength)) * sizeof(const void));
            }} else {
                __cmd->graphBuffer = NULL;
            }
        }
        /* Input: unsigned int graphBufferLength */
        {
            __cmd->graphBufferLength = graphBufferLength;
        }
    }

    struct mvnc_nc_graph_allocate_call_record *__call_record =
        (struct mvnc_nc_graph_allocate_call_record *)vmalloc(sizeof(struct mvnc_nc_graph_allocate_call_record));

    __call_record->deviceHandle = deviceHandle;

    __call_record->graphHandle = graphHandle;

    __call_record->graphBufferLength = graphBufferLength;

    __call_record->graphBuffer = graphBuffer;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGraphAllocate);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncGraphAllocate);

EXPORTED dllexport ncStatus_t
ncGraphDestroy(struct ncGraphHandle_t ** graphHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGraphDestroy = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: struct ncGraphHandle_t ** graphHandle */
        if ((graphHandle) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(struct ncGraphHandle_t *));
        }
    }
    struct mvnc_nc_graph_destroy_call *__cmd =
        (struct mvnc_nc_graph_destroy_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_graph_destroy_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GRAPH_DESTROY;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncGraphHandle_t ** graphHandle */
        {
            if ((graphHandle) != (NULL)) {
                __cmd->graphHandle =
                    (struct ncGraphHandle_t **)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd,
                    graphHandle, ((size_t) (1)) * sizeof(struct ncGraphHandle_t *));
            } else {
                __cmd->graphHandle = NULL;
            }
        }
    }

    struct mvnc_nc_graph_destroy_call_record *__call_record =
        (struct mvnc_nc_graph_destroy_call_record *)vmalloc(sizeof(struct mvnc_nc_graph_destroy_call_record));

    __call_record->graphHandle = graphHandle;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGraphDestroy);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncGraphDestroy);

EXPORTED dllexport ncStatus_t
ncGraphSetOption(struct ncGraphHandle_t * graphHandle, int option, const void *data, unsigned int dataLength)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGraphSetOption = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * data */
        if ((data) != (NULL) && (dataLength) > (0)) {
            if (kava_shm_offset(data) >= 0) {
            } else {
                __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (dataLength)) * sizeof(const void));
        }}
    }
    struct mvnc_nc_graph_set_option_call *__cmd =
        (struct mvnc_nc_graph_set_option_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_graph_set_option_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GRAPH_SET_OPTION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncGraphHandle_t * graphHandle */
        {
            __cmd->graphHandle = graphHandle;
        }
        /* Input: int option */
        {
            __cmd->option = option;
        }
        /* Input: const void * data */
        {
            if ((data) != (NULL) && (dataLength) > (0)) {
                if (kava_shm_offset(data) >= 0) {
                    __cmd->data = (void *)kava_shm_offset(data);
                    __cmd->__shm_data = 1;
                } else {
                    __cmd->__shm_data = 0;

                    __cmd->data =
                        (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, data,
                        ((size_t) (dataLength)) * sizeof(const void));
            }} else {
                __cmd->data = NULL;
            }
        }
        /* Input: unsigned int dataLength */
        {
            __cmd->dataLength = dataLength;
        }
    }

    struct mvnc_nc_graph_set_option_call_record *__call_record =
        (struct mvnc_nc_graph_set_option_call_record *)vmalloc(sizeof(struct mvnc_nc_graph_set_option_call_record));

    __call_record->graphHandle = graphHandle;

    __call_record->option = option;

    __call_record->dataLength = dataLength;

    __call_record->data = data;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGraphSetOption);

    return NC_OK;
}

EXPORT_SYMBOL(ncGraphSetOption);

EXPORTED dllexport ncStatus_t
ncGraphGetOption(struct ncGraphHandle_t * graphHandle, int option, void *data, unsigned int *dataLength)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGraphGetOption = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: unsigned int * dataLength */
        if ((dataLength) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(unsigned int));
        }

        /* Size: void * data */
        if ((data) != (NULL) && (*dataLength) > (0)) {
            if (kava_shm_offset(data) >= 0) {
            } else {

            }
        }
    }
    struct mvnc_nc_graph_get_option_call *__cmd =
        (struct mvnc_nc_graph_get_option_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_graph_get_option_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GRAPH_GET_OPTION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncGraphHandle_t * graphHandle */
        {
            __cmd->graphHandle = graphHandle;
        }
        /* Input: int option */
        {
            __cmd->option = option;
        }
        /* Input: void * data */
        {
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (kava_shm_offset(data) >= 0) {
                    __cmd->data = (void *)kava_shm_offset(data);
                    __cmd->__shm_data = 1;
                } else {
                    __cmd->__shm_data = 0;

                    __cmd->data = HAS_OUT_BUFFER_SENTINEL;
                }
            } else {
                __cmd->data = NULL;
            }
        }
        /* Input: unsigned int * dataLength */
        {
            if ((dataLength) != (NULL)) {
                __cmd->dataLength =
                    (unsigned int *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, dataLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __cmd->dataLength = NULL;
            }
        }
    }

    struct mvnc_nc_graph_get_option_call_record *__call_record =
        (struct mvnc_nc_graph_get_option_call_record *)vmalloc(sizeof(struct mvnc_nc_graph_get_option_call_record));

    __call_record->graphHandle = graphHandle;

    __call_record->option = option;

    __call_record->dataLength = dataLength;

    __call_record->data = data;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGraphGetOption);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncGraphGetOption);

EXPORTED dllexport ncStatus_t
ncGraphQueueInference(struct ncGraphHandle_t * graphHandle, struct ncFifoHandle_t ** fifoIn, unsigned int inFifoCount,
    struct ncFifoHandle_t ** fifoOut, unsigned int outFifoCount)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGraphQueueInference = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: struct ncFifoHandle_t ** fifoIn */
        if ((fifoIn) != (NULL) && (inFifoCount) > (0)) {
            __total_buffer_size +=
                chan->chan_buffer_size(chan, ((size_t) (inFifoCount)) * sizeof(struct ncFifoHandle_t *));
        }

        /* Size: struct ncFifoHandle_t ** fifoOut */
        if ((fifoOut) != (NULL) && (outFifoCount) > (0)) {
            __total_buffer_size +=
                chan->chan_buffer_size(chan, ((size_t) (outFifoCount)) * sizeof(struct ncFifoHandle_t *));
        }
    }
    struct mvnc_nc_graph_queue_inference_call *__cmd =
        (struct mvnc_nc_graph_queue_inference_call *)chan->cmd_new(chan,
        sizeof(struct mvnc_nc_graph_queue_inference_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GRAPH_QUEUE_INFERENCE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncGraphHandle_t * graphHandle */
        {
            __cmd->graphHandle = graphHandle;
        }
        /* Input: struct ncFifoHandle_t ** fifoIn */
        {
            if ((fifoIn) != (NULL) && (inFifoCount) > (0)) {
                __cmd->fifoIn =
                    (struct ncFifoHandle_t **)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, fifoIn,
                    ((size_t) (inFifoCount)) * sizeof(struct ncFifoHandle_t *));
            } else {
                __cmd->fifoIn = NULL;
            }
        }
        /* Input: unsigned int inFifoCount */
        {
            __cmd->inFifoCount = inFifoCount;
        }
        /* Input: struct ncFifoHandle_t ** fifoOut */
        {
            if ((fifoOut) != (NULL) && (outFifoCount) > (0)) {
                __cmd->fifoOut =
                    (struct ncFifoHandle_t **)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, fifoOut,
                    ((size_t) (outFifoCount)) * sizeof(struct ncFifoHandle_t *));
            } else {
                __cmd->fifoOut = NULL;
            }
        }
        /* Input: unsigned int outFifoCount */
        {
            __cmd->outFifoCount = outFifoCount;
        }
    }

    struct mvnc_nc_graph_queue_inference_call_record *__call_record =
        (struct mvnc_nc_graph_queue_inference_call_record *)vmalloc(sizeof(struct
            mvnc_nc_graph_queue_inference_call_record));

    __call_record->graphHandle = graphHandle;

    __call_record->inFifoCount = inFifoCount;

    __call_record->outFifoCount = outFifoCount;

    __call_record->fifoIn = fifoIn;

    __call_record->fifoOut = fifoOut;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGraphQueueInference);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncGraphQueueInference);

EXPORTED dllexport ncStatus_t
ncGraphQueueInferenceWithFifoElem(struct ncGraphHandle_t * graphHandle, struct ncFifoHandle_t * fifoIn,
    struct ncFifoHandle_t * fifoOut, const void *inputTensor, unsigned int *inputTensorLength, void *userParam)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGraphQueueInferenceWithFifoElem = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: unsigned int * inputTensorLength */
        if ((inputTensorLength) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(unsigned int));
        }

        /* Size: const void * inputTensor */
        if ((inputTensor) != (NULL) && (*inputTensorLength) > (0)) {
            if (kava_shm_offset(inputTensor) >= 0) {
            } else {
                __total_buffer_size +=
                    chan->chan_buffer_size(chan, ((size_t) (*inputTensorLength)) * sizeof(const void));
        }}
    }
    struct mvnc_nc_graph_queue_inference_with_fifo_elem_call *__cmd =
        (struct mvnc_nc_graph_queue_inference_with_fifo_elem_call *)chan->cmd_new(chan,
        sizeof(struct mvnc_nc_graph_queue_inference_with_fifo_elem_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GRAPH_QUEUE_INFERENCE_WITH_FIFO_ELEM;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncGraphHandle_t * graphHandle */
        {
            __cmd->graphHandle = graphHandle;
        }
        /* Input: struct ncFifoHandle_t * fifoIn */
        {
            __cmd->fifoIn = fifoIn;
        }
        /* Input: struct ncFifoHandle_t * fifoOut */
        {
            __cmd->fifoOut = fifoOut;
        }
        /* Input: const void * inputTensor */
        {
            if ((inputTensor) != (NULL) && (*inputTensorLength) > (0)) {
                if (kava_shm_offset(inputTensor) >= 0) {
                    __cmd->inputTensor = (void *)kava_shm_offset(inputTensor);
                    __cmd->__shm_inputTensor = 1;
                } else {
                    __cmd->__shm_inputTensor = 0;

                    __cmd->inputTensor =
                        (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, inputTensor,
                        ((size_t) (*inputTensorLength)) * sizeof(const void));
            }} else {
                __cmd->inputTensor = NULL;
            }
        }
        /* Input: unsigned int * inputTensorLength */
        {
            if ((inputTensorLength) != (NULL)) {
                __cmd->inputTensorLength =
                    (unsigned int *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, inputTensorLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __cmd->inputTensorLength = NULL;
            }
        }
        /* Input: void * userParam */
        {
            __cmd->userParam = userParam;
        }
    }

    struct mvnc_nc_graph_queue_inference_with_fifo_elem_call_record *__call_record =
        (struct mvnc_nc_graph_queue_inference_with_fifo_elem_call_record *)vmalloc(sizeof(struct
            mvnc_nc_graph_queue_inference_with_fifo_elem_call_record));

    __call_record->graphHandle = graphHandle;

    __call_record->fifoIn = fifoIn;

    __call_record->fifoOut = fifoOut;

    __call_record->inputTensorLength = inputTensorLength;

    __call_record->userParam = userParam;

    __call_record->inputTensor = inputTensor;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGraphQueueInferenceWithFifoElem);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncGraphQueueInferenceWithFifoElem);

EXPORTED dllexport ncStatus_t
ncGraphAllocateWithFifos(struct ncDeviceHandle_t * deviceHandle, struct ncGraphHandle_t * graphHandle,
    const void *graphBuffer, unsigned int graphBufferLength, struct ncFifoHandle_t ** inFifoHandle,
    struct ncFifoHandle_t ** outFifoHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGraphAllocateWithFifos = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * graphBuffer */
        if ((graphBuffer) != (NULL) && (graphBufferLength) > (0)) {
            if (kava_shm_offset(graphBuffer) >= 0) {
            } else {
                __total_buffer_size +=
                    chan->chan_buffer_size(chan, ((size_t) (graphBufferLength)) * sizeof(const void));
        }}
    }
    struct mvnc_nc_graph_allocate_with_fifos_call *__cmd =
        (struct mvnc_nc_graph_allocate_with_fifos_call *)chan->cmd_new(chan,
        sizeof(struct mvnc_nc_graph_allocate_with_fifos_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        {
            __cmd->deviceHandle = deviceHandle;
        }
        /* Input: struct ncGraphHandle_t * graphHandle */
        {
            __cmd->graphHandle = graphHandle;
        }
        /* Input: const void * graphBuffer */
        {
            if ((graphBuffer) != (NULL) && (graphBufferLength) > (0)) {
                if (kava_shm_offset(graphBuffer) >= 0) {
                    __cmd->graphBuffer = (void *)kava_shm_offset(graphBuffer);
                    __cmd->__shm_graphBuffer = 1;
                } else {
                    __cmd->__shm_graphBuffer = 0;

                    __cmd->graphBuffer =
                        (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, graphBuffer,
                        ((size_t) (graphBufferLength)) * sizeof(const void));
            }} else {
                __cmd->graphBuffer = NULL;
            }
        }
        /* Input: unsigned int graphBufferLength */
        {
            __cmd->graphBufferLength = graphBufferLength;
        }
        /* Input: struct ncFifoHandle_t ** inFifoHandle */
        {
            if ((inFifoHandle) != (NULL)) {
                __cmd->inFifoHandle = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->inFifoHandle = NULL;
            }
        }
        /* Input: struct ncFifoHandle_t ** outFifoHandle */
        {
            if ((outFifoHandle) != (NULL)) {
                __cmd->outFifoHandle = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->outFifoHandle = NULL;
            }
        }
    }

    struct mvnc_nc_graph_allocate_with_fifos_call_record *__call_record =
        (struct mvnc_nc_graph_allocate_with_fifos_call_record *)vmalloc(sizeof(struct
            mvnc_nc_graph_allocate_with_fifos_call_record));

    __call_record->deviceHandle = deviceHandle;

    __call_record->graphHandle = graphHandle;

    __call_record->graphBufferLength = graphBufferLength;

    __call_record->inFifoHandle = inFifoHandle;

    __call_record->outFifoHandle = outFifoHandle;

    __call_record->graphBuffer = graphBuffer;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGraphAllocateWithFifos);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncGraphAllocateWithFifos);

EXPORTED dllexport ncStatus_t
ncGraphAllocateWithFifosEx(struct ncDeviceHandle_t * deviceHandle, struct ncGraphHandle_t * graphHandle,
    const void *graphBuffer, unsigned int graphBufferLength, struct ncFifoHandle_t ** inFifoHandle,
    ncFifoType_t inFifoType, int inNumElem, ncFifoDataType_t inDataType, struct ncFifoHandle_t ** outFifoHandle,
    ncFifoType_t outFifoType, int outNumElem, ncFifoDataType_t outDataType)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncGraphAllocateWithFifosEx = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * graphBuffer */
        if ((graphBuffer) != (NULL) && (graphBufferLength) > (0)) {
            if (kava_shm_offset(graphBuffer) >= 0) {
            } else {
                __total_buffer_size +=
                    chan->chan_buffer_size(chan, ((size_t) (graphBufferLength)) * sizeof(const void));
        }}
    }
    struct mvnc_nc_graph_allocate_with_fifos_ex_call *__cmd =
        (struct mvnc_nc_graph_allocate_with_fifos_ex_call *)chan->cmd_new(chan,
        sizeof(struct mvnc_nc_graph_allocate_with_fifos_ex_call), __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS_EX;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        {
            __cmd->deviceHandle = deviceHandle;
        }
        /* Input: struct ncGraphHandle_t * graphHandle */
        {
            __cmd->graphHandle = graphHandle;
        }
        /* Input: const void * graphBuffer */
        {
            if ((graphBuffer) != (NULL) && (graphBufferLength) > (0)) {
                if (kava_shm_offset(graphBuffer) >= 0) {
                    __cmd->graphBuffer = (void *)kava_shm_offset(graphBuffer);
                    __cmd->__shm_graphBuffer = 1;
                } else {
                    __cmd->__shm_graphBuffer = 0;

                    __cmd->graphBuffer =
                        (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, graphBuffer,
                        ((size_t) (graphBufferLength)) * sizeof(const void));
            }} else {
                __cmd->graphBuffer = NULL;
            }
        }
        /* Input: unsigned int graphBufferLength */
        {
            __cmd->graphBufferLength = graphBufferLength;
        }
        /* Input: struct ncFifoHandle_t ** inFifoHandle */
        {
            if ((inFifoHandle) != (NULL)) {
                __cmd->inFifoHandle = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->inFifoHandle = NULL;
            }
        }
        /* Input: ncFifoType_t inFifoType */
        {
            __cmd->inFifoType = inFifoType;
        }
        /* Input: int inNumElem */
        {
            __cmd->inNumElem = inNumElem;
        }
        /* Input: ncFifoDataType_t inDataType */
        {
            __cmd->inDataType = inDataType;
        }
        /* Input: struct ncFifoHandle_t ** outFifoHandle */
        {
            if ((outFifoHandle) != (NULL)) {
                __cmd->outFifoHandle = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->outFifoHandle = NULL;
            }
        }
        /* Input: ncFifoType_t outFifoType */
        {
            __cmd->outFifoType = outFifoType;
        }
        /* Input: int outNumElem */
        {
            __cmd->outNumElem = outNumElem;
        }
        /* Input: ncFifoDataType_t outDataType */
        {
            __cmd->outDataType = outDataType;
        }
    }

    struct mvnc_nc_graph_allocate_with_fifos_ex_call_record *__call_record =
        (struct mvnc_nc_graph_allocate_with_fifos_ex_call_record *)vmalloc(sizeof(struct
            mvnc_nc_graph_allocate_with_fifos_ex_call_record));

    __call_record->deviceHandle = deviceHandle;

    __call_record->graphHandle = graphHandle;

    __call_record->graphBufferLength = graphBufferLength;

    __call_record->inFifoHandle = inFifoHandle;

    __call_record->inFifoType = inFifoType;

    __call_record->inNumElem = inNumElem;

    __call_record->inDataType = inDataType;

    __call_record->outFifoHandle = outFifoHandle;

    __call_record->outFifoType = outFifoType;

    __call_record->outNumElem = outNumElem;

    __call_record->outDataType = outDataType;

    __call_record->graphBuffer = graphBuffer;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncGraphAllocateWithFifosEx);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncGraphAllocateWithFifosEx);

EXPORTED dllexport ncStatus_t
ncFifoCreate(const char *ava_name, ncFifoType_t ava_type, struct ncFifoHandle_t ** fifoHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncFifoCreate = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const char * ava_name */
        if ((ava_name) != (NULL) && (strlen(ava_name) + 1) > (0)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (strlen(ava_name) + 1)) * sizeof(const char));
        }
    }
    struct mvnc_nc_fifo_create_call *__cmd =
        (struct mvnc_nc_fifo_create_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_fifo_create_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_FIFO_CREATE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: const char * ava_name */
        {
            if ((ava_name) != (NULL) && (strlen(ava_name) + 1) > (0)) {
                __cmd->ava_name =
                    (char *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, ava_name,
                    ((size_t) (strlen(ava_name) + 1)) * sizeof(const char));
            } else {
                __cmd->ava_name = NULL;
            }
        }
        /* Input: ncFifoType_t ava_type */
        {
            __cmd->ava_type = ava_type;
        }
        /* Input: struct ncFifoHandle_t ** fifoHandle */
        {
            if ((fifoHandle) != (NULL)) {
                __cmd->fifoHandle = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->fifoHandle = NULL;
            }
        }
    }

    struct mvnc_nc_fifo_create_call_record *__call_record =
        (struct mvnc_nc_fifo_create_call_record *)vmalloc(sizeof(struct mvnc_nc_fifo_create_call_record));

    __call_record->ava_name = ava_name;

    __call_record->ava_type = ava_type;

    __call_record->fifoHandle = fifoHandle;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncFifoCreate);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncFifoCreate);

EXPORTED dllexport ncStatus_t
ncFifoAllocate(struct ncFifoHandle_t * fifoHandle, struct ncDeviceHandle_t * device,
    struct ncTensorDescriptor_t * tensorDesc, unsigned int numElem)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncFifoAllocate = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: struct ncTensorDescriptor_t * tensorDesc */
        if ((tensorDesc) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(struct ncTensorDescriptor_t));
        }
    }
    struct mvnc_nc_fifo_allocate_call *__cmd =
        (struct mvnc_nc_fifo_allocate_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_fifo_allocate_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_FIFO_ALLOCATE;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncFifoHandle_t * fifoHandle */
        {
            __cmd->fifoHandle = fifoHandle;
        }
        /* Input: struct ncDeviceHandle_t * device */
        {
            __cmd->device = device;
        }
        /* Input: struct ncTensorDescriptor_t * tensorDesc */
        {
            if ((tensorDesc) != (NULL)) {
                __cmd->tensorDesc =
                    (struct ncTensorDescriptor_t *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd,
                    tensorDesc, ((size_t) (1)) * sizeof(struct ncTensorDescriptor_t));
            } else {
                __cmd->tensorDesc = NULL;
            }
        }
        /* Input: unsigned int numElem */
        {
            __cmd->numElem = numElem;
        }
    }

    struct mvnc_nc_fifo_allocate_call_record *__call_record =
        (struct mvnc_nc_fifo_allocate_call_record *)vmalloc(sizeof(struct mvnc_nc_fifo_allocate_call_record));

    __call_record->fifoHandle = fifoHandle;

    __call_record->device = device;

    __call_record->tensorDesc = tensorDesc;

    __call_record->numElem = numElem;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncFifoAllocate);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncFifoAllocate);

EXPORTED dllexport ncStatus_t
ncFifoSetOption(struct ncFifoHandle_t * fifoHandle, int option, const void *data, unsigned int dataLength)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncFifoSetOption = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: const void * data */
        if ((data) != (NULL) && (dataLength) > (0)) {
            if (kava_shm_offset(data) >= 0) {
            } else {
                __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (dataLength)) * sizeof(const void));
        }}
    }
    struct mvnc_nc_fifo_set_option_call *__cmd =
        (struct mvnc_nc_fifo_set_option_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_fifo_set_option_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_FIFO_SET_OPTION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncFifoHandle_t * fifoHandle */
        {
            __cmd->fifoHandle = fifoHandle;
        }
        /* Input: int option */
        {
            __cmd->option = option;
        }
        /* Input: const void * data */
        {
            if ((data) != (NULL) && (dataLength) > (0)) {
                if (kava_shm_offset(data) >= 0) {
                    __cmd->data = (void *)kava_shm_offset(data);
                    __cmd->__shm_data = 1;
                } else {
                    __cmd->__shm_data = 0;

                    __cmd->data =
                        (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, data,
                        ((size_t) (dataLength)) * sizeof(const void));
            }} else {
                __cmd->data = NULL;
            }
        }
        /* Input: unsigned int dataLength */
        {
            __cmd->dataLength = dataLength;
        }
    }

    struct mvnc_nc_fifo_set_option_call_record *__call_record =
        (struct mvnc_nc_fifo_set_option_call_record *)vmalloc(sizeof(struct mvnc_nc_fifo_set_option_call_record));

    __call_record->fifoHandle = fifoHandle;

    __call_record->option = option;

    __call_record->dataLength = dataLength;

    __call_record->data = data;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 1;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncFifoSetOption);

    return NC_OK;
}

EXPORT_SYMBOL(ncFifoSetOption);

EXPORTED dllexport ncStatus_t
ncFifoGetOption(struct ncFifoHandle_t * fifoHandle, int option, void *data, unsigned int *dataLength)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncFifoGetOption = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: unsigned int * dataLength */
        if ((dataLength) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(unsigned int));
        }

        /* Size: void * data */
        if ((data) != (NULL) && (*dataLength) > (0)) {
            if (kava_shm_offset(data) >= 0) {
            } else {

            }
        }
    }
    struct mvnc_nc_fifo_get_option_call *__cmd =
        (struct mvnc_nc_fifo_get_option_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_fifo_get_option_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_FIFO_GET_OPTION;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncFifoHandle_t * fifoHandle */
        {
            __cmd->fifoHandle = fifoHandle;
        }
        /* Input: int option */
        {
            __cmd->option = option;
        }
        /* Input: void * data */
        {
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (kava_shm_offset(data) >= 0) {
                    __cmd->data = (void *)kava_shm_offset(data);
                    __cmd->__shm_data = 1;
                } else {
                    __cmd->__shm_data = 0;

                    __cmd->data = HAS_OUT_BUFFER_SENTINEL;
                }
            } else {
                __cmd->data = NULL;
            }
        }
        /* Input: unsigned int * dataLength */
        {
            if ((dataLength) != (NULL)) {
                __cmd->dataLength =
                    (unsigned int *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, dataLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __cmd->dataLength = NULL;
            }
        }
    }

    struct mvnc_nc_fifo_get_option_call_record *__call_record =
        (struct mvnc_nc_fifo_get_option_call_record *)vmalloc(sizeof(struct mvnc_nc_fifo_get_option_call_record));

    __call_record->fifoHandle = fifoHandle;

    __call_record->option = option;

    __call_record->dataLength = dataLength;

    __call_record->data = data;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncFifoGetOption);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncFifoGetOption);

EXPORTED dllexport ncStatus_t
ncFifoDestroy(struct ncFifoHandle_t ** fifoHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncFifoDestroy = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: struct ncFifoHandle_t ** fifoHandle */
        if ((fifoHandle) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
        }
    }
    struct mvnc_nc_fifo_destroy_call *__cmd =
        (struct mvnc_nc_fifo_destroy_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_fifo_destroy_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_FIFO_DESTROY;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncFifoHandle_t ** fifoHandle */
        {
            if ((fifoHandle) != (NULL)) {
                __cmd->fifoHandle =
                    (struct ncFifoHandle_t **)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, fifoHandle,
                    ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            } else {
                __cmd->fifoHandle = NULL;
            }
        }
    }

    struct mvnc_nc_fifo_destroy_call_record *__call_record =
        (struct mvnc_nc_fifo_destroy_call_record *)vmalloc(sizeof(struct mvnc_nc_fifo_destroy_call_record));

    __call_record->fifoHandle = fifoHandle;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncFifoDestroy);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncFifoDestroy);

EXPORTED dllexport ncStatus_t
ncFifoWriteElem(struct ncFifoHandle_t * fifoHandle, const void *inputTensor, unsigned int *inputTensorLength,
    void *userParam)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncFifoWriteElem = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: unsigned int * inputTensorLength */
        if ((inputTensorLength) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(unsigned int));
        }

        /* Size: const void * inputTensor */
        if ((inputTensor) != (NULL) && (*inputTensorLength) > (0)) {
            if (kava_shm_offset(inputTensor) >= 0) {
            } else {
                __total_buffer_size +=
                    chan->chan_buffer_size(chan, ((size_t) (*inputTensorLength)) * sizeof(const void));
        }}
    }
    struct mvnc_nc_fifo_write_elem_call *__cmd =
        (struct mvnc_nc_fifo_write_elem_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_fifo_write_elem_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_FIFO_WRITE_ELEM;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncFifoHandle_t * fifoHandle */
        {
            __cmd->fifoHandle = fifoHandle;
        }
        /* Input: const void * inputTensor */
        {
            if ((inputTensor) != (NULL) && (*inputTensorLength) > (0)) {
                if (kava_shm_offset(inputTensor) >= 0) {
                    __cmd->inputTensor = (void *)kava_shm_offset(inputTensor);
                    __cmd->__shm_inputTensor = 1;
                } else {
                    __cmd->__shm_inputTensor = 0;

                    __cmd->inputTensor =
                        (void *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, inputTensor,
                        ((size_t) (*inputTensorLength)) * sizeof(const void));
            }} else {
                __cmd->inputTensor = NULL;
            }
        }
        /* Input: unsigned int * inputTensorLength */
        {
            if ((inputTensorLength) != (NULL)) {
                __cmd->inputTensorLength =
                    (unsigned int *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, inputTensorLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __cmd->inputTensorLength = NULL;
            }
        }
        /* Input: void * userParam */
        {
            __cmd->userParam = userParam;
        }
    }

    struct mvnc_nc_fifo_write_elem_call_record *__call_record =
        (struct mvnc_nc_fifo_write_elem_call_record *)vmalloc(sizeof(struct mvnc_nc_fifo_write_elem_call_record));

    __call_record->fifoHandle = fifoHandle;

    __call_record->inputTensorLength = inputTensorLength;

    __call_record->userParam = userParam;

    __call_record->inputTensor = inputTensor;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncFifoWriteElem);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncFifoWriteElem);

EXPORTED dllexport ncStatus_t
ncFifoReadElem(struct ncFifoHandle_t * fifoHandle, void *outputData, unsigned int *outputDataLen, void **userParam)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncFifoReadElem = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
        /* Size: unsigned int * outputDataLen */
        if ((outputDataLen) != (NULL)) {
            __total_buffer_size += chan->chan_buffer_size(chan, ((size_t) (1)) * sizeof(unsigned int));
        }

        /* Size: void * outputData */
        if ((outputData) != (NULL) && (*outputDataLen) > (0)) {
            if (kava_shm_offset(outputData) >= 0) {
            } else {

            }
        }
    }
    struct mvnc_nc_fifo_read_elem_call *__cmd =
        (struct mvnc_nc_fifo_read_elem_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_fifo_read_elem_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_FIFO_READ_ELEM;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncFifoHandle_t * fifoHandle */
        {
            __cmd->fifoHandle = fifoHandle;
        }
        /* Input: void * outputData */
        {
            if ((outputData) != (NULL) && (*outputDataLen) > (0)) {
                if (kava_shm_offset(outputData) >= 0) {
                    __cmd->outputData = (void *)kava_shm_offset(outputData);
                    __cmd->__shm_outputData = 1;
                } else {
                    __cmd->__shm_outputData = 0;

                    __cmd->outputData = HAS_OUT_BUFFER_SENTINEL;
                }
            } else {
                __cmd->outputData = NULL;
            }
        }
        /* Input: unsigned int * outputDataLen */
        {
            if ((outputDataLen) != (NULL)) {
                __cmd->outputDataLen =
                    (unsigned int *)chan->chan_attach_buffer(chan, (struct kava_cmd_base *)__cmd, outputDataLen,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __cmd->outputDataLen = NULL;
            }
        }
        /* Input: void ** userParam */
        {
            if ((userParam) != (NULL)) {
                __cmd->userParam = HAS_OUT_BUFFER_SENTINEL;
            } else {
                __cmd->userParam = NULL;
            }
        }
    }

    struct mvnc_nc_fifo_read_elem_call_record *__call_record =
        (struct mvnc_nc_fifo_read_elem_call_record *)vmalloc(sizeof(struct mvnc_nc_fifo_read_elem_call_record));

    __call_record->fifoHandle = fifoHandle;

    __call_record->outputDataLen = outputDataLen;

    __call_record->userParam = userParam;

    __call_record->outputData = outputData;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncFifoReadElem);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncFifoReadElem);

EXPORTED dllexport ncStatus_t
ncFifoRemoveElem(struct ncFifoHandle_t * fifoHandle)
{

    intptr_t __call_id = kava_get_call_id(&__kava_endpoint);
    int64_t __thread_id;

    struct kava_buffer_list *__kava_alloc_list_ncFifoRemoveElem = kava_endpoint_buffer_list_new();

    size_t __total_buffer_size = 0;
    {
    }
    struct mvnc_nc_fifo_remove_elem_call *__cmd =
        (struct mvnc_nc_fifo_remove_elem_call *)chan->cmd_new(chan, sizeof(struct mvnc_nc_fifo_remove_elem_call),
        __total_buffer_size);
    __cmd->base.mode = KAVA_CMD_MODE_API;
    __cmd->base.command_id = CALL_MVNC_NC_FIFO_REMOVE_ELEM;
    __cmd->base.thread_id = __thread_id = kava_shadow_thread_id(kava_shadow_thread_pool);

    __cmd->__call_id = __call_id;

    {

        /* Input: struct ncFifoHandle_t * fifoHandle */
        {
            __cmd->fifoHandle = fifoHandle;
        }
    }

    struct mvnc_nc_fifo_remove_elem_call_record *__call_record =
        (struct mvnc_nc_fifo_remove_elem_call_record *)vmalloc(sizeof(struct mvnc_nc_fifo_remove_elem_call_record));

    __call_record->fifoHandle = fifoHandle;

    __call_record->__call_complete = 0;
    __call_record->__handler_deallocate = 0;
    kava_add_call(&__kava_endpoint, __call_id, __call_record);

    chan->cmd_send(chan, (struct kava_cmd_base *)__cmd);

    kava_endpoint_buffer_list_free(__kava_alloc_list_ncFifoRemoveElem);

    shadow_thread_handle_command_until(kava_shadow_thread_pool, __thread_id, __call_record->__call_complete);
    ncStatus_t ret;
    ret = __call_record->ret;
    vfree(__call_record);
    return ret;
}

EXPORT_SYMBOL(ncFifoRemoveElem);

/// Kernel initialization

static int __init
mvnc_init(void)
{
    kava_register_cmd_handler(KAVA_CMD_MODE_API, NULL, NULL);
    pr_info("Create control device\n");
    init_ctrl_if();
    pr_info("Load mvnc kernel library\n");
    init_global_kapi(KAVA_API_ID_MVNC, chan_mode);

    /* Initialize endpoint */
    init_endpoint_lib();
    __handle_command_mvnc_init();

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
mvnc_fini(void)
{
    pr_info("Stop running worker\n");
    stop_worker(chan);

    pr_info("Destroy endpoint\n");
    __handle_command_mvnc_destroy();

    pr_info("Unload mvnc kernel library\n");
    if (chan)
        chan->chan_free(chan);
    put_global_kapi();
    fini_ctrl_if();
}

module_init(mvnc_init);
module_exit(mvnc_fini);


MODULE_AUTHOR("Bodun Hu");
MODULE_DESCRIPTION("MVNC kernel library");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");

////// Replacement declarations

#define ava_begin_replacement
#define ava_end_replacement
