
#define __KAVA__ 1
#define kava_is_worker 1
#define kava_is_guest 0

#include "worker.h"

#undef AVA_BENCHMARKING_MIGRATE

#include "api.h"
#include "channel.h"
// #include "control.h"
// #include "command.h"
#include "command_handler.h"
#include "debug.h"
#include "endpoint.h"
// #include "shadow_thread_pool.h"
// #include "shared_memory.h"

// Must be included before mvnc_nw.h, so that API
// functions are declared properly.
#include <mvnc.h>
#include <worker.h>
#include "../klib/mvnc/mvnc_nw.h"
#include "mvnc_nw_utilities.h"

#pragma GCC diagnostic ignored "-Wunused-function"


static struct kava_endpoint __kava_endpoint;

struct mvnc_metadata {
    struct kava_metadata_base base;
    struct metadata application;
};

static void __handle_command_mvnc_init(void);
static void __handle_command_mvnc_destroy(void);
void __handle_command_mvnc(struct kava_chan *__chan, const struct kava_cmd_base *__cmd);
void __print_command_mvnc(FILE * file, const struct kava_chan *__chan, const struct kava_cmd_base *__cmd);

#define kava_metadata(p) (&((struct mvnc_metadata*)kava_internal_metadata(&__kava_endpoint, p))->application)

void enable_constructor(void) { /*  do nothing */ }

#include "mvnc_nw_utilities.h"

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

void __attribute__ ((constructor(1))) init_mvnc_worker(void)
{
    worker_common_init();
    __handle_command_mvnc_init();
}

////// API function stub implementations

static ncStatus_t
__wrapper_ncGlobalSetOption(int option, unsigned int dataLength, const void *data)
{
    {
        ncStatus_t ret;
        ret = ncGlobalSetOption(option, data, dataLength);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGlobalGetOption(int option, unsigned int *dataLength, void *data)
{
    {
        ncStatus_t ret;
        ret = ncGlobalGetOption(option, data, dataLength);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncDeviceSetOption(struct ncDeviceHandle_t *deviceHandle, int option, unsigned int dataLength,
    const void *data)
{
    {
        ncStatus_t ret;
        ret = ncDeviceSetOption(deviceHandle, option, data, dataLength);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncDeviceGetOption(struct ncDeviceHandle_t *deviceHandle, int option, unsigned int *dataLength, void *data)
{
    {
        ncStatus_t ret;
        ret = ncDeviceGetOption(deviceHandle, option, data, dataLength);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncDeviceCreate(int index, struct ncDeviceHandle_t **deviceHandle)
{
    {
        ncStatus_t ret;
        ret = ncDeviceCreate(index, deviceHandle);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncDeviceOpen(struct ncDeviceHandle_t *deviceHandle)
{
    {
        ncStatus_t ret;
        ret = ncDeviceOpen(deviceHandle);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncDeviceClose(struct ncDeviceHandle_t *deviceHandle)
{
    {
        ncStatus_t ret;
        ret = ncDeviceClose(deviceHandle);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncDeviceDestroy(struct ncDeviceHandle_t **deviceHandle)
{
    {
        ncStatus_t ret;
        ret = ncDeviceDestroy(deviceHandle);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGraphCreate(const char *ava_name, struct ncGraphHandle_t **graphHandle)
{
    {
        ncStatus_t ret;
        ret = ncGraphCreate(ava_name, graphHandle);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGraphAllocate(struct ncDeviceHandle_t *deviceHandle, struct ncGraphHandle_t *graphHandle,
    unsigned int graphBufferLength, const void *graphBuffer)
{
    {
        ncStatus_t ret;
        ret = ncGraphAllocate(deviceHandle, graphHandle, graphBuffer, graphBufferLength);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGraphDestroy(struct ncGraphHandle_t **graphHandle)
{
    {
        ncStatus_t ret;
        ret = ncGraphDestroy(graphHandle);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGraphSetOption(struct ncGraphHandle_t *graphHandle, int option, unsigned int dataLength, const void *data)
{
    {
        ncStatus_t ret;
        ret = ncGraphSetOption(graphHandle, option, data, dataLength);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGraphGetOption(struct ncGraphHandle_t *graphHandle, int option, unsigned int *dataLength, void *data)
{
    {
        ncStatus_t ret;
        ret = ncGraphGetOption(graphHandle, option, data, dataLength);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGraphQueueInference(struct ncGraphHandle_t *graphHandle, unsigned int inFifoCount,
    unsigned int outFifoCount, struct ncFifoHandle_t **fifoIn, struct ncFifoHandle_t **fifoOut)
{
    {
        ncStatus_t ret;
        ret = ncGraphQueueInference(graphHandle, fifoIn, inFifoCount, fifoOut, outFifoCount);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGraphQueueInferenceWithFifoElem(struct ncGraphHandle_t *graphHandle, struct ncFifoHandle_t *fifoIn,
    struct ncFifoHandle_t *fifoOut, unsigned int *inputTensorLength, void *userParam, const void *inputTensor)
{
    {
        ncStatus_t ret;
        ret =
            ncGraphQueueInferenceWithFifoElem(graphHandle, fifoIn, fifoOut, inputTensor, inputTensorLength, userParam);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGraphAllocateWithFifos(struct ncDeviceHandle_t *deviceHandle, struct ncGraphHandle_t *graphHandle,
    unsigned int graphBufferLength, struct ncFifoHandle_t **inFifoHandle, struct ncFifoHandle_t **outFifoHandle,
    const void *graphBuffer)
{
    {
        ncStatus_t ret;
        ret =
            ncGraphAllocateWithFifos(deviceHandle, graphHandle, graphBuffer, graphBufferLength, inFifoHandle,
            outFifoHandle);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncGraphAllocateWithFifosEx(struct ncDeviceHandle_t *deviceHandle, struct ncGraphHandle_t *graphHandle,
    unsigned int graphBufferLength, struct ncFifoHandle_t **inFifoHandle, ncFifoType_t inFifoType, int inNumElem,
    ncFifoDataType_t inDataType, struct ncFifoHandle_t **outFifoHandle, ncFifoType_t outFifoType, int outNumElem,
    ncFifoDataType_t outDataType, const void *graphBuffer)
{
    {
        ncStatus_t ret;
        ret =
            ncGraphAllocateWithFifosEx(deviceHandle, graphHandle, graphBuffer, graphBufferLength, inFifoHandle,
            inFifoType, inNumElem, inDataType, outFifoHandle, outFifoType, outNumElem, outDataType);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncFifoCreate(const char *ava_name, ncFifoType_t ava_type, struct ncFifoHandle_t **fifoHandle)
{
    {
        ncStatus_t ret;
        ret = ncFifoCreate(ava_name, ava_type, fifoHandle);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncFifoAllocate(struct ncFifoHandle_t *fifoHandle, struct ncDeviceHandle_t *device,
    struct ncTensorDescriptor_t *tensorDesc, unsigned int numElem)
{
    {
        ncStatus_t ret;
        ret = ncFifoAllocate(fifoHandle, device, tensorDesc, numElem);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncFifoSetOption(struct ncFifoHandle_t *fifoHandle, int option, unsigned int dataLength, const void *data)
{
    {
        ncStatus_t ret;
        ret = ncFifoSetOption(fifoHandle, option, data, dataLength);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncFifoGetOption(struct ncFifoHandle_t *fifoHandle, int option, unsigned int *dataLength, void *data)
{
    {
        ncStatus_t ret;
        ret = ncFifoGetOption(fifoHandle, option, data, dataLength);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncFifoDestroy(struct ncFifoHandle_t **fifoHandle)
{
    {
        ncStatus_t ret;
        ret = ncFifoDestroy(fifoHandle);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncFifoWriteElem(struct ncFifoHandle_t *fifoHandle, unsigned int *inputTensorLength, void *userParam,
    const void *inputTensor)
{
    {
        ncStatus_t ret;
        ret = ncFifoWriteElem(fifoHandle, inputTensor, inputTensorLength, userParam);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncFifoReadElem(struct ncFifoHandle_t *fifoHandle, unsigned int *outputDataLen, void **userParam,
    void *outputData)
{
    {
        ncStatus_t ret;
        ret = ncFifoReadElem(fifoHandle, outputData, outputDataLen, userParam);

        return ret;
    }
}

static ncStatus_t
__wrapper_ncFifoRemoveElem(struct ncFifoHandle_t *fifoHandle)
{
    {
        ncStatus_t ret;
        ret = ncFifoRemoveElem(fifoHandle);

        return ret;
    }
}

void
__handle_command_mvnc_init(void)
{
    kava_endpoint_init(&__kava_endpoint, sizeof(struct mvnc_metadata),
        kava_is_worker ? KAVA_COUNTER_TAG_WORKER : KAVA_COUNTER_TAG_KLIB);
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

    case CALL_MVNC_NC_GLOBAL_SET_OPTION:{
        GPtrArray *__kava_alloc_list_ncGlobalSetOption =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_global_set_option_call *__call = (struct mvnc_nc_global_set_option_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_global_set_option_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: int option */
        int option; {
            option = (int)__call->option;
            option = __call->option;
        }

        /* Input: unsigned int dataLength */
        unsigned int dataLength; {
            dataLength = (unsigned int)__call->dataLength;
            dataLength = __call->dataLength;
        }

        /* Input: const void * data */
        void *data; {
            data =
                ((__call->data) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->data)) : ((const void *)__call->data);
            if ((__call->data) != (NULL)) {
                if (__call->__shm_data) {
                    data = kava_shm_address((long)__call->data);
                } else {

                    void *__src_data_0;
                    __src_data_0 = data;
                    volatile size_t __buffer_size = 0;
                    __buffer_size = ((size_t) (dataLength));
                    data = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->data);

                    if ((data) != (__src_data_0)) {
                        memcpy(data, __src_data_0, __buffer_size * sizeof(const void));
                    }
            }} else {
                data =
                    ((__call->data) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->data)) : ((const void *)__call->data);
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncGlobalSetOption(option, dataLength, data);

        size_t __total_buffer_size = 0;
        {
            /* Size: const void * data */
            if ((data) != (NULL) && (dataLength) > (0)) {
                if (__call->__shm_data) {
                } else {

                }
            }
        }
        struct mvnc_nc_global_set_option_ret *__ret =
            (struct mvnc_nc_global_set_option_ret *)__chan->cmd_new(__chan,
            sizeof(struct mvnc_nc_global_set_option_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GLOBAL_SET_OPTION;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGlobalSetOption);  /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GLOBAL_GET_OPTION:{
        GPtrArray *__kava_alloc_list_ncGlobalGetOption =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_global_get_option_call *__call = (struct mvnc_nc_global_get_option_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_global_get_option_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: int option */
        int option; {
            option = (int)__call->option;
            option = __call->option;
        }

        /* Input: unsigned int * dataLength */
        unsigned int *dataLength; {
            dataLength =
                ((__call->dataLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->dataLength)) : ((unsigned int *)__call->dataLength);
            if ((__call->dataLength) != (NULL)) {
                unsigned int *__src_dataLength_0;
                __src_dataLength_0 = dataLength;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                dataLength = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __call->dataLength);

                if ((dataLength) != (__src_dataLength_0)) {
                    memcpy(dataLength, __src_dataLength_0, __buffer_size * sizeof(unsigned int));
                }
            } else {
                dataLength =
                    ((__call->dataLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->dataLength)) : ((unsigned int *)__call->dataLength);
        }}

        /* Input: void * data */
        void *data; {
            data =
                ((__call->data) != (NULL)) ? ((void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->data)) : ((void *)__call->data);
            if ((__call->data) != (NULL)) { {
                    const size_t __size = ((size_t) (*dataLength));
                    data = (void *)malloc(__size * sizeof(void));
                    g_ptr_array_add(__kava_alloc_list_ncGlobalGetOption, kava_buffer_with_deallocator_new(free, data));
            }}
        }

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncGlobalGetOption(option, dataLength, data);

        size_t __total_buffer_size = 0;
        {
            /* Size: unsigned int * dataLength */
            if ((dataLength) != (NULL)) {
                __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(unsigned int));
            }

            /* Size: void * data */
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (__call->__shm_data) {
                } else {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (*dataLength)) * sizeof(void));
            }}
        }
        struct mvnc_nc_global_get_option_ret *__ret =
            (struct mvnc_nc_global_get_option_ret *)__chan->cmd_new(__chan,
            sizeof(struct mvnc_nc_global_get_option_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GLOBAL_GET_OPTION;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: unsigned int * dataLength */
        {
            if ((dataLength) != (NULL)) {
                __ret->dataLength =
                    (unsigned int *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, dataLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __ret->dataLength = NULL;
            }
        }
/* Output: void * data */
        {
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (__call->__shm_data) {
                    __ret->data = __call->data;
                } else {
                    __ret->data =
                        (void *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, data,
                        ((size_t) (*dataLength)) * sizeof(void));
            }} else {
                __ret->data = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGlobalGetOption);  /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_DEVICE_SET_OPTION:{
        GPtrArray *__kava_alloc_list_ncDeviceSetOption =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_device_set_option_call *__call = (struct mvnc_nc_device_set_option_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_device_set_option_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        struct ncDeviceHandle_t *deviceHandle = (struct ncDeviceHandle_t *)__call->deviceHandle;

        /* Input: int option */
        int option; {
            option = (int)__call->option;
            option = __call->option;
        }

        /* Input: unsigned int dataLength */
        unsigned int dataLength; {
            dataLength = (unsigned int)__call->dataLength;
            dataLength = __call->dataLength;
        }

        /* Input: const void * data */
        void *data; {
            data =
                ((__call->data) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->data)) : ((const void *)__call->data);
            if ((__call->data) != (NULL)) {
                if (__call->__shm_data) {
                    data = kava_shm_address((long)__call->data);
                } else {

                    void *__src_data_0;
                    __src_data_0 = data;
                    volatile size_t __buffer_size = 0;
                    __buffer_size = ((size_t) (dataLength));
                    data = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->data);

                    if ((data) != (__src_data_0)) {
                        memcpy(data, __src_data_0, __buffer_size * sizeof(const void));
                    }
            }} else {
                data =
                    ((__call->data) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->data)) : ((const void *)__call->data);
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncDeviceSetOption(deviceHandle, option, dataLength, data);

        size_t __total_buffer_size = 0;
        {
            /* Size: const void * data */
            if ((data) != (NULL) && (dataLength) > (0)) {
                if (__call->__shm_data) {
                } else {

                }
            }
        }
        struct mvnc_nc_device_set_option_ret *__ret =
            (struct mvnc_nc_device_set_option_ret *)__chan->cmd_new(__chan,
            sizeof(struct mvnc_nc_device_set_option_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_DEVICE_SET_OPTION;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncDeviceSetOption);  /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_DEVICE_GET_OPTION:{
        GPtrArray *__kava_alloc_list_ncDeviceGetOption =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_device_get_option_call *__call = (struct mvnc_nc_device_get_option_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_device_get_option_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        struct ncDeviceHandle_t *deviceHandle = (struct ncDeviceHandle_t *)__call->deviceHandle;

        /* Input: int option */
        int option; {
            option = (int)__call->option;
            option = __call->option;
        }

        /* Input: unsigned int * dataLength */
        unsigned int *dataLength; {
            dataLength =
                ((__call->dataLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->dataLength)) : ((unsigned int *)__call->dataLength);
            if ((__call->dataLength) != (NULL)) {
                unsigned int *__src_dataLength_0;
                __src_dataLength_0 = dataLength;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                dataLength = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __call->dataLength);

                if ((dataLength) != (__src_dataLength_0)) {
                    memcpy(dataLength, __src_dataLength_0, __buffer_size * sizeof(unsigned int));
                }
            } else {
                dataLength =
                    ((__call->dataLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->dataLength)) : ((unsigned int *)__call->dataLength);
        }}

        /* Input: void * data */
        void *data; {
            data =
                ((__call->data) != (NULL)) ? ((void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->data)) : ((void *)__call->data);
            if ((__call->data) != (NULL)) { {
                    const size_t __size = ((size_t) (*dataLength));
                    data = (void *)malloc(__size * sizeof(void));
                    g_ptr_array_add(__kava_alloc_list_ncDeviceGetOption, kava_buffer_with_deallocator_new(free, data));
            }}
        }

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncDeviceGetOption(deviceHandle, option, dataLength, data);

        size_t __total_buffer_size = 0;
        {
            /* Size: unsigned int * dataLength */
            if ((dataLength) != (NULL)) {
                __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(unsigned int));
            }

            /* Size: void * data */
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (__call->__shm_data) {
                } else {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (*dataLength)) * sizeof(void));
            }}
        }
        struct mvnc_nc_device_get_option_ret *__ret =
            (struct mvnc_nc_device_get_option_ret *)__chan->cmd_new(__chan,
            sizeof(struct mvnc_nc_device_get_option_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_DEVICE_GET_OPTION;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: unsigned int * dataLength */
        {
            if ((dataLength) != (NULL)) {
                __ret->dataLength =
                    (unsigned int *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, dataLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __ret->dataLength = NULL;
            }
        }
/* Output: void * data */
        {
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (__call->__shm_data) {
                    __ret->data = __call->data;
                } else {
                    __ret->data =
                        (void *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, data,
                        ((size_t) (*dataLength)) * sizeof(void));
            }} else {
                __ret->data = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncDeviceGetOption);  /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_DEVICE_CREATE:{
        GPtrArray *__kava_alloc_list_ncDeviceCreate =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_device_create_call *__call = (struct mvnc_nc_device_create_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_device_create_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: int index */
        int index; {
            index = (int)__call->index;
            index = __call->index;
        }

        /* Input: struct ncDeviceHandle_t ** deviceHandle */
        struct ncDeviceHandle_t **deviceHandle; {
            deviceHandle =
                ((__call->deviceHandle) != (NULL)) ? ((struct ncDeviceHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->deviceHandle)) : ((struct ncDeviceHandle_t **)__call->deviceHandle);
            if ((__call->deviceHandle) != (NULL)) { {
                    const size_t __size = ((size_t) (1));
                    deviceHandle = (struct ncDeviceHandle_t **)malloc(__size * sizeof(struct ncDeviceHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncDeviceCreate, kava_buffer_with_deallocator_new(free,
                            deviceHandle));
            }}
        }

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncDeviceCreate(index, deviceHandle);

        size_t __total_buffer_size = 0;
        {
            /* Size: struct ncDeviceHandle_t ** deviceHandle */
            if ((deviceHandle) != (NULL)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(struct ncDeviceHandle_t *));
            }
        }
        struct mvnc_nc_device_create_ret *__ret =
            (struct mvnc_nc_device_create_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_device_create_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_DEVICE_CREATE;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: struct ncDeviceHandle_t ** deviceHandle */
        {
            if ((deviceHandle) != (NULL)) { {
                    const size_t __size_deviceHandle_0 = ((size_t) (1));
                    struct ncDeviceHandle_t **__tmp_deviceHandle_0;
                    __tmp_deviceHandle_0 =
                        (struct ncDeviceHandle_t **)calloc(1,
                        __size_deviceHandle_0 * sizeof(struct ncDeviceHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncDeviceCreate, kava_buffer_with_deallocator_new(free,
                            __tmp_deviceHandle_0));
                    const size_t __deviceHandle_size_0 = __size_deviceHandle_0;
                    size_t __deviceHandle_index_0;
                    for (__deviceHandle_index_0 = 0; __deviceHandle_index_0 < __deviceHandle_size_0;
                        __deviceHandle_index_0++) {
                        const size_t ava_index = __deviceHandle_index_0;

                        struct ncDeviceHandle_t **__deviceHandle_a_0;
                        __deviceHandle_a_0 =
                            (struct ncDeviceHandle_t **)(__tmp_deviceHandle_0) + __deviceHandle_index_0;

                        struct ncDeviceHandle_t **__deviceHandle_b_0;
                        __deviceHandle_b_0 = (struct ncDeviceHandle_t **)(deviceHandle) + __deviceHandle_index_0;

                        {
                            *__deviceHandle_a_0 =
                                (struct ncDeviceHandle_t *)*__deviceHandle_b_0;
                        }
                    }
                    __ret->deviceHandle =
                        (struct ncDeviceHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_deviceHandle_0, ((size_t) (1)) * sizeof(struct ncDeviceHandle_t *));
            }} else {
                __ret->deviceHandle = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncDeviceCreate);     /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_DEVICE_OPEN:{
        GPtrArray *__kava_alloc_list_ncDeviceOpen =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_device_open_call *__call = (struct mvnc_nc_device_open_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_device_open_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        struct ncDeviceHandle_t *deviceHandle = (struct ncDeviceHandle_t *)__call->deviceHandle;

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncDeviceOpen(deviceHandle);

        size_t __total_buffer_size = 0;
        {
        }
        struct mvnc_nc_device_open_ret *__ret =
            (struct mvnc_nc_device_open_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_device_open_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_DEVICE_OPEN;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncDeviceOpen);       /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_DEVICE_CLOSE:{
        GPtrArray *__kava_alloc_list_ncDeviceClose =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_device_close_call *__call = (struct mvnc_nc_device_close_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_device_close_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        struct ncDeviceHandle_t *deviceHandle = (struct ncDeviceHandle_t *)__call->deviceHandle;

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncDeviceClose(deviceHandle);

        size_t __total_buffer_size = 0;
        {
        }
        struct mvnc_nc_device_close_ret *__ret =
            (struct mvnc_nc_device_close_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_device_close_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_DEVICE_CLOSE;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncDeviceClose);      /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_DEVICE_DESTROY:{
        GPtrArray *__kava_alloc_list_ncDeviceDestroy =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_device_destroy_call *__call = (struct mvnc_nc_device_destroy_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_device_destroy_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncDeviceHandle_t ** deviceHandle */
        struct ncDeviceHandle_t **deviceHandle; {
            deviceHandle =
                ((__call->deviceHandle) != (NULL)) ? ((struct ncDeviceHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->deviceHandle)) : ((struct ncDeviceHandle_t **)__call->deviceHandle);
            if ((__call->deviceHandle) != (NULL)) {
                struct ncDeviceHandle_t **__src_deviceHandle_0;
                __src_deviceHandle_0 = deviceHandle;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                deviceHandle = (struct ncDeviceHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __call->deviceHandle);
                if ((__call->deviceHandle) != (NULL)) { {
                        const size_t __size = ((size_t) (1));
                        deviceHandle = (struct ncDeviceHandle_t **)malloc(__size * sizeof(struct ncDeviceHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncDeviceDestroy, kava_buffer_with_deallocator_new(free,
                                deviceHandle));
                }}

                const size_t __deviceHandle_size_0 = __buffer_size;
                size_t __deviceHandle_index_0;
                for (__deviceHandle_index_0 = 0; __deviceHandle_index_0 < __deviceHandle_size_0;
                    __deviceHandle_index_0++) {
                    const size_t ava_index = __deviceHandle_index_0;

                    struct ncDeviceHandle_t **__deviceHandle_a_0;
                    __deviceHandle_a_0 = (struct ncDeviceHandle_t **)(deviceHandle) + __deviceHandle_index_0;

                    struct ncDeviceHandle_t **__deviceHandle_b_0;
                    __deviceHandle_b_0 = (struct ncDeviceHandle_t **)(__src_deviceHandle_0) + __deviceHandle_index_0;

                    {
                        *__deviceHandle_a_0 = (struct ncDeviceHandle_t *)*__deviceHandle_b_0;
                        *__deviceHandle_a_0 =
                            (struct ncDeviceHandle_t *)*__deviceHandle_b_0;
                    }
            }} else {
                if ((__call->deviceHandle) != (NULL)) { {
                        const size_t __size = ((size_t) (1));
                        deviceHandle = (struct ncDeviceHandle_t **)malloc(__size * sizeof(struct ncDeviceHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncDeviceDestroy, kava_buffer_with_deallocator_new(free,
                                deviceHandle));
                }}
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncDeviceDestroy(deviceHandle);

        size_t __total_buffer_size = 0;
        {
            /* Size: struct ncDeviceHandle_t ** deviceHandle */
            if ((deviceHandle) != (NULL)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(struct ncDeviceHandle_t *));
            }
        }
        struct mvnc_nc_device_destroy_ret *__ret =
            (struct mvnc_nc_device_destroy_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_device_destroy_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_DEVICE_DESTROY;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: struct ncDeviceHandle_t ** deviceHandle */
        {
            if ((deviceHandle) != (NULL)) { {
                    const size_t __size_deviceHandle_0 = ((size_t) (1));
                    struct ncDeviceHandle_t **__tmp_deviceHandle_0;
                    __tmp_deviceHandle_0 =
                        (struct ncDeviceHandle_t **)calloc(1,
                        __size_deviceHandle_0 * sizeof(struct ncDeviceHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncDeviceDestroy, kava_buffer_with_deallocator_new(free,
                            __tmp_deviceHandle_0));
                    const size_t __deviceHandle_size_0 = __size_deviceHandle_0;
                    size_t __deviceHandle_index_0;
                    for (__deviceHandle_index_0 = 0; __deviceHandle_index_0 < __deviceHandle_size_0;
                        __deviceHandle_index_0++) {
                        const size_t ava_index = __deviceHandle_index_0;

                        struct ncDeviceHandle_t **__deviceHandle_a_0;
                        __deviceHandle_a_0 =
                            (struct ncDeviceHandle_t **)(__tmp_deviceHandle_0) + __deviceHandle_index_0;

                        struct ncDeviceHandle_t **__deviceHandle_b_0;
                        __deviceHandle_b_0 = (struct ncDeviceHandle_t **)(deviceHandle) + __deviceHandle_index_0;

                        {
                            *__deviceHandle_a_0 =
                                (struct ncDeviceHandle_t *)*__deviceHandle_b_0;
                        }
                    }
                    __ret->deviceHandle =
                        (struct ncDeviceHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_deviceHandle_0, ((size_t) (1)) * sizeof(struct ncDeviceHandle_t *));
            }} else {
                __ret->deviceHandle = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncDeviceDestroy);    /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GRAPH_CREATE:{
        GPtrArray *__kava_alloc_list_ncGraphCreate =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_graph_create_call *__call = (struct mvnc_nc_graph_create_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_graph_create_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: const char * ava_name */
        char *ava_name; {
            ava_name =
                ((__call->ava_name) != (NULL)) ? ((const char *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->ava_name)) : ((const char *)__call->ava_name);
            if ((__call->ava_name) != (NULL)) {
                char *__src_ava_name_0;
                __src_ava_name_0 = ava_name;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (strlen(ava_name) + 1));
                ava_name = (const char *)__chan->chan_get_buffer(__chan, __cmd, __call->ava_name);

                if ((ava_name) != (__src_ava_name_0)) {
                    memcpy(ava_name, __src_ava_name_0, __buffer_size * sizeof(const char));
                }
            } else {
                ava_name =
                    ((__call->ava_name) != (NULL)) ? ((const char *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->ava_name)) : ((const char *)__call->ava_name);
        }}

        /* Input: struct ncGraphHandle_t ** graphHandle */
        struct ncGraphHandle_t **graphHandle; {
            graphHandle =
                ((__call->graphHandle) != (NULL)) ? ((struct ncGraphHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->graphHandle)) : ((struct ncGraphHandle_t **)__call->graphHandle);
            if ((__call->graphHandle) != (NULL)) { {
                    const size_t __size = ((size_t) (1));
                    graphHandle = (struct ncGraphHandle_t **)malloc(__size * sizeof(struct ncGraphHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphCreate, kava_buffer_with_deallocator_new(free, graphHandle));
            }}
        }

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncGraphCreate(ava_name, graphHandle);

        size_t __total_buffer_size = 0;
        {
            /* Size: struct ncGraphHandle_t ** graphHandle */
            if ((graphHandle) != (NULL)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(struct ncGraphHandle_t *));
            }
        }
        struct mvnc_nc_graph_create_ret *__ret =
            (struct mvnc_nc_graph_create_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_graph_create_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GRAPH_CREATE;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: struct ncGraphHandle_t ** graphHandle */
        {
            if ((graphHandle) != (NULL)) { {
                    const size_t __size_graphHandle_0 = ((size_t) (1));
                    struct ncGraphHandle_t **__tmp_graphHandle_0;
                    __tmp_graphHandle_0 =
                        (struct ncGraphHandle_t **)calloc(1, __size_graphHandle_0 * sizeof(struct ncGraphHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphCreate, kava_buffer_with_deallocator_new(free,
                            __tmp_graphHandle_0));
                    const size_t __graphHandle_size_0 = __size_graphHandle_0;
                    size_t __graphHandle_index_0;
                    for (__graphHandle_index_0 = 0; __graphHandle_index_0 < __graphHandle_size_0;
                        __graphHandle_index_0++) {
                        const size_t ava_index = __graphHandle_index_0;

                        struct ncGraphHandle_t **__graphHandle_a_0;
                        __graphHandle_a_0 = (struct ncGraphHandle_t **)(__tmp_graphHandle_0) + __graphHandle_index_0;

                        struct ncGraphHandle_t **__graphHandle_b_0;
                        __graphHandle_b_0 = (struct ncGraphHandle_t **)(graphHandle) + __graphHandle_index_0;

                        {
                            *__graphHandle_a_0 =
                                (struct ncGraphHandle_t *)*__graphHandle_b_0;
                        }
                    }
                    __ret->graphHandle =
                        (struct ncGraphHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_graphHandle_0, ((size_t) (1)) * sizeof(struct ncGraphHandle_t *));
            }} else {
                __ret->graphHandle = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGraphCreate);      /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GRAPH_ALLOCATE:{
        GPtrArray *__kava_alloc_list_ncGraphAllocate =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_graph_allocate_call *__call = (struct mvnc_nc_graph_allocate_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_graph_allocate_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        struct ncDeviceHandle_t *deviceHandle = (struct ncDeviceHandle_t *)__call->deviceHandle;

        /* Input: struct ncGraphHandle_t * graphHandle */
        struct ncGraphHandle_t *graphHandle = (struct ncGraphHandle_t *)__call->graphHandle;

        /* Input: unsigned int graphBufferLength */
        unsigned int graphBufferLength; {
            graphBufferLength = (unsigned int)__call->graphBufferLength;
            graphBufferLength = __call->graphBufferLength;
        }

        /* Input: const void * graphBuffer */
        void *graphBuffer; {
            graphBuffer =
                ((__call->graphBuffer) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->graphBuffer)) : ((const void *)__call->graphBuffer);
            if ((__call->graphBuffer) != (NULL)) {
                if (__call->__shm_graphBuffer) {
                    graphBuffer = kava_shm_address((long)__call->graphBuffer);
                } else {

                    void *__src_graphBuffer_0;
                    __src_graphBuffer_0 = graphBuffer;
                    volatile size_t __buffer_size = 0;
                    __buffer_size = ((size_t) (graphBufferLength));
                    graphBuffer = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->graphBuffer);

                    if ((graphBuffer) != (__src_graphBuffer_0)) {
                        memcpy(graphBuffer, __src_graphBuffer_0, __buffer_size * sizeof(const void));
                    }
            }} else {
                graphBuffer =
                    ((__call->graphBuffer) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->graphBuffer)) : ((const void *)__call->graphBuffer);
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncGraphAllocate(deviceHandle, graphHandle, graphBufferLength, graphBuffer);

        size_t __total_buffer_size = 0;
        {
            /* Size: const void * graphBuffer */
            if ((graphBuffer) != (NULL) && (graphBufferLength) > (0)) {
                if (__call->__shm_graphBuffer) {
                } else {

                }
            }
        }
        struct mvnc_nc_graph_allocate_ret *__ret =
            (struct mvnc_nc_graph_allocate_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_graph_allocate_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GRAPH_ALLOCATE;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGraphAllocate);    /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GRAPH_DESTROY:{
        GPtrArray *__kava_alloc_list_ncGraphDestroy =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_graph_destroy_call *__call = (struct mvnc_nc_graph_destroy_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_graph_destroy_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncGraphHandle_t ** graphHandle */
        struct ncGraphHandle_t **graphHandle; {
            graphHandle =
                ((__call->graphHandle) != (NULL)) ? ((struct ncGraphHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->graphHandle)) : ((struct ncGraphHandle_t **)__call->graphHandle);
            if ((__call->graphHandle) != (NULL)) {
                struct ncGraphHandle_t **__src_graphHandle_0;
                __src_graphHandle_0 = graphHandle;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                graphHandle = (struct ncGraphHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __call->graphHandle);
                if ((__call->graphHandle) != (NULL)) { {
                        const size_t __size = ((size_t) (1));
                        graphHandle = (struct ncGraphHandle_t **)malloc(__size * sizeof(struct ncGraphHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncGraphDestroy, kava_buffer_with_deallocator_new(free,
                                graphHandle));
                }}

                const size_t __graphHandle_size_0 = __buffer_size;
                size_t __graphHandle_index_0;
                for (__graphHandle_index_0 = 0; __graphHandle_index_0 < __graphHandle_size_0; __graphHandle_index_0++) {
                    const size_t ava_index = __graphHandle_index_0;

                    struct ncGraphHandle_t **__graphHandle_a_0;
                    __graphHandle_a_0 = (struct ncGraphHandle_t **)(graphHandle) + __graphHandle_index_0;

                    struct ncGraphHandle_t **__graphHandle_b_0;
                    __graphHandle_b_0 = (struct ncGraphHandle_t **)(__src_graphHandle_0) + __graphHandle_index_0;

                    {
                        *__graphHandle_a_0 = (struct ncGraphHandle_t *)*__graphHandle_b_0;
                        *__graphHandle_a_0 =
                            (struct ncGraphHandle_t *)*__graphHandle_b_0;
                    }
            }} else {
                if ((__call->graphHandle) != (NULL)) { {
                        const size_t __size = ((size_t) (1));
                        graphHandle = (struct ncGraphHandle_t **)malloc(__size * sizeof(struct ncGraphHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncGraphDestroy, kava_buffer_with_deallocator_new(free,
                                graphHandle));
                }}
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncGraphDestroy(graphHandle);

        size_t __total_buffer_size = 0;
        {
            /* Size: struct ncGraphHandle_t ** graphHandle */
            if ((graphHandle) != (NULL)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(struct ncGraphHandle_t *));
            }
        }
        struct mvnc_nc_graph_destroy_ret *__ret =
            (struct mvnc_nc_graph_destroy_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_graph_destroy_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GRAPH_DESTROY;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: struct ncGraphHandle_t ** graphHandle */
        {
            if ((graphHandle) != (NULL)) { {
                    const size_t __size_graphHandle_0 = ((size_t) (1));
                    struct ncGraphHandle_t **__tmp_graphHandle_0;
                    __tmp_graphHandle_0 =
                        (struct ncGraphHandle_t **)calloc(1, __size_graphHandle_0 * sizeof(struct ncGraphHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphDestroy, kava_buffer_with_deallocator_new(free,
                            __tmp_graphHandle_0));
                    const size_t __graphHandle_size_0 = __size_graphHandle_0;
                    size_t __graphHandle_index_0;
                    for (__graphHandle_index_0 = 0; __graphHandle_index_0 < __graphHandle_size_0;
                        __graphHandle_index_0++) {
                        const size_t ava_index = __graphHandle_index_0;

                        struct ncGraphHandle_t **__graphHandle_a_0;
                        __graphHandle_a_0 = (struct ncGraphHandle_t **)(__tmp_graphHandle_0) + __graphHandle_index_0;

                        struct ncGraphHandle_t **__graphHandle_b_0;
                        __graphHandle_b_0 = (struct ncGraphHandle_t **)(graphHandle) + __graphHandle_index_0;

                        {
                            *__graphHandle_a_0 =
                                (struct ncGraphHandle_t *)*__graphHandle_b_0;
                        }
                    }
                    __ret->graphHandle =
                        (struct ncGraphHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_graphHandle_0, ((size_t) (1)) * sizeof(struct ncGraphHandle_t *));
            }} else {
                __ret->graphHandle = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGraphDestroy);     /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GRAPH_SET_OPTION:{
        GPtrArray *__kava_alloc_list_ncGraphSetOption =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_graph_set_option_call *__call = (struct mvnc_nc_graph_set_option_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_graph_set_option_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncGraphHandle_t * graphHandle */
        struct ncGraphHandle_t *graphHandle = (struct ncGraphHandle_t *)__call->graphHandle;

        /* Input: int option */
        int option; {
            option = (int)__call->option;
            option = __call->option;
        }

        /* Input: unsigned int dataLength */
        unsigned int dataLength; {
            dataLength = (unsigned int)__call->dataLength;
            dataLength = __call->dataLength;
        }

        /* Input: const void * data */
        void *data; {
            data =
                ((__call->data) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->data)) : ((const void *)__call->data);
            if ((__call->data) != (NULL)) {
                if (__call->__shm_data) {
                    data = kava_shm_address((long)__call->data);
                } else {

                    void *__src_data_0;
                    __src_data_0 = data;
                    volatile size_t __buffer_size = 0;
                    __buffer_size = ((size_t) (dataLength));
                    data = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->data);

                    if ((data) != (__src_data_0)) {
                        memcpy(data, __src_data_0, __buffer_size * sizeof(const void));
                    }
            }} else {
                data =
                    ((__call->data) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->data)) : ((const void *)__call->data);
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncGraphSetOption(graphHandle, option, dataLength, data);

        size_t __total_buffer_size = 0;
        {
            /* Size: const void * data */
            if ((data) != (NULL) && (dataLength) > (0)) {
                if (__call->__shm_data) {
                } else {

                }
            }
        }
        struct mvnc_nc_graph_set_option_ret *__ret =
            (struct mvnc_nc_graph_set_option_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_graph_set_option_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GRAPH_SET_OPTION;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGraphSetOption);   /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GRAPH_GET_OPTION:{
        GPtrArray *__kava_alloc_list_ncGraphGetOption =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_graph_get_option_call *__call = (struct mvnc_nc_graph_get_option_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_graph_get_option_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncGraphHandle_t * graphHandle */
        struct ncGraphHandle_t *graphHandle = (struct ncGraphHandle_t *)__call->graphHandle;

        /* Input: int option */
        int option; {
            option = (int)__call->option;
            option = __call->option;
        }

        /* Input: unsigned int * dataLength */
        unsigned int *dataLength; {
            dataLength =
                ((__call->dataLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->dataLength)) : ((unsigned int *)__call->dataLength);
            if ((__call->dataLength) != (NULL)) {
                unsigned int *__src_dataLength_0;
                __src_dataLength_0 = dataLength;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                dataLength = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __call->dataLength);

                if ((dataLength) != (__src_dataLength_0)) {
                    memcpy(dataLength, __src_dataLength_0, __buffer_size * sizeof(unsigned int));
                }
            } else {
                dataLength =
                    ((__call->dataLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->dataLength)) : ((unsigned int *)__call->dataLength);
        }}

        /* Input: void * data */
        void *data; {
            data =
                ((__call->data) != (NULL)) ? ((void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->data)) : ((void *)__call->data);
            if ((__call->data) != (NULL)) { {
                    const size_t __size = ((size_t) (*dataLength));
                    data = (void *)malloc(__size * sizeof(void));
                    g_ptr_array_add(__kava_alloc_list_ncGraphGetOption, kava_buffer_with_deallocator_new(free, data));
            }}
        }

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncGraphGetOption(graphHandle, option, dataLength, data);

        size_t __total_buffer_size = 0;
        {
            /* Size: unsigned int * dataLength */
            if ((dataLength) != (NULL)) {
                __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(unsigned int));
            }

            /* Size: void * data */
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (__call->__shm_data) {
                } else {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (*dataLength)) * sizeof(void));
            }}
        }
        struct mvnc_nc_graph_get_option_ret *__ret =
            (struct mvnc_nc_graph_get_option_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_graph_get_option_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GRAPH_GET_OPTION;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: unsigned int * dataLength */
        {
            if ((dataLength) != (NULL)) {
                __ret->dataLength =
                    (unsigned int *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, dataLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __ret->dataLength = NULL;
            }
        }
/* Output: void * data */
        {
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (__call->__shm_data) {
                    __ret->data = __call->data;
                } else {
                    __ret->data =
                        (void *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, data,
                        ((size_t) (*dataLength)) * sizeof(void));
            }} else {
                __ret->data = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGraphGetOption);   /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GRAPH_QUEUE_INFERENCE:{
        GPtrArray *__kava_alloc_list_ncGraphQueueInference =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_graph_queue_inference_call *__call = (struct mvnc_nc_graph_queue_inference_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_graph_queue_inference_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncGraphHandle_t * graphHandle */
        struct ncGraphHandle_t *graphHandle = (struct ncGraphHandle_t *)__call->graphHandle;

        /* Input: unsigned int inFifoCount */
        unsigned int inFifoCount; {
            inFifoCount = (unsigned int)__call->inFifoCount;
            inFifoCount = __call->inFifoCount;
        }

        /* Input: unsigned int outFifoCount */
        unsigned int outFifoCount; {
            outFifoCount = (unsigned int)__call->outFifoCount;
            outFifoCount = __call->outFifoCount;
        }

        /* Input: struct ncFifoHandle_t ** fifoIn */
        struct ncFifoHandle_t **fifoIn; {
            fifoIn =
                ((__call->fifoIn) != (NULL)) ? ((struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->fifoIn)) : ((struct ncFifoHandle_t **)__call->fifoIn);
            if ((__call->fifoIn) != (NULL)) {
                struct ncFifoHandle_t **__src_fifoIn_0;
                __src_fifoIn_0 = fifoIn;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (inFifoCount));
                fifoIn = (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __call->fifoIn);
                if ((__call->fifoIn) != (NULL)) { {
                        const size_t __size = ((size_t) (inFifoCount));
                        fifoIn = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncGraphQueueInference, kava_buffer_with_deallocator_new(free,
                                fifoIn));
                }}

                const size_t __fifoIn_size_0 = __buffer_size;
                size_t __fifoIn_index_0;
                for (__fifoIn_index_0 = 0; __fifoIn_index_0 < __fifoIn_size_0; __fifoIn_index_0++) {
                    const size_t ava_index = __fifoIn_index_0;

                    struct ncFifoHandle_t **__fifoIn_a_0;
                    __fifoIn_a_0 = (struct ncFifoHandle_t **)(fifoIn) + __fifoIn_index_0;

                    struct ncFifoHandle_t **__fifoIn_b_0;
                    __fifoIn_b_0 = (struct ncFifoHandle_t **)(__src_fifoIn_0) + __fifoIn_index_0;

                    {
                        *__fifoIn_a_0 = (struct ncFifoHandle_t *)*__fifoIn_b_0;
                        *__fifoIn_a_0 =
                            (struct ncFifoHandle_t *)*__fifoIn_b_0;
                    }
            }} else {
                if ((__call->fifoIn) != (NULL)) { {
                        const size_t __size = ((size_t) (inFifoCount));
                        fifoIn = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncGraphQueueInference, kava_buffer_with_deallocator_new(free,
                                fifoIn));
                }}
        }}

        /* Input: struct ncFifoHandle_t ** fifoOut */
        struct ncFifoHandle_t **fifoOut; {
            fifoOut =
                ((__call->fifoOut) != (NULL)) ? ((struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->fifoOut)) : ((struct ncFifoHandle_t **)__call->fifoOut);
            if ((__call->fifoOut) != (NULL)) {
                struct ncFifoHandle_t **__src_fifoOut_0;
                __src_fifoOut_0 = fifoOut;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (outFifoCount));
                fifoOut = (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __call->fifoOut);
                if ((__call->fifoOut) != (NULL)) { {
                        const size_t __size = ((size_t) (outFifoCount));
                        fifoOut = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncGraphQueueInference, kava_buffer_with_deallocator_new(free,
                                fifoOut));
                }}

                const size_t __fifoOut_size_0 = __buffer_size;
                size_t __fifoOut_index_0;
                for (__fifoOut_index_0 = 0; __fifoOut_index_0 < __fifoOut_size_0; __fifoOut_index_0++) {
                    const size_t ava_index = __fifoOut_index_0;

                    struct ncFifoHandle_t **__fifoOut_a_0;
                    __fifoOut_a_0 = (struct ncFifoHandle_t **)(fifoOut) + __fifoOut_index_0;

                    struct ncFifoHandle_t **__fifoOut_b_0;
                    __fifoOut_b_0 = (struct ncFifoHandle_t **)(__src_fifoOut_0) + __fifoOut_index_0;

                    {
                        *__fifoOut_a_0 = (struct ncFifoHandle_t *)*__fifoOut_b_0;
                        *__fifoOut_a_0 =
                            (struct ncFifoHandle_t *)*__fifoOut_b_0;
                    }
            }} else {
                if ((__call->fifoOut) != (NULL)) { {
                        const size_t __size = ((size_t) (outFifoCount));
                        fifoOut = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncGraphQueueInference, kava_buffer_with_deallocator_new(free,
                                fifoOut));
                }}
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncGraphQueueInference(graphHandle, inFifoCount, outFifoCount, fifoIn, fifoOut);

        size_t __total_buffer_size = 0;
        {
            /* Size: struct ncFifoHandle_t ** fifoIn */
            if ((fifoIn) != (NULL) && (inFifoCount) > (0)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (inFifoCount)) * sizeof(struct ncFifoHandle_t *));
            }

            /* Size: struct ncFifoHandle_t ** fifoOut */
            if ((fifoOut) != (NULL) && (outFifoCount) > (0)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (outFifoCount)) * sizeof(struct ncFifoHandle_t *));
            }
        }
        struct mvnc_nc_graph_queue_inference_ret *__ret =
            (struct mvnc_nc_graph_queue_inference_ret *)__chan->cmd_new(__chan,
            sizeof(struct mvnc_nc_graph_queue_inference_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GRAPH_QUEUE_INFERENCE;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: struct ncFifoHandle_t ** fifoIn */
        {
            if ((fifoIn) != (NULL)) { {
                    const size_t __size_fifoIn_0 = ((size_t) (inFifoCount));
                    struct ncFifoHandle_t **__tmp_fifoIn_0;
                    __tmp_fifoIn_0 =
                        (struct ncFifoHandle_t **)calloc(1, __size_fifoIn_0 * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphQueueInference, kava_buffer_with_deallocator_new(free,
                            __tmp_fifoIn_0));
                    const size_t __fifoIn_size_0 = __size_fifoIn_0;
                    size_t __fifoIn_index_0;
                    for (__fifoIn_index_0 = 0; __fifoIn_index_0 < __fifoIn_size_0; __fifoIn_index_0++) {
                        const size_t ava_index = __fifoIn_index_0;

                        struct ncFifoHandle_t **__fifoIn_a_0;
                        __fifoIn_a_0 = (struct ncFifoHandle_t **)(__tmp_fifoIn_0) + __fifoIn_index_0;

                        struct ncFifoHandle_t **__fifoIn_b_0;
                        __fifoIn_b_0 = (struct ncFifoHandle_t **)(fifoIn) + __fifoIn_index_0;

                        {
                            *__fifoIn_a_0 =
                                (struct ncFifoHandle_t *)*__fifoIn_b_0);
                        }
                    }
                    __ret->fifoIn =
                        (struct ncFifoHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_fifoIn_0, ((size_t) (inFifoCount)) * sizeof(struct ncFifoHandle_t *));
            }} else {
                __ret->fifoIn = NULL;
            }
        }
/* Output: struct ncFifoHandle_t ** fifoOut */
        {
            if ((fifoOut) != (NULL)) { {
                    const size_t __size_fifoOut_0 = ((size_t) (outFifoCount));
                    struct ncFifoHandle_t **__tmp_fifoOut_0;
                    __tmp_fifoOut_0 =
                        (struct ncFifoHandle_t **)calloc(1, __size_fifoOut_0 * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphQueueInference, kava_buffer_with_deallocator_new(free,
                            __tmp_fifoOut_0));
                    const size_t __fifoOut_size_0 = __size_fifoOut_0;
                    size_t __fifoOut_index_0;
                    for (__fifoOut_index_0 = 0; __fifoOut_index_0 < __fifoOut_size_0; __fifoOut_index_0++) {
                        const size_t ava_index = __fifoOut_index_0;

                        struct ncFifoHandle_t **__fifoOut_a_0;
                        __fifoOut_a_0 = (struct ncFifoHandle_t **)(__tmp_fifoOut_0) + __fifoOut_index_0;

                        struct ncFifoHandle_t **__fifoOut_b_0;
                        __fifoOut_b_0 = (struct ncFifoHandle_t **)(fifoOut) + __fifoOut_index_0;

                        {
                            *__fifoOut_a_0 =
                                (struct ncFifoHandle_t *)*__fifoOut_b_0;
                        }
                    }
                    __ret->fifoOut =
                        (struct ncFifoHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_fifoOut_0, ((size_t) (outFifoCount)) * sizeof(struct ncFifoHandle_t *));
            }} else {
                __ret->fifoOut = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGraphQueueInference);      /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GRAPH_QUEUE_INFERENCE_WITH_FIFO_ELEM:{
        GPtrArray *__kava_alloc_list_ncGraphQueueInferenceWithFifoElem =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_graph_queue_inference_with_fifo_elem_call *__call =
            (struct mvnc_nc_graph_queue_inference_with_fifo_elem_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_graph_queue_inference_with_fifo_elem_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncGraphHandle_t * graphHandle */
        struct ncGraphHandle_t *graphHandle = (struct ncGraphHandle_t *)__call->graphHandle;

        /* Input: struct ncFifoHandle_t * fifoIn */
        struct ncFifoHandle_t *fifoIn = (struct ncFifoHandle_t *)__call->fifoIn;

        /* Input: struct ncFifoHandle_t * fifoOut */
        struct ncFifoHandle_t *fifoOut = (struct ncFifoHandle_t *)__call->fifoOut;

        /* Input: unsigned int * inputTensorLength */
        unsigned int *inputTensorLength; {
            inputTensorLength =
                ((__call->inputTensorLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->inputTensorLength)) : ((unsigned int *)__call->inputTensorLength);
            if ((__call->inputTensorLength) != (NULL)) {
                unsigned int *__src_inputTensorLength_0;
                __src_inputTensorLength_0 = inputTensorLength;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                inputTensorLength = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __call->inputTensorLength);

                if ((inputTensorLength) != (__src_inputTensorLength_0)) {
                    memcpy(inputTensorLength, __src_inputTensorLength_0, __buffer_size * sizeof(unsigned int));
                }
            } else {
                inputTensorLength =
                    ((__call->inputTensorLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->inputTensorLength)) : ((unsigned int *)__call->inputTensorLength);
        }}

        /* Input: void * userParam */
        void *userParam; {
            userParam = (void *)__call->userParam;
            userParam = __call->userParam;
        }

        /* Input: const void * inputTensor */
        void *inputTensor; {
            inputTensor =
                ((__call->inputTensor) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->inputTensor)) : ((const void *)__call->inputTensor);
            if ((__call->inputTensor) != (NULL)) {
                if (__call->__shm_inputTensor) {
                    inputTensor = kava_shm_address((long)__call->inputTensor);
                } else {

                    void *__src_inputTensor_0;
                    __src_inputTensor_0 = inputTensor;
                    volatile size_t __buffer_size = 0;
                    __buffer_size = ((size_t) (*inputTensorLength));
                    inputTensor = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->inputTensor);

                    if ((inputTensor) != (__src_inputTensor_0)) {
                        memcpy(inputTensor, __src_inputTensor_0, __buffer_size * sizeof(const void));
                    }
            }} else {
                inputTensor =
                    ((__call->inputTensor) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->inputTensor)) : ((const void *)__call->inputTensor);
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret =
            __wrapper_ncGraphQueueInferenceWithFifoElem(graphHandle, fifoIn, fifoOut, inputTensorLength, userParam,
            inputTensor);

        size_t __total_buffer_size = 0;
        {
            /* Size: unsigned int * inputTensorLength */
            if ((inputTensorLength) != (NULL)) {
                __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(unsigned int));
            }

            /* Size: const void * inputTensor */
            if ((inputTensor) != (NULL) && (*inputTensorLength) > (0)) {
                if (__call->__shm_inputTensor) {
                } else {

                }
            }
        }
        struct mvnc_nc_graph_queue_inference_with_fifo_elem_ret *__ret =
            (struct mvnc_nc_graph_queue_inference_with_fifo_elem_ret *)__chan->cmd_new(__chan,
            sizeof(struct mvnc_nc_graph_queue_inference_with_fifo_elem_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GRAPH_QUEUE_INFERENCE_WITH_FIFO_ELEM;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: unsigned int * inputTensorLength */
        {
            if ((inputTensorLength) != (NULL)) {
                __ret->inputTensorLength =
                    (unsigned int *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, inputTensorLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __ret->inputTensorLength = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGraphQueueInferenceWithFifoElem);  /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS:{
        GPtrArray *__kava_alloc_list_ncGraphAllocateWithFifos =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_graph_allocate_with_fifos_call *__call = (struct mvnc_nc_graph_allocate_with_fifos_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_graph_allocate_with_fifos_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        struct ncDeviceHandle_t *deviceHandle = (struct ncDeviceHandle_t *)__call->deviceHandle;

        /* Input: struct ncGraphHandle_t * graphHandle */
        struct ncGraphHandle_t *graphHandle = (struct ncGraphHandle_t *)__call->graphHandle;

        /* Input: unsigned int graphBufferLength */
        unsigned int graphBufferLength; {
            graphBufferLength = (unsigned int)__call->graphBufferLength;
            graphBufferLength = __call->graphBufferLength;
        }

        /* Input: struct ncFifoHandle_t ** inFifoHandle */
        struct ncFifoHandle_t **inFifoHandle; {
            inFifoHandle =
                ((__call->inFifoHandle) != (NULL)) ? ((struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->inFifoHandle)) : ((struct ncFifoHandle_t **)__call->inFifoHandle);
            if ((__call->inFifoHandle) != (NULL)) { {
                    const size_t __size = ((size_t) (1));
                    inFifoHandle = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphAllocateWithFifos, kava_buffer_with_deallocator_new(free,
                            inFifoHandle));
            }}
        }

        /* Input: struct ncFifoHandle_t ** outFifoHandle */
        struct ncFifoHandle_t **outFifoHandle; {
            outFifoHandle =
                ((__call->outFifoHandle) != (NULL)) ? ((struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->outFifoHandle)) : ((struct ncFifoHandle_t **)__call->outFifoHandle);
            if ((__call->outFifoHandle) != (NULL)) { {
                    const size_t __size = ((size_t) (1));
                    outFifoHandle = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphAllocateWithFifos, kava_buffer_with_deallocator_new(free,
                            outFifoHandle));
            }}
        }

        /* Input: const void * graphBuffer */
        void *graphBuffer; {
            graphBuffer =
                ((__call->graphBuffer) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->graphBuffer)) : ((const void *)__call->graphBuffer);
            if ((__call->graphBuffer) != (NULL)) {
                if (__call->__shm_graphBuffer) {
                    graphBuffer = kava_shm_address((long)__call->graphBuffer);
                } else {

                    void *__src_graphBuffer_0;
                    __src_graphBuffer_0 = graphBuffer;
                    volatile size_t __buffer_size = 0;
                    __buffer_size = ((size_t) (graphBufferLength));
                    graphBuffer = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->graphBuffer);

                    if ((graphBuffer) != (__src_graphBuffer_0)) {
                        memcpy(graphBuffer, __src_graphBuffer_0, __buffer_size * sizeof(const void));
                    }
            }} else {
                graphBuffer =
                    ((__call->graphBuffer) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->graphBuffer)) : ((const void *)__call->graphBuffer);
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret =
            __wrapper_ncGraphAllocateWithFifos(deviceHandle, graphHandle, graphBufferLength, inFifoHandle,
            outFifoHandle, graphBuffer);

        size_t __total_buffer_size = 0;
        {
            /* Size: struct ncFifoHandle_t ** inFifoHandle */
            if ((inFifoHandle) != (NULL)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }

            /* Size: struct ncFifoHandle_t ** outFifoHandle */
            if ((outFifoHandle) != (NULL)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }

            /* Size: const void * graphBuffer */
            if ((graphBuffer) != (NULL) && (graphBufferLength) > (0)) {
                if (__call->__shm_graphBuffer) {
                } else {

                }
            }
        }
        struct mvnc_nc_graph_allocate_with_fifos_ret *__ret =
            (struct mvnc_nc_graph_allocate_with_fifos_ret *)__chan->cmd_new(__chan,
            sizeof(struct mvnc_nc_graph_allocate_with_fifos_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: struct ncFifoHandle_t ** inFifoHandle */
        {
            if ((inFifoHandle) != (NULL)) { {
                    const size_t __size_inFifoHandle_0 = ((size_t) (1));
                    struct ncFifoHandle_t **__tmp_inFifoHandle_0;
                    __tmp_inFifoHandle_0 =
                        (struct ncFifoHandle_t **)calloc(1, __size_inFifoHandle_0 * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphAllocateWithFifos, kava_buffer_with_deallocator_new(free,
                            __tmp_inFifoHandle_0));
                    const size_t __inFifoHandle_size_0 = __size_inFifoHandle_0;
                    size_t __inFifoHandle_index_0;
                    for (__inFifoHandle_index_0 = 0; __inFifoHandle_index_0 < __inFifoHandle_size_0;
                        __inFifoHandle_index_0++) {
                        const size_t ava_index = __inFifoHandle_index_0;

                        struct ncFifoHandle_t **__inFifoHandle_a_0;
                        __inFifoHandle_a_0 = (struct ncFifoHandle_t **)(__tmp_inFifoHandle_0) + __inFifoHandle_index_0;

                        struct ncFifoHandle_t **__inFifoHandle_b_0;
                        __inFifoHandle_b_0 = (struct ncFifoHandle_t **)(inFifoHandle) + __inFifoHandle_index_0;

                        {
                            *__inFifoHandle_a_0 =
                                (struct ncFifoHandle_t *)*__inFifoHandle_b_0;
                        }
                    }
                    __ret->inFifoHandle =
                        (struct ncFifoHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_inFifoHandle_0, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }} else {
                __ret->inFifoHandle = NULL;
            }
        }
/* Output: struct ncFifoHandle_t ** outFifoHandle */
        {
            if ((outFifoHandle) != (NULL)) { {
                    const size_t __size_outFifoHandle_0 = ((size_t) (1));
                    struct ncFifoHandle_t **__tmp_outFifoHandle_0;
                    __tmp_outFifoHandle_0 =
                        (struct ncFifoHandle_t **)calloc(1, __size_outFifoHandle_0 * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphAllocateWithFifos, kava_buffer_with_deallocator_new(free,
                            __tmp_outFifoHandle_0));
                    const size_t __outFifoHandle_size_0 = __size_outFifoHandle_0;
                    size_t __outFifoHandle_index_0;
                    for (__outFifoHandle_index_0 = 0; __outFifoHandle_index_0 < __outFifoHandle_size_0;
                        __outFifoHandle_index_0++) {
                        const size_t ava_index = __outFifoHandle_index_0;

                        struct ncFifoHandle_t **__outFifoHandle_a_0;
                        __outFifoHandle_a_0 =
                            (struct ncFifoHandle_t **)(__tmp_outFifoHandle_0) + __outFifoHandle_index_0;

                        struct ncFifoHandle_t **__outFifoHandle_b_0;
                        __outFifoHandle_b_0 = (struct ncFifoHandle_t **)(outFifoHandle) + __outFifoHandle_index_0;

                        {
                            *__outFifoHandle_a_0 =
                                (struct ncFifoHandle_t *)*__outFifoHandle_b_0;
                        }
                    }
                    __ret->outFifoHandle =
                        (struct ncFifoHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_outFifoHandle_0, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }} else {
                __ret->outFifoHandle = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGraphAllocateWithFifos);   /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS_EX:{
        GPtrArray *__kava_alloc_list_ncGraphAllocateWithFifosEx =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_graph_allocate_with_fifos_ex_call *__call =
            (struct mvnc_nc_graph_allocate_with_fifos_ex_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_graph_allocate_with_fifos_ex_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncDeviceHandle_t * deviceHandle */
        struct ncDeviceHandle_t *deviceHandle = (struct ncDeviceHandle_t *)__call->deviceHandle;

        /* Input: struct ncGraphHandle_t * graphHandle */
        struct ncGraphHandle_t *graphHandle = (struct ncGraphHandle_t *)__call->graphHandle;

        /* Input: unsigned int graphBufferLength */
        unsigned int graphBufferLength = (unsigned int)__call->graphBufferLength;

        /* Input: struct ncFifoHandle_t ** inFifoHandle */
        struct ncFifoHandle_t **inFifoHandle; {
            inFifoHandle =
                ((__call->inFifoHandle) != (NULL)) ? ((struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->inFifoHandle)) : ((struct ncFifoHandle_t **)__call->inFifoHandle);
            if ((__call->inFifoHandle) != (NULL)) { {
                    const size_t __size = ((size_t) (1));
                    inFifoHandle = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphAllocateWithFifosEx, kava_buffer_with_deallocator_new(free,
                            inFifoHandle));
            }}
        }

        /* Input: ncFifoType_t inFifoType */
        ncFifoType_t inFifoType; {
            inFifoType = (ncFifoType_t) __call->inFifoType;
            inFifoType = __call->inFifoType;
        }

        /* Input: int inNumElem */
        int inNumElem; {
            inNumElem = (int)__call->inNumElem;
            inNumElem = __call->inNumElem;
        }

        /* Input: ncFifoDataType_t inDataType */
        ncFifoDataType_t inDataType; {
            inDataType = (ncFifoDataType_t) __call->inDataType;
            inDataType = __call->inDataType;
        }

        /* Input: struct ncFifoHandle_t ** outFifoHandle */
        struct ncFifoHandle_t **outFifoHandle; {
            outFifoHandle =
                ((__call->outFifoHandle) != (NULL)) ? ((struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->outFifoHandle)) : ((struct ncFifoHandle_t **)__call->outFifoHandle);
            if ((__call->outFifoHandle) != (NULL)) { {
                    const size_t __size = ((size_t) (1));
                    outFifoHandle = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphAllocateWithFifosEx, kava_buffer_with_deallocator_new(free,
                            outFifoHandle));
            }}
        }

        /* Input: ncFifoType_t outFifoType */
        ncFifoType_t outFifoType; {
            outFifoType = (ncFifoType_t) __call->outFifoType;
            outFifoType = __call->outFifoType;
        }

        /* Input: int outNumElem */
        int outNumElem; {
            outNumElem = (int)__call->outNumElem;
            outNumElem = __call->outNumElem;
        }

        /* Input: ncFifoDataType_t outDataType */
        ncFifoDataType_t outDataType; {
            outDataType = (ncFifoDataType_t) __call->outDataType;
            outDataType = __call->outDataType;
        }

        /* Input: const void * graphBuffer */
        void *graphBuffer; {
            graphBuffer =
                ((__call->graphBuffer) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->graphBuffer)) : ((const void *)__call->graphBuffer);
            if ((__call->graphBuffer) != (NULL)) {
                if (__call->__shm_graphBuffer) {
                    graphBuffer = kava_shm_address((long)__call->graphBuffer);
                } else {

                    void *__src_graphBuffer_0;
                    __src_graphBuffer_0 = graphBuffer;
                    volatile size_t __buffer_size = 0;
                    __buffer_size = ((size_t) (graphBufferLength));
                    graphBuffer = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->graphBuffer);

                    if ((graphBuffer) != (__src_graphBuffer_0)) {
                        memcpy(graphBuffer, __src_graphBuffer_0, __buffer_size * sizeof(const void));
                    }
            }} else {
                graphBuffer =
                    ((__call->graphBuffer) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->graphBuffer)) : ((const void *)__call->graphBuffer);
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret =
            __wrapper_ncGraphAllocateWithFifosEx(deviceHandle, graphHandle, graphBufferLength, inFifoHandle, inFifoType,
            inNumElem, inDataType, outFifoHandle, outFifoType, outNumElem, outDataType, graphBuffer);

        size_t __total_buffer_size = 0;
        {
            /* Size: struct ncFifoHandle_t ** inFifoHandle */
            if ((inFifoHandle) != (NULL)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }

            /* Size: struct ncFifoHandle_t ** outFifoHandle */
            if ((outFifoHandle) != (NULL)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }

            /* Size: const void * graphBuffer */
            if ((graphBuffer) != (NULL) && (graphBufferLength) > (0)) {
                if (__call->__shm_graphBuffer) {
                } else {

                }
            }
        }
        struct mvnc_nc_graph_allocate_with_fifos_ex_ret *__ret =
            (struct mvnc_nc_graph_allocate_with_fifos_ex_ret *)__chan->cmd_new(__chan,
            sizeof(struct mvnc_nc_graph_allocate_with_fifos_ex_ret), __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS_EX;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: struct ncFifoHandle_t ** inFifoHandle */
        {
            if ((inFifoHandle) != (NULL)) { {
                    const size_t __size_inFifoHandle_0 = ((size_t) (1));
                    struct ncFifoHandle_t **__tmp_inFifoHandle_0;
                    __tmp_inFifoHandle_0 =
                        (struct ncFifoHandle_t **)calloc(1, __size_inFifoHandle_0 * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphAllocateWithFifosEx, kava_buffer_with_deallocator_new(free,
                            __tmp_inFifoHandle_0));
                    const size_t __inFifoHandle_size_0 = __size_inFifoHandle_0;
                    size_t __inFifoHandle_index_0;
                    for (__inFifoHandle_index_0 = 0; __inFifoHandle_index_0 < __inFifoHandle_size_0;
                        __inFifoHandle_index_0++) {
                        const size_t ava_index = __inFifoHandle_index_0;

                        struct ncFifoHandle_t **__inFifoHandle_a_0;
                        __inFifoHandle_a_0 = (struct ncFifoHandle_t **)(__tmp_inFifoHandle_0) + __inFifoHandle_index_0;

                        struct ncFifoHandle_t **__inFifoHandle_b_0;
                        __inFifoHandle_b_0 = (struct ncFifoHandle_t **)(inFifoHandle) + __inFifoHandle_index_0;

                        {
                            *__inFifoHandle_a_0 =
                                (struct ncFifoHandle_t *)*__inFifoHandle_b_0;
                        }
                    }
                    __ret->inFifoHandle =
                        (struct ncFifoHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_inFifoHandle_0, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }} else {
                __ret->inFifoHandle = NULL;
            }
        }
/* Output: struct ncFifoHandle_t ** outFifoHandle */
        {
            if ((outFifoHandle) != (NULL)) { {
                    const size_t __size_outFifoHandle_0 = ((size_t) (1));
                    struct ncFifoHandle_t **__tmp_outFifoHandle_0;
                    __tmp_outFifoHandle_0 =
                        (struct ncFifoHandle_t **)calloc(1, __size_outFifoHandle_0 * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncGraphAllocateWithFifosEx, kava_buffer_with_deallocator_new(free,
                            __tmp_outFifoHandle_0));
                    const size_t __outFifoHandle_size_0 = __size_outFifoHandle_0;
                    size_t __outFifoHandle_index_0;
                    for (__outFifoHandle_index_0 = 0; __outFifoHandle_index_0 < __outFifoHandle_size_0;
                        __outFifoHandle_index_0++) {
                        const size_t ava_index = __outFifoHandle_index_0;

                        struct ncFifoHandle_t **__outFifoHandle_a_0;
                        __outFifoHandle_a_0 =
                            (struct ncFifoHandle_t **)(__tmp_outFifoHandle_0) + __outFifoHandle_index_0;

                        struct ncFifoHandle_t **__outFifoHandle_b_0;
                        __outFifoHandle_b_0 = (struct ncFifoHandle_t **)(outFifoHandle) + __outFifoHandle_index_0;

                        {
                            *__outFifoHandle_a_0 =
                                (struct ncFifoHandle_t *)*__outFifoHandle_b_0;
                        }
                    }
                    __ret->outFifoHandle =
                        (struct ncFifoHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_outFifoHandle_0, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }} else {
                __ret->outFifoHandle = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncGraphAllocateWithFifosEx); /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_FIFO_CREATE:{
        GPtrArray *__kava_alloc_list_ncFifoCreate =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_fifo_create_call *__call = (struct mvnc_nc_fifo_create_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_fifo_create_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: const char * ava_name */
        char *ava_name; {
            ava_name =
                ((__call->ava_name) != (NULL)) ? ((const char *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->ava_name)) : ((const char *)__call->ava_name);
            if ((__call->ava_name) != (NULL)) {
                char *__src_ava_name_0;
                __src_ava_name_0 = ava_name;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (strlen(ava_name) + 1));
                ava_name = (const char *)__chan->chan_get_buffer(__chan, __cmd, __call->ava_name);

                if ((ava_name) != (__src_ava_name_0)) {
                    memcpy(ava_name, __src_ava_name_0, __buffer_size * sizeof(const char));
                }
            } else {
                ava_name =
                    ((__call->ava_name) != (NULL)) ? ((const char *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->ava_name)) : ((const char *)__call->ava_name);
        }}

        /* Input: ncFifoType_t ava_type */
        ncFifoType_t ava_type; {
            ava_type = (ncFifoType_t) __call->ava_type;
            ava_type = __call->ava_type;
        }

        /* Input: struct ncFifoHandle_t ** fifoHandle */
        struct ncFifoHandle_t **fifoHandle; {
            fifoHandle =
                ((__call->fifoHandle) != (NULL)) ? ((struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->fifoHandle)) : ((struct ncFifoHandle_t **)__call->fifoHandle);
            if ((__call->fifoHandle) != (NULL)) { {
                    const size_t __size = ((size_t) (1));
                    fifoHandle = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncFifoCreate, kava_buffer_with_deallocator_new(free, fifoHandle));
            }}
        }

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncFifoCreate(ava_name, ava_type, fifoHandle);

        size_t __total_buffer_size = 0;
        {
            /* Size: struct ncFifoHandle_t ** fifoHandle */
            if ((fifoHandle) != (NULL)) {
                __total_buffer_size +=
                    __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }
        }
        struct mvnc_nc_fifo_create_ret *__ret =
            (struct mvnc_nc_fifo_create_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_fifo_create_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_FIFO_CREATE;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: struct ncFifoHandle_t ** fifoHandle */
        {
            if ((fifoHandle) != (NULL)) { {
                    const size_t __size_fifoHandle_0 = ((size_t) (1));
                    struct ncFifoHandle_t **__tmp_fifoHandle_0;
                    __tmp_fifoHandle_0 =
                        (struct ncFifoHandle_t **)calloc(1, __size_fifoHandle_0 * sizeof(struct ncFifoHandle_t *));
                    g_ptr_array_add(__kava_alloc_list_ncFifoCreate, kava_buffer_with_deallocator_new(free,
                            __tmp_fifoHandle_0));
                    const size_t __fifoHandle_size_0 = __size_fifoHandle_0;
                    size_t __fifoHandle_index_0;
                    for (__fifoHandle_index_0 = 0; __fifoHandle_index_0 < __fifoHandle_size_0; __fifoHandle_index_0++) {
                        const size_t ava_index = __fifoHandle_index_0;

                        struct ncFifoHandle_t **__fifoHandle_a_0;
                        __fifoHandle_a_0 = (struct ncFifoHandle_t **)(__tmp_fifoHandle_0) + __fifoHandle_index_0;

                        struct ncFifoHandle_t **__fifoHandle_b_0;
                        __fifoHandle_b_0 = (struct ncFifoHandle_t **)(fifoHandle) + __fifoHandle_index_0;

                        {
                            *__fifoHandle_a_0 =
                                (struct ncFifoHandle_t *)*__fifoHandle_b_0;
                        }
                    }
                    __ret->fifoHandle =
                        (struct ncFifoHandle_t **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret,
                        __tmp_fifoHandle_0, ((size_t) (1)) * sizeof(struct ncFifoHandle_t *));
            }} else {
                __ret->fifoHandle = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncFifoCreate);       /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_FIFO_ALLOCATE:{
        GPtrArray *__kava_alloc_list_ncFifoAllocate =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_fifo_allocate_call *__call = (struct mvnc_nc_fifo_allocate_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_fifo_allocate_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncFifoHandle_t * fifoHandle */
        struct ncFifoHandle_t *fifoHandle = (struct ncFifoHandle_t *)__call->fifoHandle;

        /* Input: struct ncDeviceHandle_t * device */
        struct ncDeviceHandle_t *device = (struct ncDeviceHandle_t *)__call->device;

        /* Input: struct ncTensorDescriptor_t * tensorDesc */
        struct ncTensorDescriptor_t *tensorDesc; {
            tensorDesc =
                ((__call->tensorDesc) != (NULL)) ? ((struct ncTensorDescriptor_t *)__chan->chan_get_buffer(__chan,
                    __cmd, __call->tensorDesc)) : ((struct ncTensorDescriptor_t *)__call->tensorDesc);
            if ((__call->tensorDesc) != (NULL)) {
                struct ncTensorDescriptor_t *__src_tensorDesc_0;
                __src_tensorDesc_0 = tensorDesc;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                tensorDesc = (struct ncTensorDescriptor_t *)__chan->chan_get_buffer(__chan, __cmd, __call->tensorDesc);

                if ((tensorDesc) != (__src_tensorDesc_0)) {
                    memcpy(tensorDesc, __src_tensorDesc_0, __buffer_size * sizeof(struct ncTensorDescriptor_t));
                }
            } else {
                tensorDesc =
                    ((__call->tensorDesc) != (NULL)) ? ((struct ncTensorDescriptor_t *)__chan->chan_get_buffer(__chan,
                        __cmd, __call->tensorDesc)) : ((struct ncTensorDescriptor_t *)__call->tensorDesc);
        }}

        /* Input: unsigned int numElem */
        unsigned int numElem; {
            numElem = (unsigned int)__call->numElem;
            numElem = __call->numElem;
        }

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncFifoAllocate(fifoHandle, device, tensorDesc, numElem);

        size_t __total_buffer_size = 0;
        {
        }
        struct mvnc_nc_fifo_allocate_ret *__ret =
            (struct mvnc_nc_fifo_allocate_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_fifo_allocate_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_FIFO_ALLOCATE;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncFifoAllocate);     /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_FIFO_SET_OPTION:{
        GPtrArray *__kava_alloc_list_ncFifoSetOption =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_fifo_set_option_call *__call = (struct mvnc_nc_fifo_set_option_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_fifo_set_option_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncFifoHandle_t * fifoHandle */
        struct ncFifoHandle_t *fifoHandle = (struct ncFifoHandle_t *)__call->fifoHandle;

        /* Input: int option */
        int option; {
            option = (int)__call->option;
            option = __call->option;
        }

        /* Input: unsigned int dataLength */
        unsigned int dataLength; {
            dataLength = (unsigned int)__call->dataLength;
            dataLength = __call->dataLength;
        }

        /* Input: const void * data */
        void *data; {
            data =
                ((__call->data) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->data)) : ((const void *)__call->data);
            if ((__call->data) != (NULL)) {
                if (__call->__shm_data) {
                    data = kava_shm_address((long)__call->data);
                } else {

                    void *__src_data_0;
                    __src_data_0 = data;
                    volatile size_t __buffer_size = 0;
                    __buffer_size = ((size_t) (dataLength));
                    data = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->data);

                    if ((data) != (__src_data_0)) {
                        memcpy(data, __src_data_0, __buffer_size * sizeof(const void));
                    }
            }} else {
                data =
                    ((__call->data) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->data)) : ((const void *)__call->data);
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncFifoSetOption(fifoHandle, option, dataLength, data);

        size_t __total_buffer_size = 0;
        {
            /* Size: const void * data */
            if ((data) != (NULL) && (dataLength) > (0)) {
                if (__call->__shm_data) {
                } else {

                }
            }
        }
        struct mvnc_nc_fifo_set_option_ret *__ret =
            (struct mvnc_nc_fifo_set_option_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_fifo_set_option_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_FIFO_SET_OPTION;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncFifoSetOption);    /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_FIFO_GET_OPTION:{
        GPtrArray *__kava_alloc_list_ncFifoGetOption =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_fifo_get_option_call *__call = (struct mvnc_nc_fifo_get_option_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_fifo_get_option_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncFifoHandle_t * fifoHandle */
        struct ncFifoHandle_t *fifoHandle = (struct ncFifoHandle_t *)__call->fifoHandle;

        /* Input: int option */
        int option; {
            option = (int)__call->option;
            option = __call->option;
        }

        /* Input: unsigned int * dataLength */
        unsigned int *dataLength; {
            dataLength =
                ((__call->dataLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->dataLength)) : ((unsigned int *)__call->dataLength);
            if ((__call->dataLength) != (NULL)) {
                unsigned int *__src_dataLength_0;
                __src_dataLength_0 = dataLength;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                dataLength = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __call->dataLength);

                if ((dataLength) != (__src_dataLength_0)) {
                    memcpy(dataLength, __src_dataLength_0, __buffer_size * sizeof(unsigned int));
                }
            } else {
                dataLength =
                    ((__call->dataLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->dataLength)) : ((unsigned int *)__call->dataLength);
        }}

        /* Input: void * data */
        void *data; {
            data =
                ((__call->data) != (NULL)) ? ((void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->data)) : ((void *)__call->data);
            if ((__call->data) != (NULL)) { {
                    const size_t __size = ((size_t) (*dataLength));
                    data = (void *)malloc(__size * sizeof(void));
                    g_ptr_array_add(__kava_alloc_list_ncFifoGetOption, kava_buffer_with_deallocator_new(free, data));
            }}
        }

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncFifoGetOption(fifoHandle, option, dataLength, data);

        size_t __total_buffer_size = 0;
        {
            /* Size: unsigned int * dataLength */
            if ((dataLength) != (NULL)) {
                __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(unsigned int));
            }

            /* Size: void * data */
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (__call->__shm_data) {
                } else {
                    __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (*dataLength)) * sizeof(void));
            }}
        }
        struct mvnc_nc_fifo_get_option_ret *__ret =
            (struct mvnc_nc_fifo_get_option_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_fifo_get_option_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_FIFO_GET_OPTION;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: unsigned int * dataLength */
        {
            if ((dataLength) != (NULL)) {
                __ret->dataLength =
                    (unsigned int *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, dataLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __ret->dataLength = NULL;
            }
        }
/* Output: void * data */
        {
            if ((data) != (NULL) && (*dataLength) > (0)) {
                if (__call->__shm_data) {
                    __ret->data = __call->data;
                } else {
                    __ret->data =
                        (void *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, data,
                        ((size_t) (*dataLength)) * sizeof(void));
            }} else {
                __ret->data = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncFifoGetOption);    /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_FIFO_DESTROY:{
        GPtrArray *__kava_alloc_list_ncFifoDestroy =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_fifo_destroy_call *__call = (struct mvnc_nc_fifo_destroy_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_fifo_destroy_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncFifoHandle_t ** fifoHandle */
        struct ncFifoHandle_t **fifoHandle; {
            fifoHandle =
                ((__call->fifoHandle) != (NULL)) ? ((struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->fifoHandle)) : ((struct ncFifoHandle_t **)__call->fifoHandle);
            if ((__call->fifoHandle) != (NULL)) {
                struct ncFifoHandle_t **__src_fifoHandle_0;
                __src_fifoHandle_0 = fifoHandle;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                fifoHandle = (struct ncFifoHandle_t **)__chan->chan_get_buffer(__chan, __cmd, __call->fifoHandle);
                if ((__call->fifoHandle) != (NULL)) { {
                        const size_t __size = ((size_t) (1));
                        fifoHandle = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncFifoDestroy, kava_buffer_with_deallocator_new(free,
                                fifoHandle));
                }}

                const size_t __fifoHandle_size_0 = __buffer_size;
                size_t __fifoHandle_index_0;
                for (__fifoHandle_index_0 = 0; __fifoHandle_index_0 < __fifoHandle_size_0; __fifoHandle_index_0++) {
                    const size_t ava_index = __fifoHandle_index_0;

                    struct ncFifoHandle_t **__fifoHandle_a_0;
                    __fifoHandle_a_0 = (struct ncFifoHandle_t **)(fifoHandle) + __fifoHandle_index_0;

                    struct ncFifoHandle_t **__fifoHandle_b_0;
                    __fifoHandle_b_0 = (struct ncFifoHandle_t **)(__src_fifoHandle_0) + __fifoHandle_index_0;

                    *__fifoHandle_a_0 = (struct ncFifoHandle_t *)*__fifoHandle_b_0;
            }} else {
                if ((__call->fifoHandle) != (NULL)) { {
                        const size_t __size = ((size_t) (1));
                        fifoHandle = (struct ncFifoHandle_t **)malloc(__size * sizeof(struct ncFifoHandle_t *));
                        g_ptr_array_add(__kava_alloc_list_ncFifoDestroy, kava_buffer_with_deallocator_new(free,
                                fifoHandle));
                }}
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncFifoDestroy(fifoHandle);

        size_t __total_buffer_size = 0;
        {
        }
        struct mvnc_nc_fifo_destroy_ret *__ret =
            (struct mvnc_nc_fifo_destroy_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_fifo_destroy_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_FIFO_DESTROY;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncFifoDestroy);      /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_FIFO_WRITE_ELEM:{
        GPtrArray *__kava_alloc_list_ncFifoWriteElem =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_fifo_write_elem_call *__call = (struct mvnc_nc_fifo_write_elem_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_fifo_write_elem_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncFifoHandle_t * fifoHandle */
        struct ncFifoHandle_t *fifoHandle = (struct ncFifoHandle_t *)__call->fifoHandle;

        /* Input: unsigned int * inputTensorLength */
        unsigned int *inputTensorLength; {
            inputTensorLength =
                ((__call->inputTensorLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->inputTensorLength)) : ((unsigned int *)__call->inputTensorLength);
            if ((__call->inputTensorLength) != (NULL)) {
                unsigned int *__src_inputTensorLength_0;
                __src_inputTensorLength_0 = inputTensorLength;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                inputTensorLength = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __call->inputTensorLength);

                if ((inputTensorLength) != (__src_inputTensorLength_0)) {
                    memcpy(inputTensorLength, __src_inputTensorLength_0, __buffer_size * sizeof(unsigned int));
                }
            } else {
                inputTensorLength =
                    ((__call->inputTensorLength) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->inputTensorLength)) : ((unsigned int *)__call->inputTensorLength);
        }}

        /* Input: void * userParam */
        void *userParam; {
            userParam = (void *)__call->userParam;
            userParam = __call->userParam;
        }

        /* Input: const void * inputTensor */
        void *inputTensor; {
            inputTensor =
                ((__call->inputTensor) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->inputTensor)) : ((const void *)__call->inputTensor);
            if ((__call->inputTensor) != (NULL)) {
                if (__call->__shm_inputTensor) {
                    inputTensor = kava_shm_address((long)__call->inputTensor);
                } else {

                    void *__src_inputTensor_0;
                    __src_inputTensor_0 = inputTensor;
                    volatile size_t __buffer_size = 0;
                    __buffer_size = ((size_t) (*inputTensorLength));
                    inputTensor = (const void *)__chan->chan_get_buffer(__chan, __cmd, __call->inputTensor);

                    if ((inputTensor) != (__src_inputTensor_0)) {
                        memcpy(inputTensor, __src_inputTensor_0, __buffer_size * sizeof(const void));
                    }
            }} else {
                inputTensor =
                    ((__call->inputTensor) != (NULL)) ? ((const void *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->inputTensor)) : ((const void *)__call->inputTensor);
        }}

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncFifoWriteElem(fifoHandle, inputTensorLength, userParam, inputTensor);

        size_t __total_buffer_size = 0;
        {
            /* Size: unsigned int * inputTensorLength */
            if ((inputTensorLength) != (NULL)) {
                __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(unsigned int));
            }

            /* Size: const void * inputTensor */
            if ((inputTensor) != (NULL) && (*inputTensorLength) > (0)) {
                if (__call->__shm_inputTensor) {
                } else {

                }
            }
        }
        struct mvnc_nc_fifo_write_elem_ret *__ret =
            (struct mvnc_nc_fifo_write_elem_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_fifo_write_elem_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_FIFO_WRITE_ELEM;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: unsigned int * inputTensorLength */
        {
            if ((inputTensorLength) != (NULL)) {
                __ret->inputTensorLength =
                    (unsigned int *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, inputTensorLength,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __ret->inputTensorLength = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncFifoWriteElem);    /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_FIFO_READ_ELEM:{
        GPtrArray *__kava_alloc_list_ncFifoReadElem =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_fifo_read_elem_call *__call = (struct mvnc_nc_fifo_read_elem_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_fifo_read_elem_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncFifoHandle_t * fifoHandle */
        struct ncFifoHandle_t *fifoHandle = (struct ncFifoHandle_t *)__call->fifoHandle;

        /* Input: unsigned int * outputDataLen */
        unsigned int *outputDataLen; {
            outputDataLen =
                ((__call->outputDataLen) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->outputDataLen)) : ((unsigned int *)__call->outputDataLen);
            if ((__call->outputDataLen) != (NULL)) {
                unsigned int *__src_outputDataLen_0;
                __src_outputDataLen_0 = outputDataLen;
                volatile size_t __buffer_size = 0;
                __buffer_size = ((size_t) (1));
                outputDataLen = (unsigned int *)__chan->chan_get_buffer(__chan, __cmd, __call->outputDataLen);

                if ((outputDataLen) != (__src_outputDataLen_0)) {
                    memcpy(outputDataLen, __src_outputDataLen_0, __buffer_size * sizeof(unsigned int));
                }
            } else {
                outputDataLen =
                    ((__call->outputDataLen) != (NULL)) ? ((unsigned int *)__chan->chan_get_buffer(__chan, __cmd,
                        __call->outputDataLen)) : ((unsigned int *)__call->outputDataLen);
        }}

        /* Input: void ** userParam */
        void **userParam; {
            userParam =
                ((__call->userParam) != (NULL)) ? ((void **)__chan->chan_get_buffer(__chan, __cmd,
                    __call->userParam)) : ((void **)__call->userParam);
            if ((__call->userParam) != (NULL)) { {
                    const size_t __size = ((size_t) (1));
                    userParam = (void **)malloc(__size * sizeof(void *));
                    g_ptr_array_add(__kava_alloc_list_ncFifoReadElem, kava_buffer_with_deallocator_new(free, userParam));
            }}
        }

        /* Input: void * outputData */
        void *outputData; {
            outputData =
                ((__call->outputData) != (NULL)) ? ((void *)__chan->chan_get_buffer(__chan, __cmd,
                    __call->outputData)) : ((void *)__call->outputData);
            if ((__call->outputData) != (NULL)) { {
                    const size_t __size = ((size_t) (*outputDataLen));
                    outputData = (void *)malloc(__size * sizeof(void));
                    g_ptr_array_add(__kava_alloc_list_ncFifoReadElem, kava_buffer_with_deallocator_new(free, outputData));
            }}
        }

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncFifoReadElem(fifoHandle, outputDataLen, userParam, outputData);

        size_t __total_buffer_size = 0;
        {
            /* Size: unsigned int * outputDataLen */
            if ((outputDataLen) != (NULL)) {
                __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(unsigned int));
            }

            /* Size: void ** userParam */
            if ((userParam) != (NULL)) {
                __total_buffer_size += __chan->chan_buffer_size(__chan, ((size_t) (1)) * sizeof(void *));
            }

            /* Size: void * outputData */
            if ((outputData) != (NULL) && (*outputDataLen) > (0)) {
                if (__call->__shm_outputData) {
                } else {
                    __total_buffer_size +=
                        __chan->chan_buffer_size(__chan, ((size_t) (*outputDataLen)) * sizeof(void));
            }}
        }
        struct mvnc_nc_fifo_read_elem_ret *__ret =
            (struct mvnc_nc_fifo_read_elem_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_fifo_read_elem_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_FIFO_READ_ELEM;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }
        /* Output: unsigned int * outputDataLen */
        {
            if ((outputDataLen) != (NULL)) {
                __ret->outputDataLen =
                    (unsigned int *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, outputDataLen,
                    ((size_t) (1)) * sizeof(unsigned int));
            } else {
                __ret->outputDataLen = NULL;
            }
        }
/* Output: void ** userParam */
        {
            if ((userParam) != (NULL)) {
                __ret->userParam =
                    (void **)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, userParam,
                    ((size_t) (1)) * sizeof(void *));
            } else {
                __ret->userParam = NULL;
            }
        }
/* Output: void * outputData */
        {
            if ((outputData) != (NULL) && (*outputDataLen) > (0)) {
                if (__call->__shm_outputData) {
                    __ret->outputData = __call->outputData;
                } else {
                    __ret->outputData =
                        (void *)__chan->chan_attach_buffer(__chan, (struct kava_cmd_base *)__ret, outputData,
                        ((size_t) (*outputDataLen)) * sizeof(void));
            }} else {
                __ret->outputData = NULL;
            }
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncFifoReadElem);     /* Deallocate all memory in the alloc list */

        break;
    }
    case CALL_MVNC_NC_FIFO_REMOVE_ELEM:{
        GPtrArray *__kava_alloc_list_ncFifoRemoveElem =
            g_ptr_array_new_full(0, (GDestroyNotify) kava_buffer_with_deallocator_free);
        struct mvnc_nc_fifo_remove_elem_call *__call = (struct mvnc_nc_fifo_remove_elem_call *)__cmd;
        assert(__call->base.mode == KAVA_CMD_MODE_API);
        assert(__call->base.command_size == sizeof(struct mvnc_nc_fifo_remove_elem_call)
            &&
            "Command size does not match ID. (Can be caused by incorrectly computed buffer sizes, especially using `strlen(s)` instead of `strlen(s)+1`)");

        /* Unpack and translate arguments */

        /* Input: struct ncFifoHandle_t * fifoHandle */
        struct ncFifoHandle_t *fifoHandle = (struct ncFifoHandle_t *)__call->fifoHandle;

        /* Perform Call */

        ncStatus_t ret;
        ret = __wrapper_ncFifoRemoveElem(fifoHandle);

        size_t __total_buffer_size = 0;
        {
        }
        struct mvnc_nc_fifo_remove_elem_ret *__ret =
            (struct mvnc_nc_fifo_remove_elem_ret *)__chan->cmd_new(__chan, sizeof(struct mvnc_nc_fifo_remove_elem_ret),
            __total_buffer_size);
        __ret->base.mode = KAVA_CMD_MODE_API;
        __ret->base.command_id = RET_MVNC_NC_FIFO_REMOVE_ELEM;
        __ret->base.thread_id = __call->base.thread_id;
        __ret->__call_id = __call->__call_id;

        /* Output: ncStatus_t ret */
        {
            __ret->ret = ret;
        }

        /* Send reply message */
        __chan->cmd_send(__chan, (struct kava_cmd_base *)__ret);
        g_ptr_array_unref(__kava_alloc_list_ncFifoRemoveElem);   /* Deallocate all memory in the alloc list */

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
