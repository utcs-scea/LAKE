#ifndef __MVNC_NW_H__
#define __MVNC_NW_H__

#include "command.h"
#include "util.h"

#include "mvnc_nw_types.h"

#define MVNC_API 1

enum mvnc_functions {
    CALL_MVNC_NC_GLOBAL_SET_OPTION, RET_MVNC_NC_GLOBAL_SET_OPTION, CALL_MVNC_NC_GLOBAL_GET_OPTION,
        RET_MVNC_NC_GLOBAL_GET_OPTION, CALL_MVNC_NC_DEVICE_SET_OPTION, RET_MVNC_NC_DEVICE_SET_OPTION,
        CALL_MVNC_NC_DEVICE_GET_OPTION, RET_MVNC_NC_DEVICE_GET_OPTION, CALL_MVNC_NC_DEVICE_CREATE,
        RET_MVNC_NC_DEVICE_CREATE, CALL_MVNC_NC_DEVICE_OPEN, RET_MVNC_NC_DEVICE_OPEN, CALL_MVNC_NC_DEVICE_CLOSE,
        RET_MVNC_NC_DEVICE_CLOSE, CALL_MVNC_NC_DEVICE_DESTROY, RET_MVNC_NC_DEVICE_DESTROY, CALL_MVNC_NC_GRAPH_CREATE,
        RET_MVNC_NC_GRAPH_CREATE, CALL_MVNC_NC_GRAPH_ALLOCATE, RET_MVNC_NC_GRAPH_ALLOCATE, CALL_MVNC_NC_GRAPH_DESTROY,
        RET_MVNC_NC_GRAPH_DESTROY, CALL_MVNC_NC_GRAPH_SET_OPTION, RET_MVNC_NC_GRAPH_SET_OPTION,
        CALL_MVNC_NC_GRAPH_GET_OPTION, RET_MVNC_NC_GRAPH_GET_OPTION, CALL_MVNC_NC_GRAPH_QUEUE_INFERENCE,
        RET_MVNC_NC_GRAPH_QUEUE_INFERENCE, CALL_MVNC_NC_GRAPH_QUEUE_INFERENCE_WITH_FIFO_ELEM,
        RET_MVNC_NC_GRAPH_QUEUE_INFERENCE_WITH_FIFO_ELEM, CALL_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS,
        RET_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS, CALL_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS_EX,
        RET_MVNC_NC_GRAPH_ALLOCATE_WITH_FIFOS_EX, CALL_MVNC_NC_FIFO_CREATE, RET_MVNC_NC_FIFO_CREATE,
        CALL_MVNC_NC_FIFO_ALLOCATE, RET_MVNC_NC_FIFO_ALLOCATE, CALL_MVNC_NC_FIFO_SET_OPTION,
        RET_MVNC_NC_FIFO_SET_OPTION, CALL_MVNC_NC_FIFO_GET_OPTION, RET_MVNC_NC_FIFO_GET_OPTION,
        CALL_MVNC_NC_FIFO_DESTROY, RET_MVNC_NC_FIFO_DESTROY, CALL_MVNC_NC_FIFO_WRITE_ELEM, RET_MVNC_NC_FIFO_WRITE_ELEM,
        CALL_MVNC_NC_FIFO_READ_ELEM, RET_MVNC_NC_FIFO_READ_ELEM, CALL_MVNC_NC_FIFO_REMOVE_ELEM,
        RET_MVNC_NC_FIFO_REMOVE_ELEM
};

#include "mvnc_nw_utility_types.h"
#include "mvnc_nw_utilities.h"

struct mvnc_nc_global_set_option_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int option;
    unsigned int dataLength;
    void *data;
    char __shm_data;
};

struct mvnc_nc_global_set_option_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_global_set_option_call_record {
    int option;
    unsigned int dataLength;
    void *data;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_global_get_option_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int option;
    unsigned int *dataLength;
    void *data;
    char __shm_data;
};

struct mvnc_nc_global_get_option_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int *dataLength;
    void *data;
    ncStatus_t ret;
};

struct mvnc_nc_global_get_option_call_record {
    int option;
    unsigned int *dataLength;
    void *data;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_device_set_option_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t *deviceHandle;
    int option;
    unsigned int dataLength;
    void *data;
    char __shm_data;
};

struct mvnc_nc_device_set_option_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_device_set_option_call_record {
    struct ncDeviceHandle_t *deviceHandle;
    int option;
    unsigned int dataLength;
    void *data;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_device_get_option_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t *deviceHandle;
    int option;
    unsigned int *dataLength;
    void *data;
    char __shm_data;
};

struct mvnc_nc_device_get_option_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int *dataLength;
    void *data;
    ncStatus_t ret;
};

struct mvnc_nc_device_get_option_call_record {
    struct ncDeviceHandle_t *deviceHandle;
    int option;
    unsigned int *dataLength;
    void *data;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_device_create_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    int index;
    struct ncDeviceHandle_t **deviceHandle;
};

struct mvnc_nc_device_create_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t **deviceHandle;
    ncStatus_t ret;
};

struct mvnc_nc_device_create_call_record {
    int index;
    struct ncDeviceHandle_t **deviceHandle;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_device_open_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t *deviceHandle;
};

struct mvnc_nc_device_open_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_device_open_call_record {
    struct ncDeviceHandle_t *deviceHandle;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_device_close_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t *deviceHandle;
};

struct mvnc_nc_device_close_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_device_close_call_record {
    struct ncDeviceHandle_t *deviceHandle;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_device_destroy_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t **deviceHandle;
};

struct mvnc_nc_device_destroy_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t **deviceHandle;
    ncStatus_t ret;
};

struct mvnc_nc_device_destroy_call_record {
    struct ncDeviceHandle_t **deviceHandle;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_graph_create_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    char *ava_name;
    struct ncGraphHandle_t **graphHandle;
};

struct mvnc_nc_graph_create_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncGraphHandle_t **graphHandle;
    ncStatus_t ret;
};

struct mvnc_nc_graph_create_call_record {
    char *ava_name;
    struct ncGraphHandle_t **graphHandle;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_graph_allocate_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t *deviceHandle;
    struct ncGraphHandle_t *graphHandle;
    unsigned int graphBufferLength;
    void *graphBuffer;
    char __shm_graphBuffer;
};

struct mvnc_nc_graph_allocate_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_graph_allocate_call_record {
    struct ncDeviceHandle_t *deviceHandle;
    struct ncGraphHandle_t *graphHandle;
    unsigned int graphBufferLength;
    void *graphBuffer;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_graph_destroy_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncGraphHandle_t **graphHandle;
};

struct mvnc_nc_graph_destroy_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncGraphHandle_t **graphHandle;
    ncStatus_t ret;
};

struct mvnc_nc_graph_destroy_call_record {
    struct ncGraphHandle_t **graphHandle;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_graph_set_option_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncGraphHandle_t *graphHandle;
    int option;
    unsigned int dataLength;
    void *data;
    char __shm_data;
};

struct mvnc_nc_graph_set_option_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_graph_set_option_call_record {
    struct ncGraphHandle_t *graphHandle;
    int option;
    unsigned int dataLength;
    void *data;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_graph_get_option_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncGraphHandle_t *graphHandle;
    int option;
    unsigned int *dataLength;
    void *data;
    char __shm_data;
};

struct mvnc_nc_graph_get_option_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int *dataLength;
    void *data;
    ncStatus_t ret;
};

struct mvnc_nc_graph_get_option_call_record {
    struct ncGraphHandle_t *graphHandle;
    int option;
    unsigned int *dataLength;
    void *data;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_graph_queue_inference_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncGraphHandle_t *graphHandle;
    unsigned int inFifoCount;
    unsigned int outFifoCount;
    struct ncFifoHandle_t **fifoIn;
    struct ncFifoHandle_t **fifoOut;
};

struct mvnc_nc_graph_queue_inference_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t **fifoIn;
    struct ncFifoHandle_t **fifoOut;
    ncStatus_t ret;
};

struct mvnc_nc_graph_queue_inference_call_record {
    struct ncGraphHandle_t *graphHandle;
    unsigned int inFifoCount;
    unsigned int outFifoCount;
    struct ncFifoHandle_t **fifoIn;
    struct ncFifoHandle_t **fifoOut;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_graph_queue_inference_with_fifo_elem_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncGraphHandle_t *graphHandle;
    struct ncFifoHandle_t *fifoIn;
    struct ncFifoHandle_t *fifoOut;
    unsigned int *inputTensorLength;
    void *userParam;
    void *inputTensor;
    char __shm_inputTensor;
};

struct mvnc_nc_graph_queue_inference_with_fifo_elem_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int *inputTensorLength;
    ncStatus_t ret;
};

struct mvnc_nc_graph_queue_inference_with_fifo_elem_call_record {
    struct ncGraphHandle_t *graphHandle;
    struct ncFifoHandle_t *fifoIn;
    struct ncFifoHandle_t *fifoOut;
    unsigned int *inputTensorLength;
    void *userParam;
    void *inputTensor;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_graph_allocate_with_fifos_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t *deviceHandle;
    struct ncGraphHandle_t *graphHandle;
    unsigned int graphBufferLength;
    struct ncFifoHandle_t **inFifoHandle;
    struct ncFifoHandle_t **outFifoHandle;
    void *graphBuffer;
    char __shm_graphBuffer;
};

struct mvnc_nc_graph_allocate_with_fifos_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t **inFifoHandle;
    struct ncFifoHandle_t **outFifoHandle;
    ncStatus_t ret;
};

struct mvnc_nc_graph_allocate_with_fifos_call_record {
    struct ncDeviceHandle_t *deviceHandle;
    struct ncGraphHandle_t *graphHandle;
    unsigned int graphBufferLength;
    struct ncFifoHandle_t **inFifoHandle;
    struct ncFifoHandle_t **outFifoHandle;
    void *graphBuffer;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_graph_allocate_with_fifos_ex_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncDeviceHandle_t *deviceHandle;
    struct ncGraphHandle_t *graphHandle;
    unsigned int graphBufferLength;
    struct ncFifoHandle_t **inFifoHandle;
    ncFifoType_t inFifoType;
    int inNumElem;
    ncFifoDataType_t inDataType;
    struct ncFifoHandle_t **outFifoHandle;
    ncFifoType_t outFifoType;
    int outNumElem;
    ncFifoDataType_t outDataType;
    void *graphBuffer;
    char __shm_graphBuffer;
};

struct mvnc_nc_graph_allocate_with_fifos_ex_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t **inFifoHandle;
    struct ncFifoHandle_t **outFifoHandle;
    ncStatus_t ret;
};

struct mvnc_nc_graph_allocate_with_fifos_ex_call_record {
    struct ncDeviceHandle_t *deviceHandle;
    struct ncGraphHandle_t *graphHandle;
    unsigned int graphBufferLength;
    struct ncFifoHandle_t **inFifoHandle;
    ncFifoType_t inFifoType;
    int inNumElem;
    ncFifoDataType_t inDataType;
    struct ncFifoHandle_t **outFifoHandle;
    ncFifoType_t outFifoType;
    int outNumElem;
    ncFifoDataType_t outDataType;
    void *graphBuffer;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_fifo_create_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    char *ava_name;
    ncFifoType_t ava_type;
    struct ncFifoHandle_t **fifoHandle;
};

struct mvnc_nc_fifo_create_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t **fifoHandle;
    ncStatus_t ret;
};

struct mvnc_nc_fifo_create_call_record {
    char *ava_name;
    ncFifoType_t ava_type;
    struct ncFifoHandle_t **fifoHandle;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_fifo_allocate_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t *fifoHandle;
    struct ncDeviceHandle_t *device;
    struct ncTensorDescriptor_t *tensorDesc;
    unsigned int numElem;
};

struct mvnc_nc_fifo_allocate_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_fifo_allocate_call_record {
    struct ncFifoHandle_t *fifoHandle;
    struct ncDeviceHandle_t *device;
    struct ncTensorDescriptor_t *tensorDesc;
    unsigned int numElem;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_fifo_set_option_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t *fifoHandle;
    int option;
    unsigned int dataLength;
    void *data;
    char __shm_data;
};

struct mvnc_nc_fifo_set_option_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_fifo_set_option_call_record {
    struct ncFifoHandle_t *fifoHandle;
    int option;
    unsigned int dataLength;
    void *data;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_fifo_get_option_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t *fifoHandle;
    int option;
    unsigned int *dataLength;
    void *data;
    char __shm_data;
};

struct mvnc_nc_fifo_get_option_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int *dataLength;
    void *data;
    ncStatus_t ret;
};

struct mvnc_nc_fifo_get_option_call_record {
    struct ncFifoHandle_t *fifoHandle;
    int option;
    unsigned int *dataLength;
    void *data;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_fifo_destroy_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t **fifoHandle;
};

struct mvnc_nc_fifo_destroy_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_fifo_destroy_call_record {
    struct ncFifoHandle_t **fifoHandle;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_fifo_write_elem_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t *fifoHandle;
    unsigned int *inputTensorLength;
    void *userParam;
    void *inputTensor;
    char __shm_inputTensor;
};

struct mvnc_nc_fifo_write_elem_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int *inputTensorLength;
    ncStatus_t ret;
};

struct mvnc_nc_fifo_write_elem_call_record {
    struct ncFifoHandle_t *fifoHandle;
    unsigned int *inputTensorLength;
    void *userParam;
    void *inputTensor;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_fifo_read_elem_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t *fifoHandle;
    unsigned int *outputDataLen;
    void **userParam;
    void *outputData;
    char __shm_outputData;
};

struct mvnc_nc_fifo_read_elem_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    unsigned int *outputDataLen;
    void **userParam;
    void *outputData;
    ncStatus_t ret;
};

struct mvnc_nc_fifo_read_elem_call_record {
    struct ncFifoHandle_t *fifoHandle;
    unsigned int *outputDataLen;
    void **userParam;
    void *outputData;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

struct mvnc_nc_fifo_remove_elem_call {
    struct kava_cmd_base base;
    intptr_t __call_id;
    struct ncFifoHandle_t *fifoHandle;
};

struct mvnc_nc_fifo_remove_elem_ret {
    struct kava_cmd_base base;
    intptr_t __call_id;
    ncStatus_t ret;
};

struct mvnc_nc_fifo_remove_elem_call_record {
    struct ncFifoHandle_t *fifoHandle;
    ncStatus_t ret;
    char __handler_deallocate;
    volatile char __call_complete;
};

#endif                                           // ndef __MVNC_NW_H__
