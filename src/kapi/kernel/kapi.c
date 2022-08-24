#include <linux/types.h>
#include <linux/module.h>
#include "commands.h"
#include "lake_kapi.h"

//#define ALLOC(s) kmalloc(s)
//#define FREE(p) kfree(p)

CUresult CUDAAPI cuInit(unsigned int flags) {
    struct lake_cmd_cuInit cmd = {
        .API_ID = LAKE_API_cuInit, .flags = flags,
    };
    lake_send_cmd((void*)&cmd, sizeof(cmd), CMD_SYNC);
    return CUDA_SUCCESS;
}
EXPORT_SYMBOL(cuInit);