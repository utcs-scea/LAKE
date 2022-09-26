#ifndef __KAPI_CUDA_UTIL_H__
#define __KAPI_CUDA_UTIL_H__

#ifdef __KERNEL__
#include <linux/ctype.h>
#include <linux/time.h>
#include <linux/string.h>
#define PRINT(...) pr_err (__VA_ARGS__)
#else
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#define PRINT(...) printf (__VA_ARGS__)
typedef unsigned char u8;
#endif

#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_kargs_kv(void);
void destroy_kargs_kv(void);
struct kernel_args_metadata* get_kargs(const void* ptr);

#ifdef __cplusplus
}
#endif

struct kernel_args_metadata {
    int func_argc;
    size_t total_size;
    char func_arg_is_handle[64];
    size_t func_arg_size[64];
};

static inline void serialize_args(struct kernel_args_metadata* meta,
                u8* buf, void** args)
{
    int i;
    for (i = 0 ; i < meta->func_argc ; i++) {
// #ifdef __KERNEL__
//         pr_err("arg %d: %lu\n", i, meta->func_arg_size[i]);
//         if(meta->func_arg_size[i] == 8)
//             pr_err("   %llx\n", *((u64*) args[i]));
//         if(meta->func_arg_size[i] == 4)
//             pr_err("   %x\n", *((u32*) args[i]));
// #endif
        memcpy(buf, args[i], meta->func_arg_size[i]);
        buf += meta->func_arg_size[i];
    }
}

static inline void construct_args(struct kernel_args_metadata* meta,
                void** args, u8* buf)
{
    int i;
    for (i = 0 ; i < meta->func_argc ; i++) {
        args[i] = (void*) buf;     
// #ifndef __KERNEL__
//         printf("arg %d: %lu\n", i, meta->func_arg_size[i]);
//         if(meta->func_arg_size[i] == 8)
//             printf(" 8B:  %lx\n", *((uint64_t*) args[i]));
//         if(meta->func_arg_size[i] == 4)
//             printf(" 4B:  %x\n", *((uint32_t*) args[i]));
// #endif
        buf += meta->func_arg_size[i];
    }
}

static inline void kava_parse_function_args(const char *name, 
            struct kernel_args_metadata* meta)
{
    int *func_argc = &meta->func_argc;
    char *func_arg_is_handle = meta->func_arg_is_handle;
    size_t *func_arg_size = meta->func_arg_size;
    int i = 0, skip = 0;

    *func_argc = 0;
    if (strncmp(name, "_Z", 2)) {
        PRINT("Wrong CUDA function name");
        return;
    }

    i = 2;
    while (i < strlen(name) && isdigit(name[i])) {
        skip = skip * 10 + name[i] - '0';
        i++;
    }

    i += skip;
    while (i < strlen(name)) {
        switch(name[i]) {
            case 'P':
                func_arg_size[(*func_argc)] = sizeof(CUdeviceptr);
                func_arg_is_handle[(*func_argc)++] = 1;
                //pr_info("case P, next: %c at %d\n", name[i+1], i);
                if (i + 1 < strlen(name) &&
                        (name[i+1] == 'f' || name[i+1] == 'i' || name[i+1] == 'j' ||
                         name[i+1] == 'l' || name[i+1] == 'h' || name[i+1] == 'c' || 
                         name[i+1] == 'v' || name[i+1] == 'm'))
                    i++;
                else if (i + 1 < strlen(name) && isdigit(name[i+1])) {
                    skip = 0;
                    while (i + 1 < strlen(name) && isdigit(name[i+1])) {
                        skip = skip * 10 + name[i+1] - '0';
                        i++;
                    }
                    i += skip;
                }
                else {
                    PRINT("CUDA function argument: wrong pointer");
                    return;
                }
                break;

            case 'f':
            case 'i': // int
            case 'j': // unsigned int
                func_arg_size[(*func_argc)] = sizeof(int);
                func_arg_is_handle[(*func_argc)++] = 0;
                break;

            case 'l':
                func_arg_size[(*func_argc)] = sizeof(long);
                func_arg_is_handle[(*func_argc)++] = 0;
                break;

            case 'c': // char
            case 'h': // unsigned char
                func_arg_size[(*func_argc)] = sizeof(char);
                func_arg_is_handle[(*func_argc)++] = 0;
                break;

            case 'S':
                func_arg_size[(*func_argc)] = sizeof(CUdeviceptr);
                func_arg_is_handle[(*func_argc)++] = 1;
                while (i < strlen(name) && name[i] != '_') i++;
                break;

            case 'v':
                i = strlen(name);
                break;

            default:
                PRINT("CUDA function argument: unrecognized type");
                return;
        }
        i++;
    }

    meta->total_size = 0;
    for (i = 0 ; i < *func_argc ; i++) {
        meta->total_size += func_arg_size[i];
    }
    //PRINT("size of args for name: %lu\n", meta->total_size);
}


#endif