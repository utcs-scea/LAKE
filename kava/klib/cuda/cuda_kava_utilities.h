#ifndef __CUDA_KAVA_UTILITIES_H__
#define __CUDA_KAVA_UTILITIES_H__

#define kava_utility static
#define kava_begin_utility
#define kava_end_utility

kava_begin_utility;
#include <linux/ctype.h>
#include <linux/time.h>
kava_end_utility;

#include "cuda.h"

kava_utility void kava_parse_function_args(const char *name, int *func_argc,
                                         char *func_arg_is_handle,
                                         size_t *func_arg_size)
{
    int i = 0, skip = 0;

    *func_argc = 0;
    if (strncmp(name, "_Z", 2)) {
        BUG_ON("Wrong CUDA function name");
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
                if (i + 1 < strlen(name) &&
                        (name[i+1] == 'f' || name[i+1] == 'i' || name[i+1] == 'j' ||
                         name[i+1] == 'l' || name[i+1] == 'h' || name[i+1] == 'c' || 
                         name[i+1] == 'v'))
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
                    BUG_ON("CUDA function argument: wrong pointer");
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
                BUG_ON("CUDA function argument: unrecognized type");
                return;
        }
        i++;
    }

    for (i = 0; i < *func_argc; i++) {
        DEBUG_PRINT("function arg#%d it is %sa handle\n", i, func_arg_is_handle[i]?"":"not ");
    }
}

kava_utility size_t cuLaunchKernel_extra_size(void **extra) {
    size_t size = 1;
    while (extra[size - 1] != CU_LAUNCH_PARAM_END)
        size++;
    return size;
}

inline void print_timestamp(const char *name) {
    struct timespec ts;
    getnstimeofday(&ts);
    //pr_info("Timestamp at %s: sec=%lu, usec=%lu\n", name, ts.tv_sec, ts.tv_nsec / 1000);

    //pr_info("Timestamp at %s: %lu \n", name, ts.tv_sec*1000000 + ts.tv_nsec / 1000);
}
#undef ava_utility
#undef ava_begin_utility
#undef ava_end_utility

#endif // undef __CUDA_KAVA_UTILITIES_H__
