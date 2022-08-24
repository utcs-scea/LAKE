#ifndef __KAPI_CUDA_UTIL_H__
#define __KAPI_CUDA_UTIL_H__

#include <linux/ctype.h>
#include <linux/time.h>

void kava_parse_function_args(const char *name, int *func_argc,
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

    //for (i = 0; i < *func_argc; i++) {
    //    DEBUG_PRINT("function arg#%d it is %sa handle\n", i, func_arg_is_handle[i]?"":"not ");
    //}
}


#endif