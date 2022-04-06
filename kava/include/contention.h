#ifndef __KAVA_CONTENTION_H__
#define __KAVA_CONTENTION_H__

#ifndef __KERNEL__
    #error "Contention management API is supposed to be used for klib!"
#endif

#include <linux/types.h>


typedef void* (*kava_offload_func_t)(void *);

/**
 * KAvA policy function arguments:
 *   device offloading function,
 *   device offloading function arguments,
 *   CPU "offloading" function,
 *   CPU offloading function arguments.
 */
typedef void* (*kava_policy_t)(kava_offload_func_t, void *, kava_offload_func_t, void *);

/**
 * kava_register_contention_policy - Register a contention policy
 * @policy: policy to be registered
 * @device_func: device offloading function
 * @cpu_func: CPU execution function
 *
 * This function returns the allocated policy ID, or -EINVAL if @policy is NULL.
 */
int kava_register_contention_policy(kava_policy_t policy,
                                    kava_offload_func_t device_func,
                                    kava_offload_func_t cpu_func);

/**
 *  kava_run_contention_func - Enforce a contention policy which selects device
 *  or CPU automatically
 *  @policy_id: policy ID to be run
 *  @device_func_args: device offloading function
 *  @cpu_func_args: CPU execution function
 *
 *  This function can return any opaque value to the caller.
 */
void *kava_run_contention_func(int policy_id,
                               void *device_func_args,
                               void *cpu_func_args);

/**
 * kava_deregister_contention_policy - Deregister a policy
 * @policy_id: policy ID to be deregistered
 */
int kava_deregister_contention_policy(int policy_id);

#endif
