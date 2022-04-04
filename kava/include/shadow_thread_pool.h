//
// Created by amp on 4/2/19.
// Modified by hyu for KAvA on 10/30/19.
//

#ifndef __KAVA_SHADOWN_THREAD_POOL_H__
#define __KAVA_SHADOWN_THREAD_POOL_H__

#ifndef __KERNEL__

#include <stdint.h>

#else

#include <linux/types.h>
typedef __s64 intptr_t;

#endif

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations of structs to avoid dependency cycles in the includes.
struct kava_chan;
struct kava_cmd_base;

/**
 * A shadow thread pool manages a set of threads based on incoming command's `thread_id`s.
 * The pool will also handle "solid" threads: threads where are not managed by the pool,
 * and have a remote shadow at the other end of the AvA transport. A thread become a solid
 * thread as soon as it calls `kava_shadow_thread_id(pool)`.
 */
struct kava_shadow_thread_pool_t;

/**
 * kava_shadow_thread_pool_new - Create a new empty shadow thread pool
 */
struct kava_shadow_thread_pool_t *kava_shadow_thread_pool_new(void);

/**
 * kava_shadow_thread_pool_free - Destroy a shadow thread pool signaling all threads
 * to exit
 * @pool: the pool to be destroyed
 *
 * Any solid threads will not exit but will disassociate from this pool.
 */
void kava_shadow_thread_pool_free(struct kava_shadow_thread_pool_t *pool);

/**
 * kava_shadow_thread_id - Get the cross-channel id of the current thread in this pool
 * @pool: the shadow thread pool which the thread belongs to
 *
 * This function returns the ID of the current thread. If the current thread is
 * a shadow, then this is the ID of the remote solid thread.
 */
intptr_t kava_shadow_thread_id(struct kava_shadow_thread_pool_t *pool);

/**
 * shadow_thread_handle_single_cmd - Execute a single command that is destined
 * for this thread
 * @pool: the shadow thread pool this thread should be executing in
 * @thread_id: the shadow thread id
 *
 * This function returns 1 if this thread has been asked to exit, 0 otherwise.
 */
int shadow_thread_handle_single_cmd_by_id(struct kava_shadow_thread_pool_t *pool,
                                          uintptr_t thread_id);

/**
 * kava_shadow_thread_pool_dispatch - Dispatch a single command to a thread
 * @pool: the shadow thread pool
 * @chan: the channel from which the command came
 * @cmd: the command to dispatch
 *
 * This call must be non-blocking.
 */
void kava_shadow_thread_pool_dispatch(struct kava_shadow_thread_pool_t *pool,
                                    struct kava_chan *chan,
                                    struct kava_cmd_base *cmd);

extern struct kava_shadow_thread_pool_t *kava_shadow_thread_pool;

/**
 * Block until a command for this thread is executed and the predicate is true.
 */
#ifndef __KERNEL__
#define shadow_thread_handle_command_until(pool, thread_id, predicate) \
    while (!(predicate)) { \
        int r = shadow_thread_handle_single_cmd_by_id(pool, thread_id); \
        assert(r == 0 && ("Thread exit requested while waiting for " #predicate)); \
    }
#else
#define shadow_thread_handle_command_until(pool, thread_id, predicate) \
    while (!(predicate)) { \
        int r = shadow_thread_handle_single_cmd_by_id(pool, thread_id); \
        BUG_ON(r != 0 && ("Thread exit requested while waiting for " #predicate)); \
    }
#endif

#ifdef __cplusplus
}
#endif

#endif // __KAVA_SHADOWN_THREAD_POOL_H__
