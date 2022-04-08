#include <linux/kthread.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/semaphore.h>
#include <linux/slab.h>
#include <linux/rhashtable.h>
#include <linux/kfifo.h>

#include "debug.h"
#include "command_handler.h"
#include "shadow_thread_pool.h"
#include "util.h"

#include <linux/time.h>

inline void coreprint_timestamp(const char *name) {
    struct timespec ts;
    getnstimeofday(&ts);
    //pr_info("Timestamp at %s: sec=%lu, usec=%lu\n", name, ts.tv_sec, ts.tv_nsec / 1000);
    pr_info("Timestamp at %s: %lu \n", name, ts.tv_sec*1000000 + ts.tv_nsec / 1000);
}

struct shadow_thread_t;

struct shadow_thread_object {
    intptr_t key; /* thread_id */
    struct rhash_head linkage;
    struct shadow_thread_t *thread;
};

const static struct rhashtable_params shadow_thread_object_params = {
    .key_len     = sizeof(const intptr_t),
    .key_offset  = offsetof(struct shadow_thread_object, key),
    .head_offset = offsetof(struct shadow_thread_object, linkage),
};

struct kava_shadow_thread_pool_t {
    /* Keys are ava IDs, values are shadow_thread_t* */
    struct rhashtable *threads;
    struct mutex lock;
};

#define KAVA_SHADOW_THREAD_COMMAND_QUEUE_SIZE 512

struct shadow_thread_t {
    intptr_t thread_id;
    DECLARE_KFIFO(queue, struct shadow_thread_command_t *,
                KAVA_SHADOW_THREAD_COMMAND_QUEUE_SIZE);
    struct semaphore queue_empty_sem;
    struct semaphore queue_full_sem;
    struct spinlock queue_spinlock;
    struct task_struct *thread;
    struct kava_shadow_thread_pool_t *pool;
};

struct shadow_thread_command_t {
    struct kava_chan* chan;
    struct kava_cmd_base* cmd;
};

struct kava_shadow_thread_pool_t *kava_shadow_thread_pool;

static struct shadow_thread_t *shadow_thread_new(struct kava_shadow_thread_pool_t *pool,
                                                intptr_t thread_id);

static struct shadow_thread_t* shadow_thread_self(struct kava_shadow_thread_pool_t *pool)
{
    struct shadow_thread_t *t;
    // A simple trick to keep a single shadow thread in the worker.
    // intptr_t key = current->pid;
    intptr_t key = 123456;
    struct shadow_thread_object *object =
        rhashtable_lookup_fast(pool->threads, &key, shadow_thread_object_params);
    if (object == NULL) {
        t = kmalloc(sizeof(struct shadow_thread_t), GFP_KERNEL);
        t->thread_id = key;
        INIT_KFIFO(t->queue);
        sema_init(&t->queue_full_sem, KAVA_SHADOW_THREAD_COMMAND_QUEUE_SIZE);
        sema_init(&t->queue_empty_sem, 0);
        spin_lock_init(&t->queue_spinlock);
        t->pool = pool;
        t->thread = current;
        BUG_ON(t->thread == NULL);
        BUG_ON(t->thread == (void *)key); // TODO: This may spuriously fail.

        /* Insert into hash table */
        object = vmalloc(sizeof(struct shadow_thread_object));
        object->key = key;
        object->thread = t;
        rhashtable_insert_fast(pool->threads, &object->linkage, shadow_thread_object_params);
        DEBUG_PRINT("Register shadow thread ID = %lld\n", key);
    }
    return object->thread;
}

/**
 * shadow_thread_pool_lookup_thread - Lookup thread in the shadow thread pool
 * by key
 * @pool: shadow thread pool
 * @key: lookup key
 */
static struct shadow_thread_t *shadow_thread_pool_lookup_thread(
        struct kava_shadow_thread_pool_t *pool, intptr_t key)
{
    struct shadow_thread_object *object;
    object = rhashtable_lookup_fast(pool->threads, &key, shadow_thread_object_params);
    return (object ? object->thread : NULL);
}

/**
 * shadow_thread_handle_single_cmd - Execute a single command that is destined
 * for this thread
 * @pool: the shadow thread pool this thread should be executing in
 * @t: the shadow thread that is executing the command
 *
 * This function returns 1 if this thread has been asked to exit, 0 otherwise.
 */
int shadow_thread_handle_single_cmd(struct kava_shadow_thread_pool_t *pool,
                                    struct shadow_thread_t *t)
{
    struct shadow_thread_command_t *scmd;
    struct kava_chan *chan;
    struct kava_cmd_base *cmd;

    BUG_ON(t == NULL);

    coreprint_timestamp("down");
    down(&t->queue_empty_sem);
    kfifo_out_spinlocked(&t->queue, &scmd, 1, &t->queue_spinlock);
    up(&t->queue_full_sem);
    coreprint_timestamp("up");

    chan = scmd->chan;
    cmd = scmd->cmd;
    kfree(scmd);

    if (cmd->mode == KAVA_CMD_MODE_INTERNAL &&
            cmd->command_id == KAVA_CMD_ID_HANDLER_THREAD_EXIT) {
        chan->cmd_free(chan, cmd);
        return 1;
    }

    BUG_ON(cmd->thread_id != t->thread_id);

    // TODO: checks MSG_SHUTDOWN messages/channel close from the other side.

    coreprint_timestamp("> kava_handle_cmd_and_notify");
    kava_handle_cmd_and_notify(chan, cmd);
    coreprint_timestamp("< kava_handle_cmd_and_notify>");
    return 0;
}

/**
 * shadow_thread_handle_single_cmd - Execute a single command that is destined
 * for this thread
 * @pool: the shadow thread pool this thread should be executing in
 * @thread_id: the shadow thread id
 *
 * This function returns 1 if this thread has been asked to exit, 0 otherwise.
 */
int shadow_thread_handle_single_cmd_by_id(struct kava_shadow_thread_pool_t *pool,
                                          uintptr_t thread_id)
{
    return shadow_thread_handle_single_cmd(pool,
                shadow_thread_pool_lookup_thread(pool, thread_id));
}

static int shadow_thread_loop(void *arg) {
    struct shadow_thread_t *t = arg;
    int exit_thread_flag;
    do {
        if (kthread_should_stop()) {
            do_exit(0);
        }
        exit_thread_flag = shadow_thread_handle_single_cmd(t->pool, t);
    } while (!exit_thread_flag);
    return 0;
}

/**
 * shadow_thread_new - Create a new shadow thread
 * @pool: the shadow thread pool which contains the new shadow thread
 * @thread_id: the command's thread ID in klib
 *
 * The shadow thread pool maintains a hash mapping between klib thread ID
 * and the shadow thread.
 */
static struct shadow_thread_t *shadow_thread_new(struct kava_shadow_thread_pool_t *pool,
                                                intptr_t thread_id) {
    int r;
    struct shadow_thread_t* t;
    struct shadow_thread_object* object;

    if ((t = shadow_thread_pool_lookup_thread(pool, (intptr_t)thread_id)) != NULL) {
        pr_err("Shadow thread ID already exists");
        return t;
    }

    t = kmalloc(sizeof(struct shadow_thread_t), GFP_KERNEL);
    t->thread_id = thread_id;
    INIT_KFIFO(t->queue);
    sema_init(&t->queue_full_sem, KAVA_SHADOW_THREAD_COMMAND_QUEUE_SIZE);
    sema_init(&t->queue_empty_sem, 0);
    spin_lock_init(&t->queue_spinlock);
    t->pool = pool;
    t->thread = kthread_create(shadow_thread_loop, t, "kava_shadow_thread_loop");
    BUG_ON(t->thread == NULL);
    BUG_ON(t->thread == (void *)thread_id); // TODO: This may spuriously fail.

    /* Insert into hash table */
    object = vmalloc(sizeof(struct shadow_thread_object));
    object->key = thread_id;
    object->thread = t;
    rhashtable_insert_fast(pool->threads, &object->linkage, shadow_thread_object_params);
    DEBUG_PRINT("Create shadow thread ID = %lld\n", thread_id);

    r = wake_up_process(t->thread);
    BUG_ON(r == 0);
    return t;
}

/**
 * shadow_thread_free_from_thread - Drop shadow or solid thread from thread pool
 * @t: shadow thread to be freed
 */
static void shadow_thread_free_from_thread(struct shadow_thread_t *t)
{
    mutex_lock(&t->pool->lock);

    /* Signal the thread to exit */
    kthread_stop(t->thread);

    /* If our ID is the same as the local thread reference then we must be a
     * solid (instead of shadow) thread. If we are solid, send a command to
     * exit the shadow.
     */
    if ((void *)t->thread_id == t->thread) {
        struct kava_cmd_base *cmd = kava_global_cmd_chan->cmd_new(
                kava_global_cmd_chan, sizeof(struct kava_cmd_base), 0);
        cmd->mode = KAVA_CMD_MODE_INTERNAL;
        cmd->command_id = KAVA_CMD_ID_HANDLER_THREAD_EXIT;
        cmd->thread_id = t->thread_id;
        kava_global_cmd_chan->cmd_send(kava_global_cmd_chan, cmd);
    }

    /* The thread is dropped by rhashtable_free_and_destroy automatically. */
    mutex_unlock(&t->pool->lock);

    kfifo_free(&t->queue);
    kfree(t);
}

/**
 * kava_shadow_thread_id - Get the cross-channel id of the current thread in this pool
 * @pool: the shadow thread pool which the thread belongs to
 */
EXPORTED_WEAKLY intptr_t kava_shadow_thread_id(struct kava_shadow_thread_pool_t *pool) {
    struct shadow_thread_t *t = shadow_thread_self(pool);
    return t->thread_id;
}

/**
 * kava_shadow_thread_pool_dispatch - Dispatch a single command to a thread
 * @pool: the shadow thread pool
 * @chan: the channel from which the command came
 * @cmd: the command to dispatch
 */
EXPORTED_WEAKLY void kava_shadow_thread_pool_dispatch(struct kava_shadow_thread_pool_t *pool,
                                                    struct kava_chan *chan,
                                                    struct kava_cmd_base *cmd) {
    struct shadow_thread_command_t *scmd;
    struct shadow_thread_t *t;

    mutex_lock(&pool->lock);
    t = shadow_thread_pool_lookup_thread(pool, (intptr_t)cmd->thread_id);
    if (t == NULL) {
        if (cmd->mode == KAVA_CMD_MODE_INTERNAL &&
                cmd->command_id == KAVA_CMD_ID_HANDLER_THREAD_EXIT) {
            /* If a thread for which we have no shadow is existing, just drop
             * the message. */
            mutex_unlock(&pool->lock);
            return;
        }
        t = shadow_thread_new(pool, cmd->thread_id);
    }
    scmd = kmalloc(sizeof(struct shadow_thread_command_t), GFP_KERNEL);
    scmd->chan = chan;
    scmd->cmd = cmd;
    down(&t->queue_full_sem);
    kfifo_in_spinlocked(&t->queue, &scmd, 1, &t->queue_spinlock);
    up(&t->queue_empty_sem);
    mutex_unlock(&pool->lock);
}

/**
 * kava_shadow_thread_pool_new - Create a new empty shadow thread pool
 */
EXPORTED_WEAKLY struct kava_shadow_thread_pool_t *kava_shadow_thread_pool_new(void)
{
    struct kava_shadow_thread_pool_t *pool =
            kmalloc(sizeof(struct kava_shadow_thread_pool_t), GFP_KERNEL);
    int r;
    pool->threads = kmalloc(sizeof(struct rhashtable), GFP_KERNEL);
    r = rhashtable_init(pool->threads, &shadow_thread_object_params);
    if (r) {
        kfree(pool->threads);
        kfree(pool);
        return NULL;
    }
    mutex_init(&pool->lock);
    return pool;
}

static void shadow_thread_pool_free_fn(void *ptr, void *arg)
{
    struct shadow_thread_t *thread = ptr;
    shadow_thread_free_from_thread(thread);
}

/**
 * kava_shadow_thread_pool_free - Destroy a shadow thread pool signaling all threads
 * to exit
 * @pool: the pool to be destroyed
 */
EXPORTED_WEAKLY void kava_shadow_thread_pool_free(struct kava_shadow_thread_pool_t *pool)
{
    rhashtable_free_and_destroy(pool->threads, shadow_thread_pool_free_fn, NULL);
    kfree(pool);
}
