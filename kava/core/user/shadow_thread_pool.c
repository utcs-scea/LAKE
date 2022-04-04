#include <stdio.h>
#include <assert.h>

#include "debug.h"
#include "command_handler.h"
#include "shadow_thread_pool.h"
#include "util.h"

struct kava_shadow_thread_pool_t {
    GHashTable *threads; /* Keys are ava IDs, values are shadow_thread_t* */
    pthread_mutex_t lock;
    pthread_key_t key;
};

struct shadow_thread_t {
    intptr_t thread_id;
    GAsyncQueue *queue;
    pthread_t thread;
    struct kava_shadow_thread_pool_t *pool;
};

struct shadow_thread_command_t {
    struct kava_chan* chan;
    struct kava_cmd_base* cmd;
};

struct kava_shadow_thread_pool_t *kava_shadow_thread_pool;

/**
 * shadow_thread_self - Get current shadow thread
 * @pool: shadow thread pool which the current thread belongs to
 * */
static struct shadow_thread_t *shadow_thread_self(struct kava_shadow_thread_pool_t *pool) {
    struct shadow_thread_t* t = pthread_getspecific(pool->key);
    if (t == NULL) {
        t = malloc(sizeof(struct shadow_thread_t));
        intptr_t thread_id = (intptr_t)pthread_self(); // TODO: This may not work correctly on non-Linux
        assert(g_hash_table_lookup(pool->threads, (gpointer)thread_id) == NULL);
        t->thread_id = thread_id;
        t->queue = g_async_queue_new_full(NULL);
        t->pool = pool;
        t->thread = pthread_self();
        gboolean r = g_hash_table_insert(pool->threads, (gpointer)thread_id, t);
        assert(r);
        pthread_setspecific(pool->key, t);
    }
    return t;
}

/**
 * shadow_thread_handle_single_cmd - Execute a single command that is destined
 * for this thread
 * @pool: the shadow thread pool this thread should be executing in
 *
 * This function returns 1 if this thread has been asked to exit, 0 otherwise.
 */
static int shadow_thread_handle_single_cmd(struct kava_shadow_thread_pool_t *pool) {
    struct shadow_thread_t *t = shadow_thread_self(pool);
    struct shadow_thread_command_t *scmd = g_async_queue_pop(t->queue);

    struct kava_chan *chan = scmd->chan;
    struct kava_cmd_base *cmd = scmd->cmd;
    free(scmd);

    if (cmd->mode == KAVA_CMD_MODE_INTERNAL &&
            cmd->command_id == KAVA_CMD_ID_HANDLER_THREAD_EXIT) {
        chan->cmd_free(chan, cmd);
        fprintf(stderr, "Shadow thread %ld is notified to exit\n", cmd->thread_id);
        return 1;
    }

    assert(cmd->thread_id == t->thread_id);

    // TODO: checks MSG_SHUTDOWN messages/channel close from the other side.
    kava_handle_cmd_and_notify(chan, cmd);
    return 0;
}

static void *shadow_thread_loop(void *arg) {
    struct shadow_thread_t *t = arg;
    pthread_setspecific(t->pool->key, t);
    int exit_thread_flag;
    do {
        exit_thread_flag = shadow_thread_handle_single_cmd(t->pool);
    } while(!exit_thread_flag);
    return NULL;
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
    char thread_name[32];
    sprintf(thread_name, "shadow_%ld", thread_id);
    fprintf(stderr, "Create shadow thread %ld\n", thread_id);

    assert(g_hash_table_lookup(pool->threads, (gpointer)thread_id) == NULL);
    struct shadow_thread_t* t = malloc(sizeof(struct shadow_thread_t));
    t->thread_id = thread_id;
    t->queue = g_async_queue_new_full(NULL);
    t->pool = pool;
    int r = pthread_create(&t->thread, NULL, shadow_thread_loop, t);
    assert(r == 0);
    pthread_setname_np(t->thread, thread_name);
    pthread_detach(t->thread);
    assert(t->thread != thread_id); // TODO: This may spuriously fail.
    r = g_hash_table_insert(pool->threads, (gpointer)thread_id, t);
    assert(r);
    return t;
}

/**
 * shadow_thread_free_from_thread - Drop shadow or solid thread from thread pool
 * @t: shadow thread to be freed
 */
static void shadow_thread_free_from_thread(struct shadow_thread_t *t) {
    pthread_mutex_lock(&t->pool->lock);

    /* If our ID is the same as the local thread reference then we must be a
     * solid (instead of shadow) thread. If we are solid, send a command to
     * exit the shadow.
     */
    if (t->thread_id == t->thread) {
        struct kava_cmd_base *cmd = kava_global_cmd_chan->cmd_new(
                kava_global_cmd_chan, sizeof(struct kava_cmd_base), 0);
        cmd->mode = KAVA_CMD_MODE_INTERNAL;
        cmd->command_id = KAVA_CMD_ID_HANDLER_THREAD_EXIT;
        cmd->thread_id = t->thread_id;
        kava_global_cmd_chan->cmd_send(kava_global_cmd_chan, cmd);
    }

    /* Drop this thread from the pool */
    g_hash_table_remove(t->pool->threads, (gpointer)t->thread_id);
    pthread_mutex_unlock(&t->pool->lock);

    g_async_queue_unref(t->queue);
    t->queue = NULL;
    free(t);
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
    pthread_mutex_lock(&pool->lock);
    struct shadow_thread_t* t = g_hash_table_lookup(pool->threads,
                                                    (gpointer)cmd->thread_id);
    if (t == NULL) {
        if (cmd->mode == KAVA_CMD_MODE_INTERNAL &&
                cmd->command_id == KAVA_CMD_ID_HANDLER_THREAD_EXIT) {
            /* If a thread for which we have no shadow is existing, just drop
             * the message. */
            pthread_mutex_unlock(&pool->lock);
            return;
        }
        t = shadow_thread_new(pool, cmd->thread_id);
    }
    struct shadow_thread_command_t* scmd = malloc(sizeof(struct shadow_thread_command_t));
    scmd->chan = chan;
    scmd->cmd = cmd;
    g_async_queue_push(t->queue, scmd);
    pthread_mutex_unlock(&pool->lock);
}

/**
 * kava_shadow_thread_pool_new - Create a new empty shadow thread pool
 */
EXPORTED_WEAKLY struct kava_shadow_thread_pool_t *kava_shadow_thread_pool_new() {
    struct kava_shadow_thread_pool_t *pool =
            malloc(sizeof(struct kava_shadow_thread_pool_t));
    pool->threads = g_hash_table_new_full(
            kava_hash_pointer,
            g_direct_equal,
            NULL, NULL);
    pthread_key_create(&pool->key, (void (*)(void *))shadow_thread_free_from_thread);
    pthread_mutex_init(&pool->lock, NULL);
    return pool;
}

/**
 * kava_shadow_thread_pool_free - Destroy a shadow thread pool signaling all threads
 * to exit
 * @pool: the pool to be destroyed
 */
EXPORTED_WEAKLY void kava_shadow_thread_pool_free(struct kava_shadow_thread_pool_t *pool) {
    pthread_mutex_lock(&pool->lock);
    g_hash_table_destroy(pool->threads);
    free(pool);
}
