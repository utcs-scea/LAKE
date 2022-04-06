#include "debug.h"
#include "command_handler.h"
#include "shadow_thread_pool.h"

#ifdef __cplusplus
#include <atomic>
using namespace std;
#else
#include <stdatomic.h>
#endif

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "command_handler.h"
#include "shadow_thread_pool.h"

EXPORTED_WEAKLY struct kava_chan* kava_global_cmd_chan;

struct command_handler_t {
    void (*handle)(struct kava_chan *__chan, const struct kava_cmd_base *__cmd);
    void (*print)(FILE *file, const struct kava_chan *__chan, const struct kava_cmd_base *__cmd);
};

/**
 * Internal command handler and library API command handler.
 */
static struct command_handler_t apis[2];
static pthread_t handler_thread;
static volatile int __init_cmd_handler_executed;
static pthread_mutex_t __handler_lock = PTHREAD_MUTEX_INITIALIZER;

/**
 * kava_register_cmd_handler - Register a function to handle commands
 * @mode: internal or library API command
 * @handle: the command processing handler
 * @print: the command printing handler
 */
EXPORTED_WEAKLY void kava_register_cmd_handler(
        kava_cmd_mode mode,
        void (*handle)(struct kava_chan *, const struct kava_cmd_base *),
        void (*print)(FILE *, const struct kava_chan *, const struct kava_cmd_base *))
{
    assert(mode < KAVA_CMD_MODE_MAX);
    DEBUG_PRINT("Registering API command handler for %s API: handler at 0x%lx\n",
                mode?"external":"internal", (uintptr_t)handle);
    struct command_handler_t *api = &apis[mode];
    assert(api->handle == NULL && "Duplicated handler cannot be registered");
    api->handle = handle;
    api->print = print;
}

static void _handle_commands_loop(struct kava_chan *chan) {
    pr_info("Dispatch thread is running\n");
    while(1) {
        struct kava_cmd_base *cmd = chan->cmd_receive(chan);
        // TODO: checks MSG_SHUTDOWN messages/channel close from the other side.
        kava_shadow_thread_pool_dispatch(kava_shadow_thread_pool, chan, cmd);
    }
}

static void *dispatch_thread_impl(void *userdata) {
    struct kava_chan *chan = (struct kava_chan *)userdata;

    /* Enable the cancellation state */
    if (pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL)) {
        perror("pthread_setcancelstate failed\n");
        exit(0);
    }

    /* Wait for the pthread_join */
    if (pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL)) {
        perror("pthread_setcanceltype failed\n");
        exit(0);
    }

    _handle_commands_loop(chan);
    return NULL;
}
// TODO: This will not correctly handle running callbacks in the initially calling thread.

/**
 * kava_init_cmd_handler - Initialize and start the command handler thread
 * @channel_create: the helper function which returns the created channel
 */
EXPORTED_WEAKLY void kava_init_cmd_handler(struct kava_chan *(*channel_create)()) {
    pthread_mutex_lock(&__handler_lock);
    if (!__init_cmd_handler_executed) {
        kava_global_cmd_chan = channel_create();
        pthread_create(&handler_thread, NULL,
                       dispatch_thread_impl, (void *)kava_global_cmd_chan);
        pthread_setname_np(handler_thread, "dispatch_thread");
        atomic_thread_fence(memory_order_release);
        __init_cmd_handler_executed = 1;
    }
    pthread_mutex_unlock(&__handler_lock);
}

/**
 * kava_init_cmd_handler_inline - Initialize and start the command handler
 * without creating a new thread
 * @channel_create: the helper function which returns the created channel
 */
EXPORTED_WEAKLY void kava_init_cmd_handler_inline(struct kava_chan *(*channel_create)()) {
    kava_global_cmd_chan = channel_create();
    _handle_commands_loop(kava_global_cmd_chan);
}

/**
 * kava_destroy_cmd_handler - Terminate the handler and close the channel
 */
EXPORTED_WEAKLY void kava_destroy_cmd_handler() {
    pthread_mutex_lock(&__handler_lock);
    if (__init_cmd_handler_executed) {
        pthread_cancel(handler_thread);
        pthread_join(handler_thread, NULL);
        kava_global_cmd_chan->chan_free(kava_global_cmd_chan);
        atomic_thread_fence(memory_order_release);
        __init_cmd_handler_executed = 0;
    }
    pthread_mutex_unlock(&__handler_lock);
}

static int handle_cmd(struct kava_chan *chan, struct kava_cmd_base *cmd) {
    const int cmd_mode = cmd->mode;
    assert(apis[cmd_mode].handle != NULL);
    apis[cmd_mode].handle(chan, cmd);
    chan->cmd_free(chan, cmd);
    return 0;
}

/**
 * kava_handle_cmd_and_notify - Handle the received command
 * @chan: the command channel where the command is received
 * @cmd: the received command
 */
EXPORTED_WEAKLY void kava_handle_cmd_and_notify(struct kava_chan *chan,
                                        struct kava_cmd_base *cmd)
{
    handle_cmd(chan, cmd);
}

/**
 * kava_print_cmd - Print command with the registered printing handler
 * @file: the stream file to which the command is printed
 * @chan: the command channel where the command is received
 * @cmd: the received command
 */
EXPORTED_WEAKLY void kava_print_cmd(FILE* file, const struct kava_chan *chan,
                            const struct kava_cmd_base *cmd) {
    const int cmd_mode = cmd->mode;
    assert(cmd_mode < KAVA_CMD_MODE_MAX);

    /* Lock the file to prevent commands from getting mixed in the print out */
    flockfile(file);
    if (apis[cmd_mode].print)
        apis[cmd_mode].print(file, chan, cmd);
    funlockfile(file);
}

/**
 * kava_wait_for_cmd_handler - Block until the command handler thread exits
 */
EXPORTED_WEAKLY void kava_wait_for_cmd_handler() {
    pthread_join(handler_thread, NULL);
}

static void internal_api_handler(struct kava_chan *chan,
                            const struct kava_cmd_base *cmd) {
    assert(cmd->mode == KAVA_CMD_MODE_INTERNAL);

    switch (cmd->command_id) {
        case KAVA_CMD_ID_HANDLER_EXIT:
            pr_info("Command handler exits\n");
            exit(0);

        default:
            pr_err("Unknown internal command: %lu", cmd->command_id);
            exit(0);
    }
}

/**
 * kava_init_internal_cmd_handler - Initialize (register) the internal command handler
 */
EXPORTED_WEAKLY void kava_init_internal_cmd_handler() {
    kava_register_cmd_handler(KAVA_CMD_MODE_INTERNAL, internal_api_handler, NULL);
}
