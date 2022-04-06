#include "debug.h"
#include "command_handler.h"
#include "shadow_thread_pool.h"

#include <stdatomic.h>
#include <linux/kthread.h>
#include <linux/time.h>

#include "command_handler.h"
#include "shadow_thread_pool.h"

EXPORTED_WEAKLY struct kava_chan* kava_global_cmd_chan;
EXPORTED_WEAKLY struct kava_handle_pool* kava_global_handle_pool;

struct command_handler_t {
    void (*handle)(struct kava_chan *__chan, const struct kava_cmd_base *__cmd);
    void (*print)(FILE *file, const struct kava_chan *__chan, const struct kava_cmd_base *__cmd);
};

/**
 * Internal command handler and library API command handler.
 */
static struct command_handler_t apis[2];
static struct task_struct *handler_thread;
static volatile int __init_cmd_handler_executed;
static DEFINE_MUTEX(__handler_lock);
static DEFINE_MUTEX(__file_lock);

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
    struct command_handler_t *api;

    BUG_ON(mode >= KAVA_CMD_MODE_MAX);
    DEBUG_PRINT("Registering API command handler for %s API: handler at 0x%lx\n",
                mode?"external":"internal", (uintptr_t)handle);
    api = &apis[mode];
    BUG_ON(api->handle != NULL && "Duplicated handler cannot be registered");
    api->handle = handle;
    api->print = print;
}

static void _handle_commands_loop(struct kava_chan *chan) {
    struct kava_cmd_base *cmd;

    while(1) {
        if (kthread_should_stop()) {
            do_exit(0);
        }
        cmd = chan->cmd_receive(chan);
        if (cmd == NULL) {
            pr_err("Received empty command pointer\n");
            continue;
        }
        // TODO: checks MSG_SHUTDOWN messages/channel close from the other side.
        kava_shadow_thread_pool_dispatch(kava_shadow_thread_pool, chan, cmd);
    }
}

static int dispatch_thread_impl(void *userdata) {
    struct kava_chan *chan = (struct kava_chan *)userdata;
    _handle_commands_loop(chan);
    return 0;
}
// TODO: This will not correctly handle running callbacks in the initially calling thread.

/**
 * kava_init_cmd_handler - Initialize and start the command handler thread
 * @channel_create: the helper function which returns the created channel
 */
EXPORTED_WEAKLY void kava_init_cmd_handler(struct kava_chan *(*channel_create)(void)) {
    mutex_lock(&__handler_lock);
    if (!__init_cmd_handler_executed) {
        kava_global_cmd_chan = channel_create();
        handler_thread = kthread_run(dispatch_thread_impl,
                                    (void *)kava_global_cmd_chan,
                                    "kavad");
        atomic_thread_fence(memory_order_release);
        __init_cmd_handler_executed = 1;
    }
    mutex_unlock(&__handler_lock);
}

/**
 * kava_destroy_cmd_handler - Terminate the handler and close the channel
 */
EXPORTED_WEAKLY void kava_destroy_cmd_handler() {
    mutex_lock(&__handler_lock);
    if (__init_cmd_handler_executed) {
        kthread_stop(handler_thread);
        kava_global_cmd_chan->chan_free(kava_global_cmd_chan);
        atomic_thread_fence(memory_order_release);
        __init_cmd_handler_executed = 0;
    }
    mutex_unlock(&__handler_lock);
}

static int handle_cmd(struct kava_chan *chan, struct kava_handle_pool *handle_pool,
                          struct kava_cmd_base *cmd) {
    const int cmd_mode = cmd->mode;
    BUG_ON(apis[cmd_mode].handle == NULL);
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
    handle_cmd(chan, kava_global_handle_pool, cmd);
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
    BUG_ON(cmd_mode >= KAVA_CMD_MODE_MAX);

    /* Lock the file to prevent commands from getting mixed in the print out */
    mutex_lock(&__file_lock);
    if (apis[cmd_mode].print)
        apis[cmd_mode].print(file, chan, cmd);
    mutex_unlock(&__file_lock);
}

/**
 * kava_wait_for_cmd_handler - Block until the command handler thread exits
 */
EXPORTED_WEAKLY void kava_wait_for_cmd_handler() {
    return;
}

static void internal_api_handler(struct kava_chan *chan,
                            const struct kava_cmd_base *cmd) {
    BUG_ON(cmd->mode != KAVA_CMD_MODE_INTERNAL);

    switch (cmd->command_id) {
        case KAVA_CMD_ID_HANDLER_EXIT:
            pr_info("Command handler exits\n");
            kthread_stop(handler_thread);
            break;

        default:
            pr_err("Unknown internal command: %lu", cmd->command_id);
    }
}

/**
 * kava_init_internal_cmd_handler - Initialize (register) the internal command handler
 */
EXPORTED_WEAKLY void kava_init_internal_cmd_handler() {
    kava_register_cmd_handler(KAVA_CMD_MODE_INTERNAL, internal_api_handler, NULL);
}
