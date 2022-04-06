#ifndef __KAVA_COMMAND_HANDLER_H__
#define __KAVA_COMMAND_HANDLER_H__

#ifdef __KERNEL__
#define FILE int
#else
#include <stdio.h>
#endif

#include "api.h"
#include "channel.h"
#include "command.h"

/**
 * kava_register_cmd_handler - Register a function to handle commands
 * @mode: internal or library API command
 * @handle: the command processing handler
 * @print: the command printing handler
 *
 * This call should abort the program if kava_init_cmd_handler has already been
 * called, or the handler of a mode is registered twice. The internal handler
 * is often registered explicitly in kava_init_internal_cmd_handler; while the
 * library API command handler is often registered in the worker's constructor.
 */
void kava_register_cmd_handler(
        kava_cmd_mode mode,
        void (*handle)(struct kava_chan *__chan, const struct kava_cmd_base *__cmd),
        void (*print)(FILE *file, const struct kava_chan *__chan, const struct kava_cmd_base *__cmd));

/**
 * kava_init_cmd_handler - Initialize and start the command handler thread
 * @channel_create: the helper function which returns the created channel
 *
 * The helper function can return a pre-created channel (fast), or create a
 * new channel by itself (slow).
 */
void kava_init_cmd_handler(struct kava_chan *(*chan_create)(void));

/**
 * kava_init_cmd_handler_inline - Initialize and start the command handler
 * without creating a new thread
 * @channel_create: the helper function which returns the created channel
 */
void kava_init_cmd_handler_inline(struct kava_chan *(*channel_create)(void));

/**
 * kava_destroy_cmd_handler - Terminate the handler and close the channel
 */
void kava_destroy_cmd_handler(void);

/**
 * kava_wait_for_cmd_handler - Block until the command handler thread exits
 */
void kava_wait_for_cmd_handler(void);

/**
 * kava_handle_cmd_and_notify - Handle the received command
 * @chan: the command channel where the command is received
 * @cmd: the received command
 */
void kava_handle_cmd_and_notify(struct kava_chan *chan, struct kava_cmd_base *cmd);

/**
 * kava_print_cmd - Print command with the registered printing handler
 * @file: file to which the command is printed
 * @chan: command channel where the command is received
 * @cmd: received command
 */
void kava_print_cmd(FILE* file, const struct kava_chan *chan,
            const struct kava_cmd_base *cmd);

/**
 * kava_init_internal_cmd_handler - Initialize (register) the internal command handler
 *
 * This function must be called explicitly by worker to register the internal
 * command handler, if there is any special internal command to handle.
 */
void kava_init_internal_cmd_handler(void);

#endif
