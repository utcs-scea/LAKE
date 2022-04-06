/*******************************************************************************

  File-poll channel implementation for user-space worker.

  The channel exposes a character device to user space, and lets user-space
  worker poll the device and read the command with data. The worker writes to
  the device to send back the command result to kernel space.

*******************************************************************************/

#include <assert.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>

#include "api.h"
#include "channel.h"
#include "channel_user.h"
#include "command.h"
#include "command_handler.h"
#include "debug.h"

static int dev_fd;
static struct pollfd dev_pfd = { .events = POLLIN | POLLRDHUP };

struct block_seeker {
    uintptr_t cur_offset;
};

#define CMD_RING_SIZE 1024
struct cmd_ring {
    struct kava_cmd_base *cmd[CMD_RING_SIZE];
    int head;
    int tail;

    /* Command which was partially read by worker */
    struct kava_cmd_base *left_cmd;

    pthread_spinlock_t idx_lock;
    sem_t count_sem;
    sem_t slot_sem;
};

struct cmd_ring *send_cmdr;
struct cmd_ring *recv_cmdr;

/**
 * file_poll_cmd_new - Allocate a new command struct
 * @chan: command channel
 * @cmd_struct_size: the size of the new command struct
 * @data_region_size: the size of the (potientially imaginary) data region; it should
 *    be computed by adding up the result of calls to `chan_buffer_size`
 *    on the same channel.
 *
 * This function returns the pointer to the new command struct.
 */
static struct kava_cmd_base *file_poll_cmd_new(struct kava_chan *chan,
                                size_t cmd_struct_size, size_t data_region_size)
{
    struct kava_cmd_base *cmd = (struct kava_cmd_base *)
        malloc(cmd_struct_size + data_region_size);
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;

    assert(sizeof(struct block_seeker) <= sizeof(cmd->reserved_area));

    memset(cmd, 0, cmd_struct_size);
    cmd->mode = KAVA_CMD_MODE_API;
    cmd->command_size = cmd_struct_size;
    cmd->data_region = (void *)cmd_struct_size;
    cmd->region_size = data_region_size;
    seeker->cur_offset = cmd_struct_size;

    return cmd;
}

/**
 * file_poll_cmd_send - Send the message and all its attached buffers
 * @chan: command channel
 * @cmd: command to be sent
 *
 * This call is responsible to free the sent command.
 */
static void file_poll_cmd_send(struct kava_chan *chan, struct kava_cmd_base *cmd)
{
    size_t ret;
    ret = write(dev_fd, cmd, cmd->command_size + cmd->region_size);
    assert(ret == cmd->command_size + cmd->region_size);

    free(cmd);
}

/**
 * file_poll_cmd_receive - Receive a command from a channel
 * @chan: command channel
 *
 * This call blocks waiting for a command. The returned pointer is the received
 * command and should be interpreted based on its `command_id` field.
 */
static struct kava_cmd_base *file_poll_cmd_receive(struct kava_chan *chan)
{
    struct kava_cmd_base *cmd;
    struct kava_cmd_base cmd_base;
    ssize_t ret;

    ret = poll(&dev_pfd, 1, -1);
    if (ret < 0) {
        pr_err("failed to channel device\n");
        exit(-1);
    }

    if (dev_pfd.revents & POLLRDHUP) {
        pr_err("channel shutdown\n");
        close(dev_pfd.fd);
        exit(-1);
    }

    if (dev_pfd.revents & POLLIN) {
        ret = read(dev_pfd.fd, &cmd_base, sizeof(struct kava_cmd_base));
        cmd = (struct kava_cmd_base *)malloc(cmd_base.command_size + cmd_base.region_size);
        memcpy(cmd, &cmd_base, sizeof(struct kava_cmd_base));
        read(dev_pfd.fd, (void *)cmd + sizeof(struct kava_cmd_base),
              cmd_base.command_size + cmd_base.region_size - sizeof(struct kava_cmd_base));
        return cmd;
    }

    return NULL;
}

/**
 * file_poll_chan_free - Disconnect this command channel and free all resources
 * associated with it.
 * @chan: command channel
 */
static void file_poll_chan_free(struct kava_chan *chan) {
    close(dev_fd);
    free(chan);

    sem_destroy(&send_cmdr->slot_sem);
    sem_destroy(&send_cmdr->count_sem);
    free(send_cmdr);
    sem_destroy(&recv_cmdr->slot_sem);
    sem_destroy(&recv_cmdr->count_sem);
    free(recv_cmdr);
}

/**
 * file_poll_cmd_free - Free a command returned by `cmd_receive`
 * @chan: command channel
 * @cmd: command to be freed
 */
static void file_poll_cmd_free(struct kava_chan *chan, struct kava_cmd_base *cmd)
{
    free(cmd);
}

/**
 * file_poll_chan_free - Print a command for debugging
 * @chan: command channel
 * @cmd: command to be printed
 */
static void file_poll_cmd_print(const struct kava_chan *chan, const struct kava_cmd_base *cmd)
{
    // DEBUG_PRINT("struct kava_cmd_base {\n"
    //             "  command_type=%ld\n"
    //             "  mode=%d\n"
    //             "  command_id=%ld\n"
    //             "  command_size=%lx\n"
    //             "  region_size=%lx\n"
    //             "}\n",
    //             cmd->command_type,
    //             cmd->mode,
    //             cmd->command_id,
    //             cmd->command_size,
    //             cmd->region_size);
    // DEBUG_PRINT_COMMAND(chan, cmd);
}

/**
 * file_poll_chan_buffer_size - Compute the buffer size that will actually be used
 * for a buffer of `size`
 * @chan: command channel
 * @size: input buffer size
 *
 * The returned value is the space needed on the channel. It may be larger than `size`.
 * For example, for shared memory implementations this should round the size up to a
 * cache line, so as to maintain the alignment of buffers when they are concatenated
 * into the data region.
 */
static size_t file_poll_chan_buffer_size(const struct kava_chan *chan, size_t size)
{
    return size;
}

/**
 * file_poll_chan_attach_buffer - Attach a buffer to a command
 * @chan: command channel
 * @cmd: command to which the buffer will be attached
 * @buffer: buffer to be attached. It  must be valid until after the call to `cmd_send`.
 * @size: buffer size
 *
 * This function returns a location independent buffer ID. The combined attached
 * buffers must fit within the initially provided `data_region_size` (to `cmd_new`).
 */
static void *file_poll_chan_attach_buffer(struct kava_chan *chan,
                                        struct kava_cmd_base *cmd,
                                        const void *buffer,
                                        size_t size)
{
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    void *offset = (void *)seeker->cur_offset;
    void *dst = (void *)((uintptr_t)cmd + seeker->cur_offset);

    assert(buffer && size);

    seeker->cur_offset += size;
    memcpy(dst, buffer, size);

    return offset;
}

/**
 * file_poll_chan_get_buffer - Translate a buffer_id (as returned by `chan_attach_buffer`
 * in the sender) into a data pointer
 * @chan: command channel
 * @cmd: command to which the buffer is attached
 * @buffer_id: buffer id to be translated
 *
 * This function returns the buffer's address. The returned pointer will be valid
 * until `cmd_free` is called on `cmd`.
 */
static void *file_poll_chan_get_buffer(const struct kava_chan *chan,
                                    const struct kava_cmd_base *cmd, void* buffer_id)
{
    return (void *)((uintptr_t)cmd + buffer_id);
}

/**
 * file_poll_chan_get_data_region - Returns the pointer to data region
 * @chan: command channel
 * @cmd: command to which the data region belongs
 *
 * This function returns the data region's address. The returned pointer is
 * mainly used for data extraction.
 */
static void *file_poll_chan_get_data_region(const struct kava_chan *chan,
                                        const struct kava_cmd_base *cmd)
{
    return (void *)((uintptr_t)cmd + cmd->command_size);
}

/**
 * kava_chan_file_poll_new - Initialize a new command channel with a character device
 * as the interface
 *
 * This function returns a command channel which connects the kernel driver via a
 * character device.
 */
struct kava_chan *kava_chan_file_poll_new(const char *dev_name)
{
    struct kava_chan *chan = (struct kava_chan *)malloc(sizeof(struct kava_chan));
    memset(chan, 0, sizeof(sizeof(struct kava_chan)));

    chan->id = KAVA_CHAN_FILE_POLL;
    chan->name = kava_chan_name[chan->id];

    /* Open character device */
    char dev_path[128];
    sprintf(dev_path, "/dev/%s", dev_name);
    dev_fd = open(dev_path, O_RDWR);
    if (dev_fd <= 0) {
        pr_err("Failed to open channel device %s\n", dev_path);
        goto free_chan;
    }
    pr_info("Channel device %s is opened\n", dev_path);
    dev_pfd.fd = dev_fd;

    /* Create command rings */
    send_cmdr = (struct cmd_ring *)malloc(sizeof(struct cmd_ring));
    memset(send_cmdr, 0, sizeof(struct cmd_ring));
    pthread_spin_init(&send_cmdr->idx_lock, 0);
    sem_init(&send_cmdr->count_sem, 0, 0);
    sem_init(&send_cmdr->slot_sem, 0, CMD_RING_SIZE);

    recv_cmdr = (struct cmd_ring *)malloc(sizeof(struct cmd_ring));
    memset(recv_cmdr, 0, sizeof(struct cmd_ring));
    pthread_spin_init(&recv_cmdr->idx_lock, 0);
    sem_init(&recv_cmdr->count_sem, 0, 0);
    sem_init(&recv_cmdr->slot_sem, 0, CMD_RING_SIZE);

    /* Assign helper functions */
    chan->cmd_new = file_poll_cmd_new;
    chan->cmd_send = file_poll_cmd_send;
    chan->cmd_receive = file_poll_cmd_receive;
    chan->cmd_free = file_poll_cmd_free;
    chan->cmd_print = file_poll_cmd_print;

    chan->chan_buffer_size = file_poll_chan_buffer_size;
    chan->chan_attach_buffer = file_poll_chan_attach_buffer;

    chan->chan_get_buffer = file_poll_chan_get_buffer;
    chan->chan_get_data_region = file_poll_chan_get_data_region;
    chan->chan_free = file_poll_chan_free;

    return chan;

free_chan:
    free(chan);
    return NULL;
}
