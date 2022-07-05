/*******************************************************************************

  Netlink socket channel implementation for user-space worker.

  The channel exposes a netlink socket to user space, and lets user-space
  worker poll the socket and read the command with data. The worker writes to
  the socket file descriptor to send back the command result to kernel space.

*******************************************************************************/

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <linux/netlink.h>
#include <netlink/netlink.h> // depends on libnl-3-dev
#include <time.h>

#include "api.h"
#include "channel.h"
#include "channel_user.h"
#include "command.h"
#include "command_handler.h"
#include "debug.h"

// TODO: group into static channel struct.
static struct nl_sock *nls = NULL;

struct block_seeker {
    uintptr_t cur_offset;
    uint32_t local_port;
    struct nlmsghdr *nlh_start;
};

#define NETLINK_KAVA_USER 31

/**
 * nl_socket_cmd_new - Allocate a new command struct
 * @chan: command channel
 * @cmd_struct_size: the size of the new command struct
 * @data_region_size: the size of the (potientially imaginary) data region; it should
 *    be computed by adding up the result of calls to `chan_buffer_size`
 *    on the same channel.
 *
 * This function returns the pointer to the new command struct.
 * The command struct is allocated inside the netlink message. The nlmsg header is at
 * `cmd - NLMSG_HDRLEN` so that the message space overlaps the command.
 */
static struct kava_cmd_base *nl_socket_cmd_new(struct kava_chan *chan,
                                size_t cmd_struct_size, size_t data_region_size)
{
    struct kava_cmd_base *cmd = (struct kava_cmd_base *)malloc(cmd_struct_size + data_region_size);
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;

    /* Initialize cmd */
    cmd->command_type = 0;
    cmd->mode = KAVA_CMD_MODE_API;
    cmd->command_size = cmd_struct_size;
    cmd->data_region = (void *)cmd_struct_size;
    cmd->region_size = data_region_size;

    seeker->cur_offset = cmd_struct_size;
    seeker->local_port = nl_socket_get_local_port(nls);
    seeker->nlh_start = NULL;

    return cmd;
}

/**
 * nl_socket_cmd_send - Send the message and all its attached buffers
 * @chan: command channel
 * @cmd: command to be sent
 *
 * This call is responsible to free the sent command.
 */
static void nl_socket_cmd_send(struct kava_chan *chan, struct kava_cmd_base *cmd)
{
    size_t ret;
    //struct timeval tv_recv;
    //gettimeofday(&tv_recv, NULL);
    //printf("sent: sec=%lu, usec=%lu\n", tv_recv.tv_sec, tv_recv.tv_usec);

    ret = nl_sendto(nls, cmd, cmd->command_size + cmd->region_size);
    assert(ret == cmd->command_size + cmd->region_size);
    free(cmd);
}

/**
 * nl_socket_cmd_receive - Receive a command from a channel
 * @chan: command channel
 *
 * This call blocks waiting for a command. The returned pointer is the received
 * command and should be interpreted based on its `command_id` field.
 */
static struct kava_cmd_base *nl_socket_cmd_receive(struct kava_chan *chan)
{
    struct nlmsghdr *nlh;
    struct kava_cmd_base *cmd;
    struct block_seeker *seeker;
    struct sockaddr_nl nla;
    ssize_t ret;
    //struct timeval tv_recv;

    while(1) {
        ret = nl_recv(nls, &nla, (unsigned char **)&nlh, NULL);
        if (ret > 0) break;
    }

    //nl_recv(nls, &nla, (unsigned char **)&nlh, NULL);

    //gettimeofday(&tv_recv, NULL);
    //printf("Upcall received: sec=%lu, usec=%lu\n", tv_recv.tv_sec, tv_recv.tv_usec);

    cmd = (struct kava_cmd_base *)NLMSG_DATA(nlh);
    assert(ret == NLMSG_SPACE(cmd->command_size + cmd->region_size));
    seeker = (struct block_seeker *)cmd->reserved_area;
    seeker->nlh_start = nlh;
    return cmd;
}

/**
 * file_poll_chan_free - Disconnect this command channel and free all resources
 * associated with it.
 * @chan: command channel
 */
static void nl_socket_chan_free(struct kava_chan *chan) {
    if (nls)
        nl_socket_free(nls);
    free(chan);
}

/**
 * nl_socket_cmd_free - Free a command returned by `cmd_receive`
 * @chan: command channel
 * @cmd: command to be freed
 */
static void nl_socket_cmd_free(struct kava_chan *chan, struct kava_cmd_base *cmd)
{
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    if (seeker->nlh_start)
        free(seeker->nlh_start);
    else
        free(cmd);
}

/**
 * nl_socket_chan_free - Print a command for debugging
 * @chan: command channel
 * @cmd: command to be printed
 */
static void nl_socket_cmd_print(const struct kava_chan *chan, const struct kava_cmd_base *cmd)
{
    DEBUG_PRINT("struct kava_cmd_base {\n"
                "  command_type=%ld\n"
                "  mode=%d\n"
                "  command_id=%ld\n"
                "  command_size=%lx\n"
                "  region_size=%lx\n"
                "}\n",
                cmd->command_type,
                cmd->mode,
                cmd->command_id,
                cmd->command_size,
                cmd->region_size);
    DEBUG_PRINT_COMMAND(chan, cmd);
}

/**
 * nl_socket_chan_buffer_size - Compute the buffer size that will actually be used
 * for a buffer of `size`
 * @chan: command channel
 * @size: input buffer size
 *
 * The returned value is the space needed on the channel. It may be larger than `size`.
 * For example, for shared memory implementations this should round the size up to a
 * cache line, so as to maintain the alignment of buffers when they are concatenated
 * into the data region.
 */
static size_t nl_socket_chan_buffer_size(const struct kava_chan *chan, size_t size)
{
    return size;
}

/**
 * nl_socket_chan_attach_buffer - Attach a buffer to a command
 * @chan: command channel
 * @cmd: command to which the buffer will be attached
 * @buffer: buffer to be attached. It  must be valid until after the call to `cmd_send`.
 * @size: buffer size
 *
 * This function returns a location independent buffer ID. The combined attached
 * buffers must fit within the initially provided `data_region_size` (to `cmd_new`).
 */
static void *nl_socket_chan_attach_buffer(struct kava_chan *chan,
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
 * nl_socket_chan_get_buffer - Translate a buffer_id (as returned by `chan_attach_buffer`
 * in the sender) into a data pointer
 * @chan: command channel
 * @cmd: command to which the buffer is attached
 * @buffer_id: buffer id to be translated
 *
 * This function returns the buffer's address. The returned pointer will be valid
 * until `cmd_free` is called on `cmd`.
 */
static void *nl_socket_chan_get_buffer(const struct kava_chan *chan,
                                    const struct kava_cmd_base *cmd, void* buffer_id)
{
    return (void *)((uintptr_t)cmd + buffer_id);
}

/**
 * nl_socket_chan_get_data_region - Returns the pointer to data region
 * @chan: command channel
 * @cmd: command to which the data region belongs
 *
 * This function returns the data region's address. The returned pointer is
 * mainly used for data extraction.
 */
static void *nl_socket_chan_get_data_region(const struct kava_chan *chan,
                                        const struct kava_cmd_base *cmd)
{
    return (void *)((uintptr_t)cmd + cmd->command_size);
}

/**
 * kava_chan_nl_socket_new - Initialize a new command channel with netlink socket
 *
 * This function returns a command channel which connects the kernel driver using
 * netlink socket.
 */
struct kava_chan *kava_chan_nl_socket_new(const char *dev_name)
{
    struct kava_chan *chan = (struct kava_chan *)malloc(sizeof(struct kava_chan));
    struct kava_cmd_base *cmd;
    size_t buf_len = 0;
    int ret;
    socklen_t buf_len_size = sizeof(size_t);
    memset(chan, 0, sizeof(sizeof(struct kava_chan)));

    chan->id = KAVA_CHAN_NL_SOCKET;
    chan->name = kava_chan_name[chan->id];

    nls = nl_socket_alloc();
    if (nls == NULL) {
        pr_err("Failed to create netlink socket\n");
        goto free_chan;
    }

    ret = nl_connect(nls, NETLINK_KAVA_USER);
    if (ret < 0) {
        nl_perror(ret, "nl_connect");
        goto free_chan;
    }

    /* Set the socket buffer sizes. */
    nl_socket_set_buffer_size(nls, 2097152, 2097152);
    nl_socket_set_msg_buf_size(nls, 2097152);

    /* Disable ack message. */
    nl_socket_disable_auto_ack(nls);
    nl_socket_set_passcred(nls, 0);
    nl_socket_recv_pktinfo(nls, 0);

    pr_info("Setup source address at port/PID %u\n", nl_socket_get_local_port(nls));

    getsockopt(nl_socket_get_fd(nls), SOL_SOCKET, SO_RCVBUF, &buf_len, &buf_len_size);
    pr_info("Default socket recv buffer size %ld\n", buf_len);
    getsockopt(nl_socket_get_fd(nls), SOL_SOCKET, SO_SNDBUF, &buf_len, &buf_len_size);
    pr_info("Default socket send buffer size %ld\n", buf_len);
    pr_info("Default socket message buffer size %ld\n", nl_socket_get_msg_buf_size(nls));

    //set non blocking to lower latency.
    nl_socket_set_nonblocking(nls);

    /* Notify klib of PID */
    cmd = nl_socket_cmd_new(chan, sizeof(struct kava_cmd_base), 0);
    cmd->mode = KAVA_CMD_MODE_INTERNAL;
    cmd->command_id = KAVA_CMD_ID_CHANNEL_INIT;
    nl_socket_cmd_send(chan, cmd);

    /* Assign helper functions */
    chan->cmd_new              = nl_socket_cmd_new;
    chan->cmd_send             = nl_socket_cmd_send;
    chan->cmd_receive          = nl_socket_cmd_receive;
    chan->cmd_free             = nl_socket_cmd_free;
    chan->cmd_print            = nl_socket_cmd_print;

    chan->chan_buffer_size     = nl_socket_chan_buffer_size;
    chan->chan_attach_buffer   = nl_socket_chan_attach_buffer;

    chan->chan_get_buffer      = nl_socket_chan_get_buffer;
    chan->chan_get_data_region = nl_socket_chan_get_data_region;
    chan->chan_free            = nl_socket_chan_free;

    return chan;

free_chan:
    if (nls)
        nl_socket_free(nls);
    free(chan);
    return NULL;
}
