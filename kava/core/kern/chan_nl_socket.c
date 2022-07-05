/*******************************************************************************

  File-poll channel implementation for kernel library.

  The channel exposes a character device to user space, and lets user-space
  worker poll the device and read the command with data. The worker writes to
  the device to send back the command result to kernel space.

*******************************************************************************/

#include <linux/module.h>
#include <linux/ctype.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/kmod.h>
#include <linux/kobject.h>
#include <linux/mm.h>
#include <linux/netlink.h>
#include <linux/poll.h>
#include <linux/semaphore.h>
#include <linux/skbuff.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <net/sock.h>
#include <asm/uaccess.h>

//#include <linux/workqueue.h>

#include "api.h"
#include "channel.h"
#include "channel_kern.h"
#include "command.h"
#include "command_handler.h"
#include "debug.h"

#define NETLINK_KAVA_USER 31
static struct sock *nl_sk = NULL;

struct block_seeker {
    uintptr_t cur_offset;
    uint32_t local_port;
    struct sk_buff *skb_start;
};

#define CMD_RING_SIZE 1024
struct cmd_ring {
    struct kava_cmd_base *cmd[CMD_RING_SIZE];
    int head;
    int tail;

    /* Command which was partially read by worker */
    struct kava_cmd_base *left_cmd;

    spinlock_t idx_lock;
    struct semaphore count_sem;
    struct semaphore slot_sem;
};

static struct cmd_ring *recv_cmdr;

static volatile int __is_worker_connected = 0;
static DECLARE_WAIT_QUEUE_HEAD(is_worker_connected);
static pid_t worker_pid;

/**
 * nl_socket_cmd_new - Allocate a new command struct
 * @chan: command channel
 * @cmd_struct_size: the size of the new command struct
 * @data_region_size: the size of the (potientially imaginary) data region; it should
 *    be computed by adding up the result of calls to `chan_buffer_size`
 *    on the same channel.
 *
 * This function returns the pointer to the new command struct.
 * This function allocates a skb which holds the netlink message. The netlink message
 * holds the command struct.
 */
static struct kava_cmd_base *nl_socket_cmd_new(struct kava_chan *chan,
                                size_t cmd_struct_size, size_t data_region_size)
{
    struct sk_buff *skb_out = nlmsg_new(cmd_struct_size + data_region_size, 0);
    struct nlmsghdr *nlh;
    struct kava_cmd_base *cmd;
    struct block_seeker *seeker;

    if (!skb_out) {
        pr_err("Failed to allocate new netlink skb\n");
        return NULL;
    }

    nlh = nlmsg_put(skb_out, 0, 0, NLMSG_DONE, cmd_struct_size + data_region_size, 0);
    cmd = (struct kava_cmd_base *)nlmsg_data(nlh);
    NETLINK_CB(skb_out).dst_group = 0;

    BUG_ON(sizeof(struct block_seeker) > sizeof(cmd->reserved_area));

    memset(cmd, 0, cmd_struct_size);
    cmd->mode = KAVA_CMD_MODE_API;
    cmd->command_size = cmd_struct_size;
    cmd->data_region = (void *)cmd_struct_size;
    cmd->region_size = data_region_size;

    seeker = (struct block_seeker *)cmd->reserved_area;
    seeker->cur_offset = cmd_struct_size;
    seeker->skb_start = skb_out;

    return cmd;
}

long nlsend_cpu(void* arg) {
    struct kava_cmd_base *cmd = (struct kava_cmd_base *) arg;
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    struct sk_buff *skb_out;
    size_t cmd_size;
    struct nlmsghdr *nlh;
    int ret;
    
    // int cpu;
    // cpu = get_cpu();
    // struct timespec ts;
    // getnstimeofday(&ts);
    // pr_info("Upcall called (cpu %d): sec=%lu, usec=%lu\n", cpu, ts.tv_sec, ts.tv_nsec / 1000);

    if (seeker->skb_start) {
        skb_out = seeker->skb_start;
        seeker->skb_start = NULL; /* Do not send this secret to user-space. */
        nlmsg_end(skb_out, (struct nlmsghdr *)skb_out->data);
        //ret = nlmsg_unicast(nl_sk, skb_out, worker_pid);
        ret = netlink_unicast(nl_sk, skb_out, worker_pid, 0);
        if (ret < 0) {
            pr_err("Failed to send netlink skb to API server, error=%d\n", ret);
            __is_worker_connected = 0;
            up(&recv_cmdr->count_sem);
        }
    }
    else {
        cmd_size = cmd->command_size + cmd->region_size;
        skb_out = nlmsg_new(cmd_size, 0);
        nlh = nlmsg_put(skb_out, 0, 0, NLMSG_DONE, cmd_size, 0);
        NETLINK_CB(skb_out).dst_group = 0;
        memcpy(nlmsg_data(nlh), cmd, cmd_size);
        nlmsg_end(skb_out, nlh);
        //ret = nlmsg_unicast(nl_sk, skb_out, worker_pid);
        ret = netlink_unicast(nl_sk, skb_out, worker_pid, 0);
        if (ret < 0) {
            pr_err("Failed to send netlink skb to API server, error=%d\n", ret);
            __is_worker_connected = 0;
            up(&recv_cmdr->count_sem);
        }
        else
            kfree(cmd);
    }

    //put_cpu();
    return 0;
}

/**
 * nl_socket_cmd_send - Send the message and all its attached buffers
 * @chan: command channel
 * @cmd: command to be sent
 *
 * This call is asynchronous and does not block for the command to
 * complete execution. It is responsible to free the sent command.
 * The message can either be allocated in a skb, or as a standalone buffer.
 */
static void nl_socket_cmd_send(struct kava_chan *chan, struct kava_cmd_base *cmd)
{
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    struct sk_buff *skb_out;
    struct nlmsghdr *nlh;
    size_t cmd_size;
    int ret;

    if (!__is_worker_connected) return;

    //work_on_cpu(2, nlsend_cpu, (void*)cmd);

    if (seeker->skb_start) {
        skb_out = seeker->skb_start;
        seeker->skb_start = NULL; /* Do not send this secret to user-space. */
        nlmsg_end(skb_out, (struct nlmsghdr *)skb_out->data);
        //ret = nlmsg_unicast(nl_sk, skb_out, worker_pid);
        ret = netlink_unicast(nl_sk, skb_out, worker_pid, 0);
        if (ret < 0) {
            pr_err("Failed to send netlink skb to API server, error=%d\n", ret);
            __is_worker_connected = 0;
            up(&recv_cmdr->count_sem);
        }
    }
    else {
        cmd_size = cmd->command_size + cmd->region_size;
        skb_out = nlmsg_new(cmd_size, 0);
        nlh = nlmsg_put(skb_out, 0, 0, NLMSG_DONE, cmd_size, 0);
        NETLINK_CB(skb_out).dst_group = 0;
        memcpy(nlmsg_data(nlh), cmd, cmd_size);
        nlmsg_end(skb_out, nlh);
        //ret = nlmsg_unicast(nl_sk, skb_out, worker_pid);
        ret = netlink_unicast(nl_sk, skb_out, worker_pid, 0);
        if (ret < 0) {
            pr_err("Failed to send netlink skb to API server, error=%d\n", ret);
            __is_worker_connected = 0;
            up(&recv_cmdr->count_sem);
        }
        else
            kfree(cmd);
    }
}

static void netlink_recv_msg(struct sk_buff *skb)
{
    struct kava_cmd_base *cmd = (struct kava_cmd_base *)skb->data;
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    struct kava_cmd_base *cmd_new;

    //struct timespec ts;
    //getnstimeofday(&ts);
    //pr_info("krecv:  sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);

    if (cmd->mode == KAVA_CMD_MODE_INTERNAL && cmd->command_id == KAVA_CMD_ID_CHANNEL_INIT) {
        if (__is_worker_connected) {
            // TODO: clear old recv_cmd ring.
        }
        else {
            memset(recv_cmdr, 0, sizeof(struct cmd_ring));
            spin_lock_init(&recv_cmdr->idx_lock);
            sema_init(&recv_cmdr->count_sem, 0);
            sema_init(&recv_cmdr->slot_sem, CMD_RING_SIZE);

            __is_worker_connected = 1;
            worker_pid = seeker->local_port;
            wake_up_interruptible(&is_worker_connected);
            pr_info("Netlink channel initialized with port/PID %u\n", worker_pid);
        }
        return;
    }

    if (!__is_worker_connected) {
        pr_info("Netlink messages before CHANNEL_INIT are ignored\n");
        return;
    }

    cmd_new = vmalloc(cmd->command_size + cmd->region_size);
    memcpy(cmd_new, cmd, cmd->command_size + cmd->region_size);
    seeker = (struct block_seeker *)cmd_new->reserved_area;
    seeker->skb_start = NULL;

    down(&recv_cmdr->slot_sem);
    recv_cmdr->cmd[recv_cmdr->tail] = cmd_new;
    recv_cmdr->tail = (recv_cmdr->tail + 1) & (CMD_RING_SIZE - 1);
    up(&recv_cmdr->count_sem);
}

/**
 * nl_socket_cmd_receive - Receive a command from a channel
 * @chan: command channel
 *
 * This call blocks waiting for a command to be sent along this channel.
 * The returned pointer is the received command and should be interpreted
 * based on its `command_id` field.
 */
static struct kava_cmd_base *nl_socket_cmd_receive(struct kava_chan *chan)
{
    struct kava_cmd_base *cmd;
    if (!__is_worker_connected) {
        wait_event_interruptible(is_worker_connected, __is_worker_connected == 1);
    }

    //struct timespec ts;
    //getnstimeofday(&ts);
    //pr_info("k cb: sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);

    down(&recv_cmdr->count_sem);

    if (!__is_worker_connected) {
        pr_err("Worker has exit\n");
        do_exit(0);
    }

    spin_lock(&recv_cmdr->idx_lock);
    cmd = recv_cmdr->cmd[recv_cmdr->head];
    recv_cmdr->cmd[recv_cmdr->head] = NULL;
    recv_cmdr->head = (recv_cmdr->head + 1) & (CMD_RING_SIZE - 1);
    spin_unlock(&recv_cmdr->idx_lock);

    up(&recv_cmdr->slot_sem);

    return cmd;
}

/**
 * nl_socket_chan_free - Disconnect this command channel and free all resources
 * associated with it.
 * @chan: command channel
 */
static void nl_socket_chan_free(struct kava_chan *chan) {
    if (nl_sk)
        netlink_kernel_release(nl_sk);
    kfree(recv_cmdr);
    kfree(chan);
}

/**
 * nl_socket_cmd_free - Free a command returned by `cmd_receive`
 * @chan: command channel
 * @cmd: command to be freed
 *
 * The command might be allocated within a skb.
 */
static void nl_socket_cmd_free(struct kava_chan *chan, struct kava_cmd_base *cmd)
{
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;
    if (seeker->skb_start)
        kfree_skb(seeker->skb_start);
    else
        vfree(cmd);
}

/**
 * nl_socket_chan_free - Print a command for debugging
 * @chan: command channel
 * @cmd: command to be printed
 */
static void nl_socket_cmd_print(const struct kava_chan *chan, const struct kava_cmd_base *cmd)
{
    // DEBUG_PRINT("struct command_base {\n"
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

    BUG_ON(!buffer || size == 0);

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

static struct netlink_kernel_cfg netlink_cfg = {
    .input = netlink_recv_msg,
};

/**
 * kava_chan_nl_socket_new - Initialize a new command channel with a character device
 * as the interface
 *
 * This function returns a command channel which connects the kernel driver to only
 * one worker. Multiple kernel drivers will need multiple channels which shares one
 * kernel framework library. That means the command should contains the channel
 * information to avoid locks for accesses to the framework library.
 */
struct kava_chan *kava_chan_nl_socket_new()
{
    int i;

    struct kava_chan *chan = (struct kava_chan *)kmalloc(sizeof(struct kava_chan), GFP_KERNEL);
    memset(chan, 0, sizeof(sizeof(struct kava_chan)));

    chan->id = KAVA_CHAN_NL_SOCKET;
    chan->name = kava_chan_name[chan->id];
    sprintf(chan->dev_name, "%s_%s", global_kapi->name, chan->name);
    for (i = 0; i < strlen(chan->dev_name); i++)
        chan->dev_name[i] = tolower(chan->dev_name[i]);

    /* Create receive command ring */
    recv_cmdr = (struct cmd_ring *)kmalloc(sizeof(struct cmd_ring), GFP_KERNEL);
    if (!recv_cmdr) {
        pr_err("Error creating received command ring\n");
        goto error;
    }

    /* Create netlink socket */
    nl_sk = netlink_kernel_create(&init_net, NETLINK_KAVA_USER, &netlink_cfg);
    if (!nl_sk) {
        pr_err("Error creating netlink socket\n");
        goto error;
    }
    pr_info("Netlink socket created");

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

error:
    kfree(chan);
    return NULL;
}
EXPORT_SYMBOL(kava_chan_nl_socket_new);
