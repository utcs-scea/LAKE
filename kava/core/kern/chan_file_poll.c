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
#include <linux/poll.h>
#include <linux/semaphore.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <asm/uaccess.h>

#include "api.h"
#include "channel.h"
#include "channel_kern.h"
#include "command.h"
#include "command_handler.h"
#include "debug.h"

static struct class *dev_class;
static struct device *dev_node;

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

    spinlock_t idx_lock;
    struct semaphore count_sem;
    struct semaphore slot_sem;
};

static struct cmd_ring *send_cmdr;
static struct cmd_ring *recv_cmdr;

static DECLARE_WAIT_QUEUE_HEAD(chan_poll_wait);

static volatile int __is_worker_connected;
static DECLARE_WAIT_QUEUE_HEAD(is_worker_connected);

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
        vmalloc(cmd_struct_size + data_region_size);
    struct block_seeker *seeker = (struct block_seeker *)cmd->reserved_area;

    BUG_ON(sizeof(struct block_seeker) > sizeof(cmd->reserved_area));

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
 * This call is asynchronous and does not block for the command to
 * complete execution. It is responsible to free the sent command.
 */
static void file_poll_cmd_send(struct kava_chan *chan, struct kava_cmd_base *cmd)
{
    if (!__is_worker_connected) return;

    down(&send_cmdr->slot_sem);

    spin_lock(&send_cmdr->idx_lock);
    send_cmdr->cmd[send_cmdr->tail] = cmd;
    send_cmdr->tail = (send_cmdr->tail + 1) & (CMD_RING_SIZE - 1);
    spin_unlock(&send_cmdr->idx_lock);

    up(&send_cmdr->count_sem);
    wake_up_interruptible(&chan_poll_wait);
}

/**
 * file_poll_cmd_receive - Receive a command from a channel
 * @chan: command channel
 *
 * This call blocks waiting for a command to be sent along this channel.
 * The returned pointer is the received command and should be interpreted
 * based on its `command_id` field.
 */
static struct kava_cmd_base *file_poll_cmd_receive(struct kava_chan *chan)
{
    struct kava_cmd_base *cmd;
    if (!__is_worker_connected) {
        wait_event_interruptible(is_worker_connected, __is_worker_connected == 1);
    }

    down(&recv_cmdr->count_sem);

    spin_lock(&recv_cmdr->idx_lock);
    cmd = recv_cmdr->cmd[recv_cmdr->head];
    recv_cmdr->cmd[recv_cmdr->head] = NULL;
    recv_cmdr->head = (recv_cmdr->head + 1) & (CMD_RING_SIZE - 1);
    spin_unlock(&recv_cmdr->idx_lock);

    up(&recv_cmdr->slot_sem);

    return cmd;
}

/**
 * file_poll_chan_free - Disconnect this command channel and free all resources
 * associated with it.
 * @chan: command channel
 */
static void file_poll_chan_free(struct kava_chan *chan) {
    device_destroy(dev_class, MKDEV(KAVA_CHAN_DEV_MAJOR, chan->id));
    class_unregister(dev_class);
    class_destroy(dev_class);
    unregister_chrdev(KAVA_CHAN_DEV_MAJOR, chan->dev_name);

    kfree(chan);
}

/**
 * file_poll_cmd_free - Free a command returned by `cmd_receive`
 * @chan: command channel
 * @cmd: command to be freed
 */
static void file_poll_cmd_free(struct kava_chan *chan, struct kava_cmd_base *cmd)
{
    vfree(cmd);
}

/**
 * file_poll_chan_free - Print a command for debugging
 * @chan: command channel
 * @cmd: command to be printed
 */
static void file_poll_cmd_print(const struct kava_chan *chan, const struct kava_cmd_base *cmd)
{
    DEBUG_PRINT("struct command_base {\n"
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

    BUG_ON(!buffer || size == 0);

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

static int chan_dev_open(struct inode *inode, struct file *filp)
{
    // TODO: it will need to verify workers when there are multiple instances
    // of workers.
    pr_info("Channel device is opened\n");

    /* Create send command ring */
    send_cmdr = (struct cmd_ring *)kmalloc(sizeof(struct cmd_ring), GFP_KERNEL);
    if (send_cmdr == NULL)
        return -ENOMEM;
    memset(send_cmdr, 0, sizeof(struct cmd_ring));
    spin_lock_init(&send_cmdr->idx_lock);
    sema_init(&send_cmdr->count_sem, 0);
    sema_init(&send_cmdr->slot_sem, CMD_RING_SIZE);

    /* Create receive command ring */
    recv_cmdr = (struct cmd_ring *)kmalloc(sizeof(struct cmd_ring), GFP_KERNEL);
    if (recv_cmdr == NULL) {
        kfree(send_cmdr);
        return -ENOMEM;
    }
    memset(recv_cmdr, 0, sizeof(struct cmd_ring));
    spin_lock_init(&recv_cmdr->idx_lock);
    sema_init(&recv_cmdr->count_sem, 0);
    sema_init(&recv_cmdr->slot_sem, CMD_RING_SIZE);

    __is_worker_connected = 1;
    wake_up_interruptible(&is_worker_connected);

    return 0;
}

static ssize_t chan_dev_read(struct file *filp, char __user *buf, size_t size, loff_t *offp)
{
    struct kava_cmd_base *cmd = send_cmdr->left_cmd;
    int unlockable;

    if (size == 0)
        return 0;

    if (cmd) {
        if (size + *offp > cmd->command_size + cmd->region_size) {
            return -EBUSY;
        }
        copy_to_user(buf, ((void *)cmd) + *offp, size);

        *offp += size;
        if (*offp == cmd->command_size + cmd->region_size) {
            *offp = 0;
            send_cmdr->left_cmd = NULL;
            vfree(cmd);
        }

        return size;
    }

    unlockable = down_trylock(&send_cmdr->count_sem);
    if (unlockable) {
        return -ENODATA;
    }

    /* Spinlock is not needed because there is only one consumer */
    cmd = send_cmdr->cmd[send_cmdr->head];
    send_cmdr->cmd[send_cmdr->head] = NULL;
    send_cmdr->head = (send_cmdr->head + 1) & (CMD_RING_SIZE - 1);

    if (size > cmd->command_size + cmd->region_size) {
        send_cmdr->left_cmd = cmd;
        *offp = 0;
        return -EBUSY;
    }

    copy_to_user(buf, cmd, size);
    if (size < cmd->command_size + cmd->region_size) {
        send_cmdr->left_cmd = cmd;
        *offp = size;
    }
    else {
        send_cmdr->left_cmd = NULL;
        /* Free local copy of command struct */
        vfree(cmd);
    }

    up(&send_cmdr->slot_sem);

    return size;
}


static ssize_t chan_dev_write(struct file *filp, const char __user *buf, size_t size, loff_t *offp)
{
    struct kava_cmd_base *cmd;

    if (size < sizeof(struct kava_cmd_base)) {
        pr_err("command size is too small\n");
        return -EBUSY;
    }

    cmd = vmalloc(size);
    copy_from_user(cmd, buf, size);

    down(&recv_cmdr->slot_sem);

    /* Spinlock is not needed because there is only one producer */
    recv_cmdr->cmd[recv_cmdr->tail] = cmd;
    recv_cmdr->tail = (recv_cmdr->tail + 1) & (CMD_RING_SIZE - 1);

    up(&recv_cmdr->count_sem);

    return size;
}

static unsigned int chan_dev_poll(struct file *filp, poll_table *wait)
{
    poll_wait(filp, &chan_poll_wait, wait);
    if (send_cmdr->left_cmd) {
        return POLLIN | POLLRDNORM;
    }
    if (!down_trylock(&send_cmdr->count_sem)) {
        up(&send_cmdr->count_sem);
        return POLLIN | POLLRDNORM;
    }
    return 0;
}

static int chan_dev_release(struct inode *inode, struct file *filp)
{
    __is_worker_connected = 0;
    kfree(send_cmdr);
    kfree(recv_cmdr);
    return 0;
}

static const struct file_operations fops =
{
    .owner          = THIS_MODULE,
    .open           = chan_dev_open,
    .read           = chan_dev_read,
    .write          = chan_dev_write,
    .poll           = chan_dev_poll,
    .release        = chan_dev_release,
};

static char *mod_dev_node(struct device *dev, umode_t *mode)
{
    if (mode)
        *mode = 0666;
    return NULL;
}

/**
 * umh_spawn_worker - Spawn worker from kernel space
 * @dev_name: the name of the character device
 *
 * This function spawn a user-space worker to execute the real API on the
 * accelerator. The spawned worker is owned by `root`, and the working
 * directory is suspected to be `/root`. So in the worker code, it may
 * need `chdir` to change its working directory.
 */
static int umh_spawn_worker(const char *dev_name)
{
    struct subprocess_info *sub_info;
    char worker[128];
    char *argv[] = { "worker", (char *)global_kapi->name, (char *)dev_name, NULL };
    static char *envp[] = {
        "PATH=/sbin:/bin:/usr/sbin:/usr/bin", NULL };

    sprintf(worker, "%s/%s/worker", global_kapi->worker_path, global_kapi->name);

    sub_info = call_usermodehelper_setup(worker, argv, envp, GFP_ATOMIC,
                                        NULL, NULL, NULL);
    if (sub_info == NULL)
        return -ENOMEM;

    return call_usermodehelper_exec(sub_info, UMH_WAIT_EXEC);
}

/**
 * kava_chan_file_poll_new - Initialize a new command channel with a character device
 * as the interface
 *
 * This function returns a command channel which connects the kernel driver to only
 * one worker. Multiple kernel drivers will need multiple channels which shares one
 * kernel framework library. That means the command should contains the channel
 * information to avoid locks for accesses to the framework library.
 */
struct kava_chan *kava_chan_file_poll_new()
{
    int i;

    struct kava_chan *chan = (struct kava_chan *)kmalloc(sizeof(struct kava_chan), GFP_KERNEL);
    memset(chan, 0, sizeof(sizeof(struct kava_chan)));

    chan->id = KAVA_CHAN_FILE_POLL;
    chan->name = kava_chan_name[chan->id];

    /* Create character device */

    sprintf(chan->dev_name, "%s_%s", global_kapi->name, chan->name);
    for (i = 0; i < strlen(chan->dev_name); i++)
        chan->dev_name[i] = tolower(chan->dev_name[i]);

    register_chrdev(KAVA_CHAN_DEV_MAJOR, chan->dev_name, &fops);
    pr_info("Registered %s device with major number %d\n", chan->dev_name, KAVA_CHAN_DEV_MAJOR);

    if (!(dev_class = class_create(THIS_MODULE, KAVA_CHAN_DEV_CLASS))) {
        pr_err("%s class_create error\n", KAVA_CHAN_DEV_CLASS);
        goto unregister_dev;
    }
    dev_class->devnode = mod_dev_node;

    if (!(dev_node = device_create(dev_class, NULL,
                    MKDEV(KAVA_CHAN_DEV_MAJOR, chan->id), NULL, chan->dev_name))) {
        pr_err("%s device_create error\n", chan->dev_name);
        goto destroy_class;
    }

#ifdef __DEPRECATED_SPAWN_WORKER_BY_KLIB
    /* Spawn and connect the worker */
    int err;
    if ((err = umh_spawn_worker(chan->dev_name))) {
        pr_err("failed to spawn worker at %s/%s (%d)\n",
                global_kapi->worker_path, global_kapi->name, err);
        goto destroy_device;
    }
#endif

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

destroy_device:
    device_destroy(dev_class, MKDEV(KAVA_CHAN_DEV_MAJOR, chan->id));

destroy_class:
    class_unregister(dev_class);
    class_destroy(dev_class);

unregister_dev:
    unregister_chrdev(KAVA_CHAN_DEV_MAJOR, chan->dev_name);
    return NULL;
}
EXPORT_SYMBOL(kava_chan_file_poll_new);
