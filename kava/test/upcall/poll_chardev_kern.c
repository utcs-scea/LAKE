#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <linux/wait.h>
#include <asm/uaccess.h>
#include <linux/poll.h>

#include "upcall_impl.h"

#define CMD_RING_SIZE 1024
struct cmd_ring {
    struct base_buffer cmd[CMD_RING_SIZE];
    int head;
    int tail;

    spinlock_t idx_lock;
    struct semaphore count_sem;
    struct semaphore slot_sem;
};

struct upcall_handle {
    struct class *dev_class;
    struct device *dev_node;

    struct cmd_ring *send_cmdr;
    wait_queue_head_t poll_wait;

    volatile int __is_worker_connected;
};

upcall_handle_t handle;

static int chardev_open(struct inode *inode, struct file *filp)
{
    /* Create command ring */
    handle->send_cmdr = (struct cmd_ring *)kmalloc(sizeof(struct cmd_ring), GFP_KERNEL);
    if (handle->send_cmdr == NULL)
        return -ENOMEM;
    memset(handle->send_cmdr, 0, sizeof(struct cmd_ring));
    spin_lock_init(&handle->send_cmdr->idx_lock);
    sema_init(&handle->send_cmdr->count_sem, 0);
    sema_init(&handle->send_cmdr->slot_sem, CMD_RING_SIZE);
    init_waitqueue_head(&handle->poll_wait);

    /* Flag the connection of worker */
    handle->__is_worker_connected = 1;
    pr_info("Worker is connected\n");

    return 0;
}

static ssize_t chardev_read(struct file *filp, char __user *buf, size_t size, loff_t *offp)
{
    struct base_buffer *cmd;

    if (size != sizeof(struct base_buffer))
        return -EINVAL;

    wait_event_interruptible(handle->poll_wait, down_trylock(&handle->send_cmdr->count_sem) == 0);

    /* Spinlock is not needed because there is only one consumer */
    cmd = &handle->send_cmdr->cmd[handle->send_cmdr->head];
    handle->send_cmdr->head = (handle->send_cmdr->head + 1) & (CMD_RING_SIZE - 1);

    copy_to_user(buf, cmd, size);
    up(&handle->send_cmdr->slot_sem);

    return size;
}

static int chardev_release(struct inode *inode, struct file *filp)
{
    handle->__is_worker_connected = 0;
    kfree(handle->send_cmdr);
    pr_info("Worker is disconnected\n");

    return 0;
}

static unsigned int chardev_poll(struct file *filp, poll_table *wait)
{
    poll_wait(filp, &handle->poll_wait, wait);

    if (!down_trylock(&handle->send_cmdr->count_sem)) {
        up(&handle->send_cmdr->count_sem);
        return POLLIN | POLLRDNORM;
    }
    return 0;
}

static struct file_operations chardev_fops = {
	.open = chardev_open,
    .read = chardev_read,
    .poll = chardev_poll,
    .release = chardev_release,
};

upcall_handle_t init_upcall(void)
{
    handle = kmalloc(sizeof(struct upcall_handle), GFP_KERNEL);
    memset(handle, 0, sizeof(struct upcall_handle));

    register_chrdev(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_NAME, &chardev_fops);
    handle->dev_class = class_create(THIS_MODULE, UPCALL_TEST_DEV_CLASS);
    handle->dev_class->devnode = mod_dev_node;

    handle->dev_node = device_create(handle->dev_class, NULL,
            MKDEV(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_MINOR_NUM),
            NULL, UPCALL_TEST_DEV_NAME);

    return handle;
}

void close_upcall(upcall_handle_t handle)
{
    device_destroy(handle->dev_class,
                MKDEV(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_MINOR_NUM));
    class_unregister(handle->dev_class);
    class_destroy(handle->dev_class);
    unregister_chrdev(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_NAME);

    kfree(handle);
}

void _do_upcall(upcall_handle_t handle,
                uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
                void *buf, size_t size)
{
    struct base_buffer *cmd;
#if PRINT_TIME_K_TO_U
    struct timespec ts;
#endif

    if (!handle->__is_worker_connected) return;

    down(&handle->send_cmdr->slot_sem);

    spin_lock(&handle->send_cmdr->idx_lock);
    cmd = &handle->send_cmdr->cmd[handle->send_cmdr->tail];
    cmd->r0 = r0;
    cmd->r1 = r1;
    cmd->r2 = r2;
    cmd->r3 = r3;
    cmd->buf_size = size;
    handle->send_cmdr->tail = (handle->send_cmdr->tail + 1) & (CMD_RING_SIZE - 1);
    spin_unlock(&handle->send_cmdr->idx_lock);

    up(&handle->send_cmdr->count_sem);
#if PRINT_TIME_K_TO_U
    getnstimeofday(&ts);
    pr_info("Upcall called: sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);
#endif
    wake_up_interruptible(&handle->poll_wait);
}

static int __init test_upcall_init(void)
{
    create_test_device();
    handle = init_upcall();
    return 0;
}

static void __exit test_upcall_fini(void)
{
    close_upcall(handle);
    close_test_device();
}

module_init(test_upcall_init);
module_exit(test_upcall_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Upcall benchmarking module (chardev)");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
