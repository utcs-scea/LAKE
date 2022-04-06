#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#include <asm/uaccess.h>

#include <asm/siginfo.h>
#include <linux/rcupdate.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>

#include "upcall_impl.h"

struct upcall_handle {
    struct class *dev_class;
    struct device *dev_node;

    size_t size;
    void *buf;
    struct task_struct *t;
};

upcall_handle_t handle;

static int chardev_open(struct inode *inode, struct file *filp)
{
    return 0;
}

static long chardev_ioctl(struct file *filp,
                        unsigned int cmd,
                        unsigned long arg)
{
    int r = -EINVAL;

    switch (cmd) {
        case KAVA_SET_MESSAGE_SIZE:
            handle->size = (size_t)((int)arg);
            pr_info("Message size is set to %lu\n", handle->size);

            if (handle->buf)
                vfree(handle->buf);
            handle->buf = vmalloc(handle->size);
            r = 0;
            break;

        case KAVA_SEND_MESSAGE:
            if (handle->buf) {
                copy_from_user(handle->buf, (void *)arg, handle->size);
                r = 0;
            }
            break;

        case KAVA_RECV_MESSAGE:
            if (handle->buf) {
                copy_to_user((void *)arg, handle->buf, handle->size);
                r = 0;
            }
            break;

        default:
            pr_err("Unrecognized IOCTL command: %u\n", cmd);
    }

    return r;
}

static int chardev_release(struct inode *inode, struct file *filp)
{
    return 0;
}

static struct file_operations chardev_fops = {
	.open = chardev_open,
    .unlocked_ioctl = chardev_ioctl,
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

    if (handle->buf)
        vfree(handle->buf);
    kfree(handle);
}

void _do_upcall(upcall_handle_t handle,
                uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
                void *buf, size_t size)
{
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
MODULE_DESCRIPTION("Upcall benchmarking module (signal)");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
