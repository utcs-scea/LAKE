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

    int pid;
    struct task_struct *t;
    wait_queue_head_t wait_ack;
    volatile int __ack_received;
};

upcall_handle_t handle;

static int chardev_open(struct inode *inode, struct file *filp)
{
    init_waitqueue_head(&handle->wait_ack);
    handle->__ack_received = 0;
    return 0;
}

static long chardev_ioctl(struct file *filp,
                        unsigned int cmd,
                        unsigned long arg)
{
    int r = -EINVAL;

    switch (cmd) {
        case KAVA_SET_USER_PID:
            pr_info("PID is set to %d\n", (int)arg);
            handle->pid = (int)arg;

            rcu_read_lock();
            handle->t = pid_task(find_pid_ns(handle->pid, &init_pid_ns), PIDTYPE_PID);
            rcu_read_unlock();
            if (handle->t == NULL) {
                pr_err("PID %d not found\n", handle->pid);
                r = -EINVAL;
            }
            else {
                r = 0;
            }

            break;

        case KAVA_ACK_SINGAL:
            handle->__ack_received = 1;
            wake_up_interruptible(&handle->wait_ack);
            r = 0;
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

    kfree(handle);
}

void _do_upcall(upcall_handle_t handle,
                uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
                void *buf, size_t size)
{
#if LINUX_VERSION_CODE < KERNEL_VERSION(5,0,0)
    struct siginfo info;
#else
    struct kernel_siginfo info;
#endif
#if PRINT_TIME_K_TO_U
    struct timespec ts;
#endif

    memset(&info, 0, sizeof(info));
    info.si_signo = UPCALL_TEST_SIG;
    /**
     * SI_QUEUE is normally used by sigqueue from user space, and
     * kernel space should use SI_KERNEL. But if SI_KERNEL is used
     * the real_time data  is not delivered to the user space signal
     * handler function.
     */
    info.si_code = SI_QUEUE;

    /**
     * Store data to be sent
     * https://elixir.bootlin.com/linux/v4.15/source/include/uapi/asm-generic/siginfo.h#L49
     */
    info.si_utime = (__ARCH_SI_CLOCK_T)r0;
    info.si_stime = (__ARCH_SI_CLOCK_T)r1;
    info.si_ptr = (void *)r2;
    info.si_addr = (void *)r3;
    info.si_call_addr = (void *)size;

#if PRINT_TIME_K_TO_U
    getnstimeofday(&ts);
    pr_info("Upcall called: sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);
#endif

    if (handle->t != NULL) {
        if (send_sig_info(UPCALL_TEST_SIG, &info, handle->t) < 0)
            pr_err("Send signal failed\n");
    }

    /**
     * Spin until acked from user-space.
     * In the real-world, this should be moved to the top of do_upcall.
     */
    wait_event_interruptible(handle->wait_ack, handle->__ack_received == 1);
    handle->__ack_received = 0;
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
