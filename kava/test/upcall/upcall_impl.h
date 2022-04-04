#ifndef __KAVA_UPCALL_IMPL_H__
#define __KAVA_UPCALL_IMPL_H__

#include "debug.h"
#include "upcall.h"

#define PRINT_TIME_K_TO_U 1
#define PRINT_TIME_U_TO_K 0

struct base_buffer {
    uint64_t r0;
    uint64_t r1;
    uint64_t r2;
    uint64_t r3;
    uint64_t buf_size;
};

/**
 * Netlink
 */
#define NETLINK_USER 31
#define NL_MSG_LEN_MAX (16 << 20)

/**
 * IOCTL interface for all implementations or polling for chardev
 */
#define UPCALL_TEST_DEV_NAME  "kava_upcall_test"
#define UPCALL_TEST_DEV_CLASS "kava_upcall_test"
#define UPCALL_TEST_DEV_MAJOR_NUM 150
#define UPCALL_TEST_DEV_MINOR_NUM 98

/**
 * Signal
 */
#define UPCALL_TEST_SIG 44
#define KAVA_SET_USER_PID _IOR(UPCALL_TEST_DEV_MAJOR_NUM, 0x1, int)
#define KAVA_ACK_SINGAL   _IO(UPCALL_TEST_DEV_MAJOR_NUM, 0x2)

/**
 * Shared memory
 */
#define MMAP_NUM_PAGES 1

struct shared_region {
    volatile int doorbell;
    struct base_buffer data;
};

/**
 * Tester
 */
#define UPCALL_DEV_NAME    "kava_upcall"
#define UPCALL_DEV_CLASS   "kava_upcall"
#define MAJOR_NUM 151
#define MINOR_NUM 98

#define KAVA_TEST_START_UPCALL _IOR(MAJOR_NUM, 0x1, int)

#ifdef __KERNEL__

#include <linux/device.h>
#include <linux/ioctl.h>
#include <linux/fs.h>
#include <linux/time.h>
#include <linux/delay.h>
#include <linux/uaccess.h>

extern upcall_handle_t handle;

static int upcall_dev_open(struct inode *inode, struct file *filp)
{
    return 0;
}

static long upcall_dev_ioctl(struct file *filp,
                            unsigned int cmd,
                            unsigned long arg)
{
    int r = -EINVAL;
    unsigned long i;

    switch (cmd) {
        case KAVA_TEST_START_UPCALL:
            pr_info("Upcall for %lu times\n", arg);
            for (i = 0; i < arg; i++) {
                do_upcall(handle);

                /**
                 * Singal and shared memory need spin on acknowledge
                 * from user-space worker. To measure a round-trip, it
                 * should use `upcall_exit - upcall_called`.
                 */
#if PRINT_TIME_U_TO_K
                getnstimeofday(&ts);
                pr_info("Upcall[%lu] exit: sec=%lu, usec=%lu\n", i, ts.tv_sec, ts.tv_nsec / 1000);
#endif

                if (i < arg - 1)
                    msleep(100);
            }

            r = 0;
            break;

        default:
            pr_err("Unrecognized IOCTL command: %u\n", cmd);
    }

    return r;
}

static struct file_operations fops = {
	.open = upcall_dev_open,
	.unlocked_ioctl = upcall_dev_ioctl,
};

static char *mod_dev_node(struct device *dev, umode_t *mode)
{
    if (mode)
        *mode = 0666;
    return NULL;
}

static struct class *dev_class;
static struct device *dev_node;

static void create_test_device(void) {
    register_chrdev(MAJOR_NUM, UPCALL_DEV_NAME, &fops);
    dev_class = class_create(THIS_MODULE, UPCALL_DEV_CLASS);
    dev_class->devnode = mod_dev_node;

    dev_node = device_create(dev_class, NULL,
            MKDEV(MAJOR_NUM, MINOR_NUM), NULL, UPCALL_DEV_NAME);
}

static void close_test_device(void) {
    device_destroy(dev_class, MKDEV(MAJOR_NUM, MINOR_NUM));
    class_unregister(dev_class);
    class_destroy(dev_class);
    unregister_chrdev(MAJOR_NUM, UPCALL_DEV_NAME);
}

#endif

#endif
