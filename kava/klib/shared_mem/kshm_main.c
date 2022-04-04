/*******************************************************************************

  Kernel-space shared memory driver.

*******************************************************************************/

#include <linux/module.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "config.h"
#include "debug.h"
#include "shared_memory.h"
#include "test.h"

static struct device *dev_node;
static struct class  *dev_class;

static long shm_size = KAVA_DEFAULT_SHARED_MEM_SIZE;
module_param(shm_size, long, 0444);
MODULE_PARM_DESC(shm_size, "Shared memory size in MB, default 32 MB");

static int run_test = 0;
module_param(run_test, int, 0444);
MODULE_PARM_DESC(run_test, "Run example tests");

static int kshm_open(struct inode *inode, struct file *filp)
{
    return 0;
}

static int kshm_release(struct inode *inode, struct file *filp)
{
    return 0;
}

static long kshm_ioctl(struct file *filp, unsigned int cmd,
                       unsigned long arg)
{
    int r = -EINVAL;
    long size_in_bytes;

    switch (cmd)
    {
    case KAVA_SHM_GET_SHM_SIZE:
        size_in_bytes = (shm_size << 20);
        copy_to_user((void *)arg, (void *)&size_in_bytes, sizeof(long));
        r = 0;
        break;

    default:
        pr_err("[kava-shm] Unsupported IOCTL command\n");
    }

    return r;
}

static const struct file_operations fops =
{
    .owner          = THIS_MODULE,
    .open           = kshm_open,
    .mmap           = kshm_mmap_helper,
    .release        = kshm_release,
    .unlocked_ioctl = kshm_ioctl,
};

static char *mod_dev_node(struct device *dev, umode_t *mode)
{
    if (mode)
        *mode = 0444;
    return NULL;
}

static int __init kshm_init(void)
{
    /* Register chardev */
    int err = -ENOMEM;

    register_chrdev(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_NAME, &fops);
    pr_info("[kava-shm] Registered shared memory device with major number %d\n", KAVA_SHM_DEV_MAJOR);

    if (!(dev_class = class_create(THIS_MODULE, KAVA_SHM_DEV_CLASS))) {
        pr_err("[kava-shm] Class_create error\n");
        goto unregister_dev;
    }
    dev_class->devnode = mod_dev_node;

    if (!(dev_node = device_create(dev_class, NULL,
                    MKDEV(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_MINOR), NULL, KAVA_SHM_DEV_NAME))) {
        pr_err("[kava-shm] Device_create error\n");
        goto destroy_class;

    }
    pr_info("[kava-shm] Create shared memory device\n");

    /* Initialize allocator */
    kava_allocator_init((shm_size << 20));

    /* Sample tests */
    if (run_test) {
        test_alloc_and_free(shm_allocator);
    }

    return 0;

destroy_device:
    device_destroy(dev_class, MKDEV(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_MINOR));

destroy_class:
    class_unregister(dev_class);
    class_destroy(dev_class);

unregister_dev:
    unregister_chrdev(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_NAME);
    return err;
}

static void __exit kshm_fini(void)
{
    kava_allocator_fini();

    device_destroy(dev_class, MKDEV(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_MINOR));
    class_unregister(dev_class);
    class_destroy(dev_class);
    unregister_chrdev(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_NAME);
}

module_init(kshm_init);
module_exit(kshm_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("KAvA shared memory driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
