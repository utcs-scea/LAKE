/*******************************************************************************

  This module provides the Control APIs for klib to manage the execution states.

*******************************************************************************/

#define pr_fmt(fmt) "[kava] %s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/kobject.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "api.h"
#include "command.h"
#include "config.h"
#include "control.h"

static struct class *dev_class;
static struct device *dev_node;

static struct kava_chan *chan;

static int kava_dev_open(struct inode *inode, struct file *filp)
{
    return 0;
}

static long kava_dev_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    int r = -EINVAL;

    switch (cmd) {
        case KAVA_IOCTL_STOP_WORKER:
            if (chan) {
                stop_worker(chan);
                r = 0;
            }
            break;

        default:
            pr_err("Unrecognized IOCTL command: %u\n", cmd);
    }

    return r;
}

static int kava_dev_release(struct inode *inode, struct file *filp)
{
    return 0;
}

static const struct file_operations fops =
{
    .owner          = THIS_MODULE,
    .open           = kava_dev_open,
    .unlocked_ioctl = kava_dev_ioctl,
    .release        = kava_dev_release,
};

static char *mod_dev_node(struct device *dev, umode_t *mode)
{
    if (mode)
        *mode = 0660;
    return NULL;
}

/**
 * ctrl_if_register_chan - Register channel in control interface
 * @chan: the channel via which the worker communicates
 */
int ctrl_if_register_chan(struct kava_chan *_chan) {
    if (chan) {
        pr_err("Channel has been registered in control interface\n");
        return -EBUSY;
    }
    chan = _chan;
    return 0;
}

/**
 * init_ctrl_if - Initialize control interface
 */
void init_ctrl_if(void)
{
    /* Create a device */
    register_chrdev(KAVA_DEV_MAJOR, KAVA_DEV_NAME, &fops);
    pr_info("Registered %s device with major number %d\n", KAVA_DEV_NAME, KAVA_DEV_MAJOR);

    if (!(dev_class = class_create(THIS_MODULE, KAVA_DEV_CLASS))) {
        pr_err("Create class %s error\n", KAVA_DEV_CLASS);
        goto unregister_dev;
    }
    dev_class->devnode = mod_dev_node;

    if (!(dev_node = device_create(dev_class, NULL,
                    MKDEV(KAVA_DEV_MAJOR, KAVA_DEV_MINOR), NULL, KAVA_DEV_NAME))) {
        pr_err("Create device %s error\n", KAVA_DEV_NAME);
        goto destroy_class;
    }

    return;

//destroy_device:
//    device_destroy(dev_class, MKDEV(KAVA_DEV_MAJOR, KAVA_DEV_MINOR));

destroy_class:
    class_unregister(dev_class);
    class_destroy(dev_class);

unregister_dev:
    unregister_chrdev(KAVA_DEV_MAJOR, KAVA_DEV_NAME);
}

/**
 * fini_ctrl_if - Release the control interface
 */
void fini_ctrl_if(void)
{
    device_destroy(dev_class, MKDEV(KAVA_DEV_MAJOR, KAVA_DEV_MINOR));
    class_unregister(dev_class);
    class_destroy(dev_class);
    unregister_chrdev(KAVA_DEV_MAJOR, KAVA_DEV_NAME);
}

/**
 * stop_worker - Stop the running worker
 * @chan: the channel via which the worker communicates
 */
void stop_worker(struct kava_chan *chan)
{
    struct kava_cmd_base *cmd =
        chan->cmd_new(chan, sizeof(struct kava_cmd_base), 0);
    cmd->mode = KAVA_CMD_MODE_INTERNAL;
    cmd->command_id = KAVA_CMD_ID_HANDLER_EXIT;
    chan->cmd_send(chan, cmd);
}
