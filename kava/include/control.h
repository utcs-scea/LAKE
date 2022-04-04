#ifndef __KAVA_CONTROL_H__
#define __KAVA_CONTROL_H__

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
#else
#include <sys/ioctl.h>
#include <stdint.h>
#endif

#include "channel.h"
#include "config.h"

/**
 * stop_worker - Stop the running worker
 * @chan: the channel via which the worker communicates
 */
void stop_worker(struct kava_chan *chan);

/**
 * ctrl_if_register_chan - Register channel in control interface
 * @chan: the channel via which the worker communicates
 *
 * On success, this function returns 0.
 */
int ctrl_if_register_chan(struct kava_chan *_chan);

/**
 * init_ctrl_if - Initialize control interface
 */
void init_ctrl_if(void);

/**
 * fini_ctrl_if - Release the control interface
 */
void fini_ctrl_if(void);

#define KAVA_IOCTL_STOP_WORKER _IO(KAVA_DEV_MAJOR, 0x10)

#endif
