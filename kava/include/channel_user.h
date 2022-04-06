#if defined(__KAVA_CHANNEL_KERN_H__)
#error channel_user.h and channel_kern.h are exclusive headers and cannot be included at the same time.
#endif

#ifndef __KAVA_CHANNEL_USER_H__
#define __KAVA_CHANNEL_USER_H__

#include "channel.h"

struct kava_chan *kava_chan_file_poll_new(const char *dev_name);
struct kava_chan *kava_chan_nl_socket_new(const char *dev_name);

#endif
