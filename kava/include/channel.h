#ifndef __KAVA_CHANNEL_H__
#define __KAVA_CHANNEL_H__

#include "command.h"

typedef enum {
    KAVA_CHAN_FILE_POLL = 0,
    KAVA_CHAN_NL_SOCKET,
    KAVA_CHAN_MAX,
} kava_chan_mode;

static const char* const kava_chan_name[KAVA_CHAN_MAX] = {
    "FILE_POLL",
    "NETLINK_SOCKET",
};

#define KAVA_CHAN_DEV_MAJOR 150
#define KAVA_CHAN_DEV_CLASS "kava_chan"

struct kava_chan {
    kava_chan_mode id;
    const char *name;
    char dev_name[32];

    struct kava_cmd_base *(*cmd_new)(struct kava_chan *chan, size_t cmd_struct_size, size_t data_region_size);
    void (*cmd_send)(struct kava_chan *chan, struct kava_cmd_base *cmd);
    struct kava_cmd_base *(*cmd_receive)(struct kava_chan *chan);
    void (*cmd_free)(struct kava_chan *chan, struct kava_cmd_base *cmd);
    void (*cmd_print)(const struct kava_chan *chan, const struct kava_cmd_base *cmd);

    size_t (*chan_buffer_size)(const struct kava_chan *chan, size_t size);
    void* (*chan_attach_buffer)(struct kava_chan *chan, struct kava_cmd_base *cmd, const void *buffer, size_t size);
    void* (*chan_get_buffer)(const struct kava_chan *chan, const struct kava_cmd_base *cmd, void* buffer_id);
    void* (*chan_get_data_region)(const struct kava_chan *chan, const struct kava_cmd_base *cmd);
    void (*chan_free)(struct kava_chan *chan);
};

/**
 * The global channel used by this process (either the klib or the
 * worker).
 */
extern struct kava_chan *kava_global_cmd_chan;

#endif
