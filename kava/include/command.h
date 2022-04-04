#ifndef __KAVA_COMMAND_H__
#define __KAVA_COMMAND_H__

#include "util.h"

typedef enum {
    KAVA_CMD_MODE_INTERNAL = 0,
    KAVA_CMD_MODE_API,
    KAVA_CMD_MODE_MAX
} kava_cmd_mode;

typedef enum {
    KAVA_CMD_ID_UNSPEC = 0,
    KAVA_CMD_ID_HANDLER_THREAD_EXIT,
    KAVA_CMD_ID_HANDLER_EXIT,
    KAVA_CMD_ID_CHANNEL_INIT,
    KAVA_CMD_ID_MAX,
} kava_internal_cmd_id;

struct kava_cmd_base {
    /**
     * The mode of the command. This identifies whether the command is an
     * internal command or a library API command.
     */
    kava_cmd_mode mode;
    /**
     * The VM ID is assigned by hypervisor.
     * FIXME: this should not be sent by guestlib but I have not found any
     * workaround.
     */
    uintptr_t command_type;
    /**
     * The ID of the thread which sent this command.
     */
    int64_t thread_id;
    /**
     * The command ID (within the API). This ID defines what fields this
     * command struct contains.
     */
    uintptr_t command_id;
    /**
     * The size of this command struct.
     */
    size_t command_size;
    /**
     * A reference to the data region associated with this command. It
     * may be a pointer, but can also be an offset or something else.
     */
    void* data_region;
    /**
     * The size of the data region attached to this command.
     */
    size_t region_size;
    /**
     * Reserved region for other purposes, for example, param_block seeker
     * in shared memory implementation. */
    char reserved_area[64];
};

#endif
