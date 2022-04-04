/*******************************************************************************

  Demo linux driver module for kernel-space CUDA API.

*******************************************************************************/

#define pr_fmt(fmt) "[kava] %s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/ctype.h>
#include <linux/fs.h>
#include <linux/kobject.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "api.h"
#include "channel.h"
#include "config.h"

struct kava_api *global_kapi = NULL;

/**
 * Channel sysfs entry helper functions.
 */
static ssize_t chan_mode_show(struct kobject *kobj,
                                struct kobj_attribute *attr,
                                char *buf)
{
    const char *name = kava_chan_name[global_kapi->chan_mode];
    return snprintf(buf, strlen(name) + 1, "%s", name);
}

static ssize_t chan_mode_store(struct kobject *kobj,
                                struct kobj_attribute *attr,
                                const char *buf, size_t count)
{
    int i, j;
    for (i = KAVA_CHAN_MAX - 1; i >= 0; i--) {
        if (strlen(kava_chan_name[i]) != strlen(buf))
            continue;
        for (j = 0; j < strlen(buf); j++)
            if (tolower(buf[j]) != tolower(kava_chan_name[i][j]))
                break;
        if (j == strlen(buf))
            break;
    }
    if (i >= 0)
        global_kapi->chan_mode = i;
    return count;
}

/**
 * Shared sysfs entry helper functions.
 */

/**
 * Channel sysfs entry attribute.
 */
static struct kobj_attribute chan_mode_attr = __ATTR(channel,
                                    0444, chan_mode_show, chan_mode_store);

static struct attribute *attrs[] = {
    &chan_mode_attr.attr,
    NULL, /* need to NULL terminate the list of attributes */
};

/**
 * The attribute group's name needs to be specified, so that a subdirectory
 * will be created for the attributes.
 */
static struct attribute_group attr_group = {
    .attrs = attrs,
};

/**
 * create_sysfs_entry - Create sysfs entries
 */
int create_sysfs_entry(void)
{
    const char *name;
    char entry_path[128];
    int err;

    BUG_ON(global_kapi == NULL);

    name = global_kapi->name;
    sprintf(entry_path, "kava_%s", name);

    /* Create /sys/kernel/kava_$(API_NAME) directory */
    global_kapi->sysfs_kobj = kobject_create_and_add(entry_path, kernel_kobj);
    if (!global_kapi->sysfs_kobj)
        return -ENOMEM;

    /* Create sysfs entries */
    err = sysfs_create_group(global_kapi->sysfs_kobj, &attr_group);
    if (err) {
        pr_debug("fail to create group at %s\n", entry_path);
        kobject_put(global_kapi->sysfs_kobj);
        global_kapi->sysfs_kobj = NULL;
    }

    return err;
}

/**
 * put_sysfs_entry - Release sysfs entries
 */
void put_sysfs_entry(void)
{
    if (global_kapi->sysfs_kobj)
        kobject_put(global_kapi->sysfs_kobj);
}

/**
 * init_global_kapi - Initialize global_kapi
 * @id: API ID
 */
void init_global_kapi(kava_api_id id, const char *chan_mode)
{
    BUG_ON(global_kapi != NULL);
    BUG_ON(id == KAVA_API_ID_UNSPEC || id >= KAVA_API_ID_MAX);

    global_kapi = kmalloc(sizeof(struct kava_api), GFP_KERNEL);
    memset(global_kapi, 0, sizeof(struct kava_api));

    global_kapi->id = id;
    global_kapi->name = kava_api_name[id];
    if (!chan_mode)
        global_kapi->chan_mode = 0;
    else
        chan_mode_store(NULL, NULL, chan_mode, strlen(chan_mode));
    pr_info("%s is using channel: %s\n",
            kava_api_name[id], kava_chan_name[global_kapi->chan_mode]);

    create_sysfs_entry();
}

/**
 * put_global_kapi - Release global_kapi
 */
void put_global_kapi(void)
{
    put_sysfs_entry();
    kfree(global_kapi);
}
