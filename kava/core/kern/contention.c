#include <linux/atomic.h>
#include <linux/err.h>
#include <linux/rhashtable.h>
#include <linux/slab.h>

#include "contention.h"


static atomic_t policy_counter = ATOMIC_INIT(0);

struct _policy_info_object {
    int key; /* policy_id */
    struct rhash_head linkage;

    kava_policy_t policy;
    kava_offload_func_t device_func;
    kava_offload_func_t cpu_func;
};

const static struct rhashtable_params _policy_info_object_params = {
    .key_len     = sizeof(const int),
    .key_offset  = offsetof(struct _policy_info_object, key),
    .head_offset = offsetof(struct _policy_info_object, linkage),
};

static struct rhashtable policy_table;


int kava_register_contention_policy(kava_policy_t policy,
                                    kava_offload_func_t device_func,
                                    kava_offload_func_t cpu_func)
{
    int id;
    struct _policy_info_object* object;
    int r;

    if (policy) {
        id = atomic_fetch_inc(&policy_counter);

        if (id == 0) {
            r = rhashtable_init(&policy_table, &_policy_info_object_params);
            if (r)
                return -ENOMEM;
        }

        /* Insert into hash table */
        object = vmalloc(sizeof(struct _policy_info_object));
        object->key = id;
        object->policy = policy;
        object->device_func = device_func;
        object->cpu_func = cpu_func;
        rhashtable_insert_fast(&policy_table, &object->linkage, _policy_info_object_params);

        return id;
    }

    return -EINVAL;
}
EXPORT_SYMBOL(kava_register_contention_policy);

static struct _policy_info_object *_lookup_policy_by_id(int policy_id)
{
    struct _policy_info_object *object =
        rhashtable_lookup_fast(&policy_table, &policy_id, _policy_info_object_params);
    return object;
}

void *kava_run_contention_func(int policy_id,
                               void *device_func_args,
                               void *cpu_func_args)
{
    struct _policy_info_object *object = _lookup_policy_by_id(policy_id);

    if (object == NULL)
        return ERR_PTR(-EINVAL);
    return object->policy(object->device_func, device_func_args,
                               object->cpu_func, cpu_func_args);
}
EXPORT_SYMBOL(kava_run_contention_func);

int kava_deregister_contention_policy(int policy_id)
{
    struct _policy_info_object *object;

    object = rhashtable_lookup_fast(&policy_table, &policy_id, _policy_info_object_params);
    if (object == NULL)
        return -EINVAL;

    rhashtable_remove_fast(&policy_table, &object->linkage, _policy_info_object_params);
    vfree(object);
    return 0;
}
EXPORT_SYMBOL(kava_deregister_contention_policy);
