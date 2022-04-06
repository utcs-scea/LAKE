#include <linux/slab.h>
#include <linux/types.h>

#include "metadata_map.h"

struct rhashtable *kava_global_metadata_map;
DEFINE_MUTEX(kava_global_metadata_map_mutex);

struct metadata_object {
    const void *key;
    struct rhash_head linkage;
    struct kava_metadata_base *metadata;
};

const static struct rhashtable_params metadata_object_params = {
    .key_len     = sizeof(const void *),
    .key_offset  = offsetof(struct metadata_object, key),
    .head_offset = offsetof(struct metadata_object, linkage),
};

/**
 * kava_metadata_map_new - Create a new metadata map
 */
struct rhashtable *kava_metadata_map_new(void) {
    struct rhashtable *metadata_map;
    int r;
    metadata_map = kmalloc(sizeof(struct rhashtable), GFP_KERNEL);
    r = rhashtable_init(metadata_map, &metadata_object_params);
    if (r) {
        kfree(metadata_map);
        return NULL;
    }
    return metadata_map;
}

static struct kava_metadata_base *internal_metadata_unlocked(
                                    struct kava_endpoint *endpoint,
                                    const void *ptr)
{
    struct metadata_object *object;
    struct kava_metadata_base *metadata;
    object = rhashtable_lookup_fast(kava_metadata_map, &ptr, metadata_object_params);
    if (object == NULL) {
        metadata = (struct kava_metadata_base *)vmalloc(endpoint->metadata_size);
        memset(metadata, 0, endpoint->metadata_size);
        object = (struct metadata_object *)vmalloc(sizeof(struct metadata_object));
        object->key = ptr;
        object->metadata = metadata;
        rhashtable_insert_fast(kava_metadata_map, &object->linkage, metadata_object_params);
    }
    return object->metadata;
}

/**
 * kava_internal_metadata - Get the metadata for an object and creates the
 * metadata if it does not exist
 * @endpoint: the endpoint to which the metadata is attached
 * @ptr: the handle with which that returned metadata is associated
 */
struct kava_metadata_base *kava_internal_metadata(struct kava_endpoint *endpoint,
                                                const void *ptr)
{
    struct kava_metadata_base *ret;

    mutex_lock(&kava_metadata_map_mutex);
    ret = internal_metadata_unlocked(endpoint, ptr);
    mutex_unlock(&kava_metadata_map_mutex);
    return ret;
}

/**
 * kava_internal_metadata_no_create - Get the metadata for an object
 * @endpoint: the endpoint to which the metadata is attached
 * @ptr: the handle with which that returned metadata is associated
 *
 * This function does not create the metadata if it does not exist.
 */
struct kava_metadata_base *kava_internal_metadata_no_create(
                            struct kava_endpoint *endpoint,
                            const void *ptr)
{
    struct metadata_object *ret;

    mutex_lock(&kava_metadata_map_mutex);
    ret = rhashtable_lookup_fast(kava_metadata_map, &ptr, metadata_object_params);
    mutex_unlock(&kava_metadata_map_mutex);
    return (ret ? ret->metadata : NULL);
}

/**
 * kava_internal_metadata_remove - Free and remove the metadata for an object
 * @endpoint: the endpoint to which the metadata is attached
 * @ptr: the handle with which that returned metadata is associated
 *
 * This function also frees the metadata object.
 */
void kava_internal_metadata_remove(struct kava_endpoint *endpoint,
                            const void *ptr)
{
    struct metadata_object *object;

    mutex_lock(&kava_metadata_map_mutex);
    object = rhashtable_lookup_fast(kava_metadata_map, &ptr, metadata_object_params);
    if (object != NULL) {
        rhashtable_remove_fast(kava_metadata_map, &object->linkage, metadata_object_params);
        vfree(object->metadata);
        vfree(object);
    }
    mutex_unlock(&kava_metadata_map_mutex);
}

/**
 * kava_metadata_map_insert - Insert a new metadata into the map
 * @rhtable: the metadata map to which the metadata is inserted
 * @key: the key to be inserted to the map
 * @metadata: the metadata associated with the key
 *
 * This function returns 0 for success, otherwise -1.
 */
int kava_metadata_map_insert(struct rhashtable *rhtable, void *key, void *metadata)
{
    struct metadata_object *object;

    object = vmalloc(sizeof(struct metadata_object));
    object->key = key;
    object->metadata = (struct kava_metadata_base *)metadata;
    return rhashtable_insert_fast(rhtable, &object->linkage, metadata_object_params);
}

/**
 * kava_metadata_map_remove - Remove a key from the map
 * @rhtable: the metadata map from which the key is removed
 * @key: the key to be removed from the map
 *
 * This function returns the metadata associated with the removed key.
 */
void *kava_metadata_map_remove(struct rhashtable *rhtable, void *key)
{
    struct metadata_object *object;
    struct kava_metadata_base *metadata = NULL;

    object = rhashtable_lookup_fast(rhtable, &key, metadata_object_params);
    if (object)
        metadata = object->metadata;
    rhashtable_remove_fast(rhtable, &object->linkage, metadata_object_params);

    return (void *)metadata;
}

static void metadata_map_free_fn(void *ptr, void *arg)
{
    struct metadata_object *obj = (struct metadata_object *)ptr;
    if (obj) {
        vfree(obj->metadata);
        vfree(obj);
    }
}

/**
 * kava_metadata_map_free - Destroy a metadata map
 * @rhtable: the map to be freed
 */
void kava_metadata_map_free(struct rhashtable *rhtable)
{
    rhashtable_free_and_destroy(rhtable, metadata_map_free_fn, NULL);
    kfree(rhtable);
}
