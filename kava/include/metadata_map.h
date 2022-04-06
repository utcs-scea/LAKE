#ifndef __KAVA_METADATA_MAP_H__
#define __KAVA_METADATA_MAP_H__

#ifndef __KERNEL__

#include <glib.h>
#include <pthread.h>

#else

#include <linux/kthread.h>
#include <linux/rhashtable.h>
typedef struct rhashtable GHashTable;

#endif

#include "endpoint.h"

#define kava_metadata_map        kava_global_metadata_map
#define kava_metadata_map_mutex  kava_global_metadata_map_mutex

struct kava_metadata_base {
    /* For handles and buffers */
    struct kava_shadow_record_t *shadow;

    /* For buffers */
    kava_deallocator deallocator;
};

#ifndef __KERNEL__

extern GHashTable *kava_global_metadata_map;

/**
 * kava_metadata_map_new - Create a new metadata map
 */
GHashTable *kava_metadata_map_new(void);

#else

extern struct rhashtable *kava_global_metadata_map;

struct rhashtable *kava_metadata_map_new(void);

/**
 * kava_metadata_map_free - Destroy a metadata map
 * @rhtable: the map to be freed
 */
void kava_metadata_map_free(struct rhashtable * rhtable);

#endif

/**
 * kava_internal_metadata - Get the metadata for an object and creates the
 * metadata if it does not exist
 * @endpoint: the endpoint to which the metadata is attached
 * @ptr: the handle with which that returned metadata is associated
 *
 * The `pure` attribute tells the compiler that subsequent calls to the function
 * with the same `ptr` can be replaced by the result of the first call provided
 * the state of the program observable by this function. This attribute is a bit
 * of a stretch since we do insert elements into the hash table, but having 
 * the compiler perform CSE on this function is pretty important and this
 * function is idempotent.
 */
__attribute__ ((pure))
struct kava_metadata_base *kava_internal_metadata(struct kava_endpoint *endpoint,
                                                const void *ptr);

/**
 * kava_internal_metadata_no_create - Get the metadata for an object
 * @endpoint: the endpoint to which the metadata is attached
 * @ptr: the handle with which that returned metadata is associated
 *
 * This function should create the metadata if it does not exist.
 */
struct kava_metadata_base *kava_internal_metadata_no_create(
                            struct kava_endpoint *endpoint,
                            const void *ptr);

/**
 * kava_internal_metadata_remove - Free and remove the metadata for an object
 * @endpoint: the endpoint to which the metadata is attached
 * @ptr: the handle with which that returned metadata is associated
 *
 * This function also frees the metadata object.
 */
void kava_internal_metadata_remove(struct kava_endpoint *endpoint,
                            const void *ptr);

#ifdef __KERNEL__

/**
 * kava_metadata_map_insert - Insert a new metadata into the map
 * @rhtable: the metadata map to which the metadata is inserted
 * * @key: the key to be inserted to the map
 * @metadata: the metadata associated with the key
 *
 * This function returns 0 for success, otherwise -1.
 */
int kava_metadata_map_insert(struct rhashtable *rhtable, void *key, void *metadata);

/**
 * kava_metadata_map_remove - Remove a key from the map
 * @rhtable: the metadata map from which the key is removed
 * @key: the key to be removed from the map
 *
 * This function returns the metadata associated with the removed key.
 */
void *kava_metadata_map_remove(struct rhashtable *rhtable, void *key);

#endif

#endif
