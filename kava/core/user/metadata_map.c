#include <stdint.h>

#include "metadata_map.h"

GHashTable *kava_global_metadata_map;
pthread_mutex_t kava_global_metadata_map_mutex = PTHREAD_MUTEX_INITIALIZER;

/**
 * hash_mix64variant13 - 64-bit to 64-bit hash
 * @ptr: the pointer to be hashed
 *
 * This function is modified to produce a 32-bit result from
 * http://dx.doi.org/10.1145/2714064.2660195. This hash was chosen based
 * on the paper above and https://nullprogram.com/blog/2018/07/31/.
 */
static guint hash_mix64variant13(gconstpointer ptr) {
    uintptr_t x = (uintptr_t)ptr;
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return (guint)x;
}

/**
 * metadata_map_new - Create a new metadata map
 */
GHashTable *kava_metadata_map_new() {
    return g_hash_table_new(hash_mix64variant13, g_direct_equal);
}

static struct kava_metadata_base *internal_metadata_unlocked(
                                    struct kava_endpoint *endpoint,
                                    const void *ptr)
{
    void *metadata = g_hash_table_lookup(kava_metadata_map, ptr);
    if (metadata == NULL) {
        metadata = calloc(1, endpoint->metadata_size);
        g_hash_table_insert(kava_metadata_map, (void*)ptr, metadata);
    }
    return (struct kava_metadata_base *)metadata;
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
    pthread_mutex_lock(&kava_metadata_map_mutex);
    struct kava_metadata_base *ret = internal_metadata_unlocked(endpoint, ptr);
    pthread_mutex_unlock(&kava_metadata_map_mutex);
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
    pthread_mutex_lock(&kava_metadata_map_mutex);
    struct kava_metadata_base *ret = g_hash_table_lookup(kava_metadata_map, ptr);
    pthread_mutex_unlock(&kava_metadata_map_mutex);
    return ret;
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
    pthread_mutex_lock(&kava_metadata_map_mutex);
    void *metadata = g_hash_table_lookup(kava_metadata_map, ptr);
    if (metadata != NULL) {
        free(metadata);
        g_hash_table_steal(kava_metadata_map, ptr);
    }
    pthread_mutex_unlock(&kava_metadata_map_mutex);
}
