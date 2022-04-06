#include "debug.h"
#include "endpoint.h"
#include "shadow_thread_pool.h"

// Wait for worker to read
// TODO: free the list at kapp's exit
GPtrArray *__kava_alloc_list_async_in;

// Wait for klib to read
// TODO: free the set at buffer's retrieval
GHashTable *__kava_alloc_list_async_out;

void __attribute__ ((constructor)) init_endpoint_lib(void)
{
    kava_global_metadata_map = kava_metadata_map_new();
    kava_shadow_thread_pool = kava_shadow_thread_pool_new();
}

__attribute__ ((pure))
static inline guint nw_hash_struct(gconstpointer ptr, int size) {
    guint ret;
    MurmurHash3_x86_32(ptr, size, 0xfbcdabc7 + size, &ret);
    return ret;
}

__attribute__ ((const))
static inline guint nw_hash_pointer(gconstpointer ptr) {
    return kava_hash_mix64variant13(ptr);
}

EXPORTED_WEAKLY void kava_endpoint_init(struct kava_endpoint *endpoint,
                                        size_t metadata_size)
{
    endpoint->metadata_size = metadata_size;

    endpoint->call_map = kava_metadata_map_new();
    atomic_init(&endpoint->call_counter, 0);
    pthread_mutex_init(&endpoint->managed_buffer_map_mutex, NULL);
    pthread_mutex_init(&endpoint->call_map_mutex, NULL);

    __kava_alloc_list_async_in =
        g_ptr_array_new_full(0, (GDestroyNotify)kava_buffer_with_deallocator_free);
    __kava_alloc_list_async_out =
        g_hash_table_new_full(g_direct_hash, g_direct_equal, NULL, g_free);
}

EXPORTED_WEAKLY void kava_endpoint_destroy(struct kava_endpoint *endpoint)
{
    g_hash_table_unref(endpoint->managed_buffer_map);
    g_hash_table_unref(endpoint->managed_by_coupled_map);
    // g_hash_table_unref(endpoint->metadata_map);
    g_hash_table_unref(endpoint->call_map);
}

struct kava_buffer_with_deallocator {
    void (*deallocator)(void *);
    void *buffer;
};

EXPORTED_WEAKLY struct kava_buffer_with_deallocator *kava_buffer_with_deallocator_new(
        void (*deallocator)(void *), void *buffer)
{
    struct kava_buffer_with_deallocator *ret = malloc(
            sizeof(struct kava_buffer_with_deallocator));
    ret->buffer = buffer;
    ret->deallocator = deallocator;
    return ret;
}

EXPORTED_WEAKLY void kava_buffer_with_deallocator_free(
        struct kava_buffer_with_deallocator *buffer)
{
    buffer->deallocator(buffer->buffer);
    free(buffer);
}
