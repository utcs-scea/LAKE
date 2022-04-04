#include <linux/list.h>

#include "debug.h"
#include "endpoint.h"
#include "shadow_thread_pool.h"

EXPORTED_WEAKLY void __attribute__ ((constructor)) init_endpoint_lib(void)
{
    kava_global_metadata_map = kava_metadata_map_new();
    kava_shadow_thread_pool = kava_shadow_thread_pool_new();
}

EXPORTED_WEAKLY void kava_endpoint_init(struct kava_endpoint *endpoint,
                                        size_t metadata_size)
{
    endpoint->metadata_size = metadata_size;

    endpoint->call_map = kava_metadata_map_new();
    endpoint->call_counter = 0;
    mutex_init(&endpoint->managed_buffer_map_mutex);
    mutex_init(&endpoint->call_map_mutex);
}

EXPORTED_WEAKLY void kava_endpoint_destroy(struct kava_endpoint *endpoint)
{
    // g_hash_table_unref(endpoint->managed_buffer_map);
    // g_hash_table_unref(endpoint->managed_by_coupled_map);
    // kava_metadata_map_free(endpoint->metadata_map);
    kava_metadata_map_free(endpoint->call_map);
}

EXPORTED_WEAKLY intptr_t kava_get_call_id(struct kava_endpoint *endpoint)
{
    return __atomic_fetch_add(&endpoint->call_counter, 1, __ATOMIC_RELEASE);
}

EXPORTED_WEAKLY void kava_add_call(struct kava_endpoint* endpoint, intptr_t id, void *ptr)
{
    int ret;
    mutex_lock(&endpoint->call_map_mutex);
    ret = kava_metadata_map_insert(endpoint->call_map, (void *)id, ptr);
    BUG_ON(ret && "Adding a call ID which currently exists.");
    mutex_unlock(&endpoint->call_map_mutex);
}

EXPORTED_WEAKLY void *kava_remove_call(struct kava_endpoint* endpoint, intptr_t id)
{
    void *ptr;
    mutex_lock(&endpoint->call_map_mutex);
    ptr = kava_metadata_map_remove(endpoint->call_map, (void *)id);
    BUG_ON(ptr == NULL && "Removing a call ID which does not exist");
    mutex_unlock(&endpoint->call_map_mutex);
    return ptr;
}

struct kava_buffer_with_deallocator {
    void (*deallocator)(const void *);
    void *buffer;
};

EXPORTED_WEAKLY struct kava_buffer_with_deallocator *kava_buffer_with_deallocator_new(
        void (*deallocator)(const void *), void *buffer)
{
    struct kava_buffer_with_deallocator *ret = vmalloc(
            sizeof(struct kava_buffer_with_deallocator));
    ret->buffer = buffer;
    ret->deallocator = deallocator;
    return ret;
}

void kava_buffer_with_deallocator_free(struct kava_buffer_with_deallocator *buffer)
{
    if (buffer->deallocator && buffer->buffer)
        buffer->deallocator(buffer->buffer);
    vfree(buffer);
}

struct kava_buffer_list {
    struct kava_buffer_with_deallocator *buf;
    struct list_head list;
};

EXPORTED_WEAKLY struct kava_buffer_list *kava_endpoint_buffer_list_new(void)
{
    struct kava_buffer_list *buf_list = (struct kava_buffer_list *)vmalloc(
            sizeof(struct kava_buffer_list));
    buf_list->buf = kava_buffer_with_deallocator_new(NULL, NULL);
    INIT_LIST_HEAD(&buf_list->list);
    return buf_list;
}

EXPORTED_WEAKLY void kava_endpoint_buffer_list_add(
        struct kava_buffer_list *buf_list,
        struct kava_buffer_with_deallocator *buf)
{
    struct kava_buffer_list *buf_node = (struct kava_buffer_list *)vmalloc(
            sizeof(struct kava_buffer_list));
    buf_node->buf = buf;
    list_add(&buf_node->list, &buf_list->list);
}

EXPORTED_WEAKLY void kava_endpoint_buffer_list_free(struct kava_buffer_list *buf_list)
{
    struct kava_buffer_list *pos, *n;
    list_for_each_entry_safe(pos, n, &buf_list->list, list) {
        list_del(&pos->list);
        kava_buffer_with_deallocator_free(pos->buf);
        vfree(pos);
    }
}
