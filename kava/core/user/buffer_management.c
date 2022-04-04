struct ava_buffer_with_deallocator {
    void (*deallocator)(void*);
    void *buffer;
};

struct ava_buffer_with_deallocator *ava_buffer_with_deallocator_new(void (*deallocator)(void *), void *buffer)
{
    struct ava_buffer_with_deallocator *ret = malloc(sizeof(struct ava_buffer_with_deallocator));
    ret->buffer = buffer;
    ret->deallocator = deallocator;
    return ret;
}

void ava_buffer_with_deallocator_free(struct ava_buffer_with_deallocator *buffer)
{
    buffer->deallocator(buffer->buffer);
    free(buffer);
}



void *ava_cached_alloc(struct ava_endpoint *endpoint, int call_id, const void *coupled, size_t size)
{
    pthread_mutex_lock(&endpoint->managed_buffer_map_mutex);
    struct call_id_and_handle_t key = { call_id, coupled };
    GArray *buffer = (GArray *) g_hash_table_lookup(endpoint->managed_buffer_map, &key);
    if (buffer == NULL) {
        buffer = g_array_sized_new(FALSE, TRUE, 1, size);
        struct call_id_and_handle_t *pkey = (struct call_id_and_handle_t *)malloc(sizeof(struct call_id_and_handle_t));
        *pkey = key;
        g_hash_table_insert(endpoint->managed_buffer_map, pkey, buffer);
        struct ava_coupled_record_t *rec = ava_get_coupled_record_unlocked(endpoint, coupled);
        g_ptr_array_add(rec->key_list, pkey);
        g_ptr_array_add(rec->buffer_list, buffer);
    }
    // TODO: This will probably never shrink the buffer. We may need to implement that for large changes.
    g_array_set_size(buffer, size);
    pthread_mutex_unlock(&endpoint->managed_buffer_map_mutex);
    return buffer->data;
}

void *ava_uncached_alloc(struct ava_endpoint *endpoint, const void *coupled, size_t size)
{
    pthread_mutex_lock(&endpoint->managed_buffer_map_mutex);
    GArray *buffer = g_array_sized_new(FALSE, TRUE, 1, size);
    struct ava_coupled_record_t *rec = ava_get_coupled_record_unlocked(endpoint, coupled);
    g_ptr_array_add(rec->buffer_list, buffer);
    pthread_mutex_unlock(&endpoint->managed_buffer_map_mutex);
    return buffer->data;
}

void *ava_static_alloc(struct ava_endpoint *endpoint, int call_id, size_t size)
{
    return ava_cached_alloc(endpoint, call_id, NULL, size);
}
