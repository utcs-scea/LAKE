#ifndef __KAVA_ENDPOINT_H__
#define __KAVA_ENDPOINT_H__

#ifndef __KERNEL__

#include <glib.h>
#include <pthread.h>
#include <stdint.h>

#ifdef __cplusplus
#include <atomic>
using namespace std;
#else
#include <stdatomic.h>
#endif

#else

#include <linux/kthread.h>
#include <linux/mutex.h>
#include <linux/types.h>

#endif

#include "util.h"

typedef void *(*kava_allocator)(size_t size);
#ifndef __KERNEL__
typedef void (*kava_deallocator)(void *ptr);
#else
typedef void (*kava_deallocator)(const void *ptr);
#endif

#ifndef __KERNEL__
extern GPtrArray *__kava_alloc_list_async_in;
extern GHashTable *__kava_alloc_list_async_out;
#endif

struct kava_endpoint;

#include "metadata_map.h"

struct kava_endpoint {
    size_t metadata_size;
#ifndef __KERNEL__
    GHashTable *managed_buffer_map;
    GHashTable *managed_by_coupled_map;
    pthread_mutex_t managed_buffer_map_mutex;
    // GHashTable* metadata_map;
    // pthread_mutex_t metadata_map_mutex = PTHREAD_MUTEX_INITIALIZER;
    GHashTable *call_map;
    pthread_mutex_t call_map_mutex;
#else
    struct rhashtable *managed_buffer_map;
    struct rhashtable *managed_by_coupled_map;
    struct mutex managed_buffer_map_mutex;
    // struct hlist_head *metadata_map;
    // struct mutex metadata_map_mutex = PTHREAD_MUTEX_INITIALIZER;
    struct rhashtable *call_map;
    struct mutex call_map_mutex;
#endif
    volatile intptr_t call_counter;
};
// TODO: remove unneeded fields.

/* Sentinel to tell worker there is a buffer to return data into. */
#define HAS_OUT_BUFFER_SENTINEL ((void*)1)

#ifdef __KERNEL__
void init_endpoint_lib(void);
#endif

void kava_endpoint_init(struct kava_endpoint *endpoint,
                        size_t metadata_size);

void kava_endpoint_destroy(struct kava_endpoint *endpoint);

intptr_t kava_get_call_id(struct kava_endpoint *endpoint);
void kava_add_call(struct kava_endpoint* endpoint, intptr_t id, void* ptr);
void *kava_remove_call(struct kava_endpoint* endpoint, intptr_t id);

struct kava_buffer_with_deallocator;
struct kava_buffer_list;

/**
 * kava_buffer_with_deallocator_new - Create a metadata to store the buffer and
 * its deallocator
 * @deallocator: the deallocator of the buffer
 * @buffer: the buffer
 */
#ifdef __KERNEL__
struct kava_buffer_with_deallocator *kava_buffer_with_deallocator_new(
        void (*deallocator)(const void *), void *buffer);
#else
struct kava_buffer_with_deallocator *kava_buffer_with_deallocator_new(
        void (*deallocator)(void *), void *buffer);
#endif

/**
 * kava_buffer_with_deallocator_free - Free an endpoint buffer
 * @buffer: the buffer to be freed
 */
void kava_buffer_with_deallocator_free(struct kava_buffer_with_deallocator *buffer);

/**
 * kava_endpoint_buffer_list_new - Create a list for endpoint buffers
 */
struct kava_buffer_list *kava_endpoint_buffer_list_new(void);

/**
 * kava_endpoint_buffer_list_add - Add a new buffer to the endpoint buffer list
 * @buf_list: the endpoint buffer list
 * @buf: the buf to be inserted into the list
 */
void kava_endpoint_buffer_list_add(struct kava_buffer_list *buf_list,
                                struct kava_buffer_with_deallocator *buf);

/**
 * kava_endpoint_buffer_list_free - Free the endpoint buffer list
 * @buf_list: the endpoint buffer list to be freed
 *
 * This function must free the buffer list itself as well.
 */
void kava_endpoint_buffer_list_free(struct kava_buffer_list *buf_list);

#endif
