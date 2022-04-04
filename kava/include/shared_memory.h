#ifndef __KAVA_SHARED_MEMORY_H__
#define __KAVA_SHARED_MEMORY_H__

#ifdef __KERNEL__

#include <linux/ioctl.h>
#include <linux/types.h>
#include <linux/fs.h>

#define KAVA_DEFAULT_SHARED_MEM_SIZE 32
#define BIN_COUNT 20
#define BIN_MAX_IDX (BIN_COUNT - 1)

#define MIN_ALLOC_SIZE 4
#define MIN_NODE_SIZE (sizeof(node_t) + MIN_ALLOC_SIZE + sizeof(footer_t))

typedef struct node_t {
    uint32_t hole;
    uint32_t size;
    struct node_t *next; /* Allocated memory starts from here */
    struct node_t *prev;
} node_t;

typedef struct footer_t {
    node_t *header;
} footer_t;

typedef struct bin_t {
    node_t *head;
} bin_t;

typedef struct allocator_t {
    long start;
    long end;
    size_t size;
    bin_t bins[BIN_COUNT];
    uint32_t is_dma;
} allocator_t;

extern allocator_t *shm_allocator;

void kava_allocator_init(size_t size);
void kava_allocator_fini(void);
void *kava_alloc(size_t size);
void kava_free(void *p);
long kava_shm_offset(const void *p);
int kshm_mmap_helper(struct file *filp, struct vm_area_struct *vma);

#else

#include <sys/ioctl.h>

#endif // __KERNEL

#define KAVA_SHM_GET_SHM_SIZE _IOW(KAVA_SHM_DEV_MAJOR, 0x1, long *)

#endif // __KAVA_SHARED_MEMORY_H__
