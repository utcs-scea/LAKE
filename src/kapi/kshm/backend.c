#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/dma-map-ops.h>
#include <linux/dma-mapping.h>
#include <asm/dma.h>
#include <asm/page.h>
#include <asm/uaccess.h>

#include "lake_shm.h"

#define EXPORTED_WEAKLY __attribute__ ((visibility ("default"))) __attribute__ ((weak))

allocator_t *shm_allocator = NULL;
static dma_addr_t dma_handle;
static size_t overhead = sizeof(node_t) + sizeof(footer_t);
static size_t offset = offsetof(node_t, next);
static struct device *dev_node;
static struct class  *dev_class;
static long shm_size;

static void add_node(bin_t *bin, node_t *node)
{
    node_t *temp = bin->head;
    node_t *cur, *prev;

    node->prev = node->next = NULL;

    if (temp == NULL) {
        bin->head = node;
        return;
    }

    /* Put larger nodes in the front */
    cur = bin->head;
    prev = NULL;
    while (cur && cur->size < node->size) {
        prev= cur;
        cur = cur->next;
    }

    if (cur == NULL) {
        /* Reach the end of the list */
        prev->next = node;
        node->prev = prev;
    }
    else {
        if (prev == NULL) {
            /* Need to insert at head */
            node->next = bin->head;
            bin->head->prev = node;
            bin->head = node;
        }
        else {
            /* Insert node in the middle of the list */
            node->next = cur;
            cur->prev = node;
            prev->next = node;
            node->prev = prev;
        }
    }

}

static void remove_node(bin_t *bin, node_t *node)
{
    node_t *temp;

    if (bin->head == NULL)
        return;

    if (bin->head == node) {
        bin->head = node->next;;
        return;
    }

    temp = bin->head->next;
    while (temp) {
        if (temp == node) {
            if (temp->next == NULL)
                temp->prev->next = NULL;
            else {
                temp->prev->next = node->next;
                temp->next->prev = node->prev;
            }
            return;
        }

        temp = temp->next;
    }
}

static node_t *get_best_fit(bin_t *bin, size_t size)
{
    node_t *temp = bin->head;

    if (temp == NULL)
        return NULL;

    while (temp != NULL) {
        if (temp->size >= size) {
            return temp;
        }
        temp = temp->next;
    }

    return NULL;
}

static footer_t *get_foot(node_t *node)
{
    return (footer_t *)((char *)node + sizeof(node_t) + node->size);
}

static void create_foot(node_t *head)
{
    footer_t *foot = get_foot(head);
    foot->header = head;
}

/*
 * The regions in the first bucket are smaller than 8 bytes.
 */
static unsigned get_bin_index(size_t size)
{
    unsigned index = -2;
    while (size >>= 1) index++;
    if (index < 0) index = 0;
    if (index > BIN_MAX_IDX) index = BIN_MAX_IDX;
    return index;
}

static char *mod_dev_node(struct device *dev, umode_t *mode)
{
    if (mode)
        *mode = 0444;
    return NULL;
}

static int kshm_open(struct inode *inode, struct file *filp)
{
    return 0;
}

static int kshm_release(struct inode *inode, struct file *filp)
{
    return 0;
}

static long kshm_ioctl(struct file *filp, unsigned int cmd,
                       unsigned long arg)
{
    int r = -EINVAL;
    long size_in_bytes;

    switch (cmd)
    {
    case KAVA_SHM_GET_SHM_SIZE:
        size_in_bytes = shm_size;
        copy_to_user((void *)arg, (void *)&size_in_bytes, sizeof(long));
        r = 0;
        break;

    default:
        pr_err("[kava-shm] Unsupported IOCTL command\n");
    }

    return r;
}

static const struct file_operations fops =
{
    .owner          = THIS_MODULE,
    .open           = kshm_open,
    .mmap           = kshm_mmap_helper,
    .release        = kshm_release,
    .unlocked_ioctl = kshm_ioctl,
};

static int create_chrdevice(void) {
    register_chrdev(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_NAME, &fops);
    pr_info("[kava-shm] Registered shared memory device with major number %d\n", KAVA_SHM_DEV_MAJOR);

    if (!(dev_class = class_create(THIS_MODULE, KAVA_SHM_DEV_CLASS))) {
        pr_err("[kava-shm] Class_create error\n");
        goto unregister_dev;
    }
    dev_class->devnode = mod_dev_node;

    device_destroy(dev_class, MKDEV(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_MINOR));

    if (!(dev_node = device_create(dev_class, NULL,
                    MKDEV(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_MINOR), NULL, KAVA_SHM_DEV_NAME))) {
        pr_err("[kava-shm] Device_create error\n");
        goto destroy_class;
    }
    pr_info("[kava-shm] Create shared memory device\n");
    return 0;

    destroy_class:
    class_unregister(dev_class);
    class_destroy(dev_class);

unregister_dev:
    unregister_chrdev(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_NAME);
    return -1;
}

/**
 * kava_allocator_init - Initialize shared memory allocator
 * @size: size of managed shared memory
 *
 * This function initializes the static shm_allocator.
 */
int kava_allocator_init(size_t size)
{
    void *start;
    node_t *init_region;
    int err;
    u64 dmamask;

    shm_size = size;

    /* Register chardev */
    err = create_chrdevice();
    if (err) 
        return err;

    /* Create allocator */
    shm_allocator = (allocator_t *)kmalloc(sizeof(allocator_t), GFP_USER);
    memset(shm_allocator, 0, sizeof(allocator_t));

    if (size < overhead) {
        size = PAGE_SIZE;
        pr_info("[kava-shm] Round up shared memory size to %ld bytes\n", size);
    }

    /* Allocate memory */
    // pr_info("[kava-shm] Executing dma_alloc_coherent\n");
    // err = dma_set_mask_and_coherent(dev_node, DMA_BIT_MASK(32));
    // if (err) {
    //     pr_err("dma_set_mask returned: %d\n", err);
    //     return -EIO;
    // }
    dmamask = DMA_BIT_MASK(32);
    dev_node->dma_mask = (u64*)&dmamask;
    dev_node->coherent_dma_mask = DMA_BIT_MASK(32);

    start = dma_alloc_coherent(dev_node, size, &dma_handle, GFP_KERNEL);
    if (start) {
        shm_allocator->is_dma = 1;
    }
    else {
        pr_err("[kava-shm] Failed to allocate shared memory\n");
        kfree(shm_allocator);
        return -ENOMEM;
    }

    pr_info("[kava-shm] Allocate shared %smemory region pa = 0x%lx, va = 0x%lx\n",
            (shm_allocator->is_dma ? "DMA " : ""),
            (uintptr_t)virt_to_phys(start), (uintptr_t)start);
    shm_allocator->start = (long)start;
    shm_allocator->size = size;
    shm_allocator->end = (long)start + size;

    /* Initialize header */
    init_region = (node_t *)start;
    init_region->hole = 1;
    init_region->size = size - overhead;

    /* Initialize footer */
    create_foot(init_region);

    /* Add initial region to buckets */
    add_node(&shm_allocator->bins[get_bin_index(init_region->size)], init_region);

    return 0;
}
EXPORT_SYMBOL(kava_allocator_init);

/**
 * kava_allocator_fini - Free allocated memory region
 */
void kava_allocator_fini(void)
{
    if (shm_allocator) {
        pr_info("[kava-shm] Deallocate shared %smemory region pa = 0x%lx, va = 0x%lx\n",
                (shm_allocator->is_dma ? "DMA " : ""),
                (uintptr_t)virt_to_phys((void *)shm_allocator->start),
                (uintptr_t)shm_allocator->start);

        if (shm_allocator->is_dma) {
            /* BUG: dma_free_coherent has segfault in a virtual machine. */
            dma_free_coherent(dev_node, shm_allocator->size, (void *)shm_allocator->start, dma_handle);
        }
        //else
        //    vfree((void *)shm_allocator->start);
        kfree(shm_allocator);
    }

    unregister_chrdev(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_NAME);
    device_destroy(dev_class, MKDEV(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_MINOR));
    //class_unregister(dev_class);
    class_destroy(dev_class);
}
EXPORT_SYMBOL(kava_allocator_fini);

void *__kava_alloc(allocator_t *allocator, size_t size)
{
    unsigned index = get_bin_index(size);
    bin_t *bin = &allocator->bins[index];
    node_t *found = get_best_fit(bin, size);

    while (found == NULL) {
        if (index >= BIN_MAX_IDX)
            return NULL;

        bin = &allocator->bins[++index];
        found = get_best_fit(bin, size);
    }

    /* Split memory region if the left space can be a new region */
    if (found->size - size > MIN_NODE_SIZE) {
        unsigned new_idx;
        node_t *split = (node_t *)((char *)found + overhead + size);
        split->size = found->size - size - overhead;
        split->hole = 1;
        create_foot(split);

        new_idx = get_bin_index(split->size);
        add_node(&allocator->bins[new_idx], split);

        found->size = size;
        create_foot(found);
    }

    found->hole = 0;
    remove_node(&allocator->bins[index], found);

    // TODO: expend memory region if it is running out of space

    found->prev = NULL;
    found->next = NULL;
    return &found->next;
}

/**
 * kava_alloc - Allocate a memory from shared memory region
 * @size: size of memory to allocate
 *
 * This function returns the allocated memory's kernel virtual address.
 */
void *kava_alloc(size_t size)
{
    return __kava_alloc(shm_allocator, size);
}
EXPORT_SYMBOL(kava_alloc);

void __kava_free(allocator_t *allocator, void *p)
{
    node_t *head = (node_t *)((char *)p - offset);
    node_t *prev, *next;
    bin_t *list;
    footer_t *new_foot, *old_foot;

    if (head == (node_t *)(uintptr_t)allocator->start) {
        head->hole = 1;
        add_node(&allocator->bins[get_bin_index(head->size)], head);
        return;
    }

    next = (node_t *)((char *)get_foot(head) + sizeof(footer_t));
    prev = ((footer_t *)((char *)head - sizeof(footer_t)))->header;

    /* Merge previous hole */
    if (prev->hole) {
        list = &allocator->bins[get_bin_index(prev->size)];
        remove_node(list, prev);

        prev->size += overhead + head->size;
        new_foot = get_foot(head);
        new_foot->header = prev;

        head = prev;
    }

    /* Merge next hole */
    if (next->hole) {
        list = &allocator->bins[get_bin_index(next->size)];
        remove_node(list, next);

        old_foot = get_foot(head);
        head->size += overhead + next->size;
        old_foot->header = NULL;
        next->size = 0;
        next->hole = 0;

        new_foot = get_foot(head);
        new_foot->header = head;
    }

    head->hole = 1;
    add_node(&allocator->bins[get_bin_index(head->size)], head);
}

/**
 * kava_free - Free a memory allocated by kava_alloc
 * @p: memory allocated by kava_alloc
 */
void kava_free(void *p)
{
    __kava_free(shm_allocator, p);
}
EXPORT_SYMBOL(kava_free);

/**
 * kava_shm_offset - Offset of the address in the shared memory region
 * @p: memory address
 *
 * This function returns -1 if p is not inside the shared memory.
 */
long kava_shm_offset(const void *p)
{
    if (shm_allocator->start <= (long)p && (long)p < shm_allocator->end)
        return (long)p - shm_allocator->start;
    return -1;
}
EXPORT_SYMBOL(kava_shm_offset);

static vm_fault_t va_shm_vm_fault(struct vm_fault *vmf)
{
    vmf->page = vmalloc_to_page((void *)shm_allocator->start + (vmf->pgoff << PAGE_SHIFT));
    get_page(vmf->page);
    return 0;
}

static const struct vm_operations_struct va_shm_vm_ops = {
    .fault = va_shm_vm_fault,
};

EXPORTED_WEAKLY int kshm_mmap_helper(struct file *filp, struct vm_area_struct *vma)
{
    u64 rsize = vma->vm_end - vma->vm_start;
    if (rsize != shm_allocator->size || vma->vm_pgoff != 0) {
        pr_err("[kava] Error: shared memory size does not match "
            "%llu != %lu  or  %lu != 0\n", rsize, shm_allocator->size,
            vma->vm_pgoff);
        return -EINVAL;
    }

    pr_info("[kava] Map shared memory length = 0x%lx, offset = 0x%lx\n",
            shm_allocator->size, vma->vm_pgoff);

    if (shm_allocator->is_dma) {
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
        return dma_mmap_coherent(dev_node, vma, (void *)shm_allocator->start, dma_handle, shm_allocator->size);
    }
    else {
        vma->vm_ops = &va_shm_vm_ops;
    }
	return 0;
}
