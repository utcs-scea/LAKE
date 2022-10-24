#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/dma-map-ops.h>
#include <linux/dma-mapping.h>
#include <asm/dma.h>
#include <asm/page.h>
#include <asm/uaccess.h>

#include "lake_shm.h"
#include "mymemory.h"

#define EXPORTED_WEAKLY __attribute__ ((visibility ("default"))) __attribute__ ((weak))

static dma_addr_t dma_handle;
static struct device *dev_node;
static struct class  *dev_class;

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
 */
int kava_allocator_init(size_t size)
{
    void *start;
    node_t *init_region;
    int err;
    u64 dmamask;

    /* Register chardev */
    err = create_chrdevice();
    if (err) 
        return err;

    dmamask = DMA_BIT_MASK(32);
    dev_node->dma_mask = (u64*)&dmamask;
    dev_node->coherent_dma_mask = DMA_BIT_MASK(32);

    start = dma_alloc_coherent(dev_node, size, &dma_handle, GFP_KERNEL);
    if (!start) 
        return -ENOMEM;

    //init mem manager
    mymalloc_init(start, size);
    return 0;
}
EXPORT_SYMBOL(kava_allocator_init);

/**
 * kava_allocator_fini - Free allocated memory region
 */
void kava_allocator_fini(void)
{
    pr_info("[kava-shm] Deallocate shared DMA memory region pa = 0x%lx, va = 0x%lx\n",
            (uintptr_t)virt_to_phys((void *)shm_start),
            (uintptr_t)shm_start);

    /* BUG: dma_free_coherent has segfault in a virtual machine. */
    dma_free_coherent(dev_node, shm_size, (void *)shm_start, dma_handle);

    unregister_chrdev(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_NAME);
    device_destroy(dev_class, MKDEV(KAVA_SHM_DEV_MAJOR, KAVA_SHM_DEV_MINOR));
    //class_unregister(dev_class);
    class_destroy(dev_class);
}
EXPORT_SYMBOL(kava_allocator_fini);

/**
 * kava_alloc - Allocate a memory from shared memory region
 * @size: size of memory to allocate
 *
 * This function returns the allocated memory's kernel virtual address.
 */
void *kava_alloc(size_t size)
{
    return mymalloc(size);
}
EXPORT_SYMBOL(kava_alloc);

/**
 * kava_free - Free a memory allocated by kava_alloc
 * @p: memory allocated by kava_alloc
 */
void kava_free(void *p)
{
    myfree(p);
}
EXPORT_SYMBOL(kava_free);

/**
 * kava_shm_offset - Offset of the address in the shared memory region
 * @p: memory address
 *
 * This function returns -1 if p is not inside the shared memory.
 */
s64 kava_shm_offset(const void *p)
{
    //if (shm_allocator->start <= (long)p && (long)p < shm_allocator->end)
    if ((u64)shm_start <= (u64)p && (u64)p < (u64)shm_end)
        return (u64)p - (u64)shm_start;
    return -1;
}
EXPORT_SYMBOL(kava_shm_offset);

static vm_fault_t va_shm_vm_fault(struct vm_fault *vmf)
{
    //vmf->page = vmalloc_to_page((void *)shm_allocator->start + (vmf->pgoff << PAGE_SHIFT));
    vmf->page = vmalloc_to_page((void *)shm_start + (vmf->pgoff << PAGE_SHIFT));
    get_page(vmf->page);
    return 0;
}

static const struct vm_operations_struct va_shm_vm_ops = {
    .fault = va_shm_vm_fault,
};

EXPORTED_WEAKLY int kshm_mmap_helper(struct file *filp, struct vm_area_struct *vma)
{
    u64 rsize = vma->vm_end - vma->vm_start;
    if (rsize != shm_size || vma->vm_pgoff != 0) {
        pr_err("[kava] Error: shared memory size does not match "
            "%llu != %llu  or  %lu != 0\n", rsize, shm_size,
            vma->vm_pgoff);
        return -EINVAL;
    }

    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    return dma_mmap_coherent(dev_node, vma, shm_start, dma_handle, shm_size);
    // if (shm_allocator->is_dma) {
    //     vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    //     return dma_mmap_coherent(dev_node, vma, (void *)shm_allocator->start, dma_handle, shm_allocator->size);
    // }
    // else {
    //     vma->vm_ops = &va_shm_vm_ops;
    // }
	return 0;
}
