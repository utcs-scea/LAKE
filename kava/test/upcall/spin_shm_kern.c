#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <linux/string.h>
#include <asm/uaccess.h>

#include "upcall.h"
#include "upcall_impl.h"

#define SHM_SIZE (MMAP_NUM_PAGES * PAGE_SIZE)

struct upcall_handle {
    struct class *dev_class;
    struct device *dev_node;

    void *shared_mem;
    struct mutex lock;
};

upcall_handle_t handle;

static int spin_shm_open(struct inode *inode, struct file *file) {
    pr_info("Worker is connected\n");
    return 0;
}

static int spin_shm_release(struct inode *inode, struct file *file) {
    pr_info("Worker is disconnected\n");
    return 0;
}

static int spin_shm_mmap(struct file *filp, struct vm_area_struct *vma) {
    phys_addr_t start = virt_to_phys(handle->shared_mem) >> PAGE_SHIFT;
    unsigned long len = vma->vm_end - vma->vm_start;

    if (len != SHM_SIZE)
        return -EINVAL;

    vma->vm_flags |= VM_LOCKED;
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
    return remap_pfn_range(vma, vma->vm_start, start, len, vma->vm_page_prot);
}

static struct file_operations shm_fops = {
    .owner = THIS_MODULE,
	.open = spin_shm_open,
    .mmap = spin_shm_mmap,
    .release = spin_shm_release,
};

upcall_handle_t init_upcall(void) {
    handle = kmalloc(sizeof(struct upcall_handle), GFP_KERNEL);
    memset(handle, 0, sizeof(struct upcall_handle));

    register_chrdev(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_NAME, &shm_fops);
    handle->dev_class = class_create(THIS_MODULE, UPCALL_TEST_DEV_CLASS);
    handle->dev_class->devnode = mod_dev_node;
    handle->dev_node = device_create(handle->dev_class, NULL,
            MKDEV(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_MINOR_NUM),
            NULL, UPCALL_TEST_DEV_NAME);

    handle->shared_mem = kzalloc(SHM_SIZE, GFP_KERNEL);
    mutex_init(&handle->lock);

    return handle;
}

void close_upcall(upcall_handle_t handle) {
    device_destroy(handle->dev_class,
                MKDEV(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_MINOR_NUM));
    class_unregister(handle->dev_class);
    class_destroy(handle->dev_class);
    unregister_chrdev(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_NAME);

    kfree(handle->shared_mem);
    kfree(handle);
}

void _do_upcall(upcall_handle_t handle,
        uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
        void *buf, size_t size)
{
    struct shared_region *shm = (struct shared_region *)handle->shared_mem;
#if PRINT_TIME_K_TO_U
    struct timespec ts;
#endif

    mutex_lock_interruptible(&handle->lock);

    shm->data.r0 = r0;
    shm->data.r1 = r1;
    shm->data.r2 = r2;
    shm->data.r3 = r3;
    shm->data.buf_size = size;

#if PRINT_TIME_K_TO_U
    getnstimeofday(&ts);
    pr_info("Upcall called: sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);
#endif

    shm->doorbell = 1;

    /* Spin on acknowledge */
    while (shm->doorbell);

    mutex_unlock(&handle->lock);
}

static int __init test_upcall_init(void)
{
    create_test_device();
    handle = init_upcall();
    return 0;
}

static void __exit test_upcall_fini(void)
{
    close_upcall(handle);
    close_test_device();
}

module_init(test_upcall_init);
module_exit(test_upcall_fini);

MODULE_AUTHOR("Hangchen Yu, Bodun Hu");
MODULE_DESCRIPTION("Upcall benchmarking module (shm)");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
