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

    struct task_struct *kth;
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

static int shm_fault(struct vm_fault *vmf) {
    vmf->page = vmalloc_to_page(handle->shared_mem + (vmf->pgoff << PAGE_SHIFT));
    get_page(vmf->page);

    return 0;
}

static const struct vm_operations_struct shm_vm_ops = {
    .fault = shm_fault
};

static int spin_shm_mmap(struct file *filp, struct vm_area_struct *vma) {
    unsigned long len = vma->vm_end - vma->vm_start;
    if (len != SHM_SIZE || vma->vm_pgoff != 0)
        return -EINVAL;

    vma->vm_ops = &shm_vm_ops;
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);

    return 0;
}

static struct file_operations shm_fops = {
    .owner = THIS_MODULE,
	.open = spin_shm_open,
    .mmap = spin_shm_mmap,
    .release = spin_shm_release,
};

static int kthread_poll(void *args)
{
    struct upcall_handle* handle = (struct upcall_handle *)args;
    struct shared_region *shm = (struct shared_region *)handle->shared_mem;
    size_t size;
    struct shared_region *buf = (struct shared_region *)vmalloc(SHM_SIZE);

    while (!kthread_should_stop()) {
        while (shm->doorbell_kern == 0 && !kthread_should_stop());
        if (kthread_should_stop()) break;

        mutex_lock_interruptible(&handle->lock);

        size = shm->size;
        memcpy(buf, shm, size);
        shm->doorbell_kern = 0;
        buf->doorbell_kern = 0;
        buf->doorbell_user = 0;
        memcpy(shm, buf, size);
        shm->doorbell_user = 1;

        mutex_unlock(&handle->lock);
    }
    return 0;
}

void _do_upcall(upcall_handle_t handle,
        uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
        void *buf, size_t size)
{
}

upcall_handle_t init_upcall(void) {
    handle = kmalloc(sizeof(struct upcall_handle), GFP_KERNEL);
    memset(handle, 0, sizeof(struct upcall_handle));

    register_chrdev(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_NAME, &shm_fops);
    handle->dev_class = class_create(THIS_MODULE, UPCALL_TEST_DEV_CLASS);
    handle->dev_class->devnode = mod_dev_node;
    handle->dev_node = device_create(handle->dev_class, NULL,
            MKDEV(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_MINOR_NUM),
            NULL, UPCALL_TEST_DEV_NAME);

    handle->shared_mem = vmalloc_user(SHM_SIZE);
    memset(handle->shared_mem, 0, SHM_SIZE);
    mutex_init(&handle->lock);

    /* Create kthread to poll. */
    handle->kth = kthread_run(kthread_poll, (void *)handle, "kava_upcall");
    return handle;
}

void close_upcall(upcall_handle_t handle) {
    kthread_stop(handle->kth);

    device_destroy(handle->dev_class,
                MKDEV(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_MINOR_NUM));
    class_unregister(handle->dev_class);
    class_destroy(handle->dev_class);
    unregister_chrdev(UPCALL_TEST_DEV_MAJOR_NUM, UPCALL_TEST_DEV_NAME);

    vfree(handle->shared_mem);
    kfree(handle);
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

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Upcall throughput benchmarking module (shm)");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
