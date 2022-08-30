/*******************************************************************************

  Kernel-space shared memory driver.

*******************************************************************************/

#include <linux/module.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "lake_shm.h"

static long shm_size = KAVA_DEFAULT_SHARED_MEM_SIZE;
module_param(shm_size, long, 0444);
MODULE_PARM_DESC(shm_size, "Shared memory size in MB, default 32 MB");

static int run_test = 0;
module_param(run_test, int, 0444);
MODULE_PARM_DESC(run_test, "Run example tests");

static int __init kshm_init(void)
{
    int err = -ENOMEM;
    /* Initialize allocator */
    err = kava_allocator_init((shm_size << 20));

    if (!err) return 0;
    else {
        pr_err("kava_allocator_init error\n");
        return -1;
    }
}

static void __exit kshm_fini(void)
{
    kava_allocator_fini();
}

module_init(kshm_init);
module_exit(kshm_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("KAvA shared memory driver");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
