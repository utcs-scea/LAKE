/*******************************************************************************

  Test Linux driver module for testing channel performance.

*******************************************************************************/

#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include "cuda_kava.h"
#include "shared_memory.h"

#define data_size 128 // must larger than 128
#define num_test 1000

static int __init drv_init(void)
{
    struct timespec ts_s, ts_e;
    int i;
    uint64_t total_time;

    pr_info("load cuda demo driver\n");

    getnstimeofday(&ts_s);
    for (i = 0; i < num_test; i++) {
        cuTestChannel(data_size);
    }
    getnstimeofday(&ts_e);

    total_time = (ts_e.tv_sec - ts_s.tv_sec) * 1000000000 + (ts_e.tv_nsec - ts_s.tv_nsec);
    pr_info("Total time: %llu nsec, throughput: %llu nsec/cmd\n",
            total_time, total_time / num_test);

    return 0;
}

static void __exit drv_fini(void)
{
    pr_info("unload cuda demo driver\n");
}

module_init(drv_init);
module_exit(drv_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Test driver module for testing channel performance");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
