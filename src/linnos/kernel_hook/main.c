#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/blkdev.h>

#include "sde.h"

const char *devices[] = {
    "/dev/vda1",
	0
};

static long *weights[][4] = {
	{weight_0_T_sde, weight_1_T_sde, bias_0_sde, bias_1_sde}
};

static void test(void) {
	struct block_device *dev;
	struct request_queue *q;
	pr_warn("trying to get by path\n");
	dev = blkdev_get_by_path("/dev/vda", FMODE_READ|FMODE_WRITE, THIS_MODULE);
	if (IS_ERR(dev)) {
		pr_warn("didnt work, err %ld\n", PTR_ERR(dev));
		return;
	}
	pr_warn("worked! disk name: %s\n", dev->bd_disk->disk_name);
	q = bdev_get_queue(dev);
	pr_warn("is queue ml enabled? %d\n", q->ml_enabled);
}

static void attach_to_queue(int idx) {
	struct block_device *dev;
	struct request_queue *q;
	long **wts = weights[idx];

	pr_warn("Attaching to queue on %s\n", devices[idx]);
	dev = blkdev_get_by_path(devices[idx], FMODE_READ|FMODE_WRITE, THIS_MODULE);
	q = bdev_get_queue(dev);

	pr_warn("is queue ml enabled? %d\n", q->ml_enabled);
	pr_warn("wt test  %ld %ld %ld %ld \n", wts[0][0], wts[1][0], wts[2][0], wts[3][0]);
	pr_warn("test done\n");
}

static int run_hook(void)
{
	long *wts;
	const char *devs;
	int i;

	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]) {
		pr_warn("dev %i  %p\n", i, devs);
		attach_to_queue(i);
	}

	return 0;
}

/**
 * Program main
 */
static int __init hook_init(void)
{
	return run_hook();
}

static void __exit hook_fini(void)
{
}

module_init(hook_init);
module_exit(hook_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("kernel hook for linnos");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
