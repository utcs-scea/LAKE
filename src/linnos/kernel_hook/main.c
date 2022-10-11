#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include <linux/delay.h>
#include <linux/blkdev.h>

#include "predictors.h"

#define SET_SYSCTL_DEBUG 1

extern unsigned long sysctl_lake_enable_linnos;
extern unsigned long sysctl_lake_linnos_debug;

//adding a model to a device requires:
// 1. include the header with the weights
// 2. put device name in devices
// 3. set the pointers into a new array in weights (dont mess with the ending 0)

#include "sde.h"
#include "weights_header/w_nvme0n1.h"
#include "weights_header/w_nvme1n1.h"
#include "weights_header/w_nvme2n1.h"

const char *devices[] = {
    //"/dev/vdb",
	"/dev/nvme0n1p1",
	"/dev/nvme1n1p1",
	"/dev/nvme2n1p1",
	0
};

static long *weights[][4] = {
	//{weight_0_T_sde, weight_1_T_sde, bias_0_sde, bias_1_sde}
	{weight_0_T_nvme0n1, weight_1_T_nvme0n1, bias_0_nvme0n1, bias_1_nvme0n1},
	{weight_0_T_nvme1n1, weight_1_T_nvme1n1, bias_0_nvme1n1, bias_1_nvme1n1},
	{weight_0_T_nvme2n1, weight_1_T_nvme2n1, bias_0_nvme2n1, bias_1_nvme2n1},
};

// static void test(void) {
// 	struct block_device *dev;
// 	struct request_queue *q;
// 	pr_warn("trying to get by path\n");
// 	dev = blkdev_get_by_path("/dev/vda", FMODE_READ|FMODE_WRITE, THIS_MODULE);
// 	if (IS_ERR(dev)) {
// 		pr_warn("didnt work, err %ld\n", PTR_ERR(dev));
// 		return;
// 	}
// 	pr_warn("worked! disk name: %s\n", dev->bd_disk->disk_name);
// 	q = bdev_get_queue(dev);
// 	pr_warn("is queue ml enabled? %d\n", q->ml_enabled);
// }

static int attach_to_queue(int idx) {
	struct block_device *dev;
	struct request_queue *q;
	long **wts = weights[idx];

	pr_warn("Attaching to queue on %s\n", devices[idx]);
	dev = blkdev_get_by_path(devices[idx], FMODE_READ|FMODE_WRITE, THIS_MODULE);
	if(IS_ERR(dev)) {
		pr_warn("Error getting dev by path (%s): %ld\n", devices[idx], PTR_ERR(dev));
		return -1;
	}
	q = bdev_get_queue(dev);
	pr_warn("wt test  %ld %ld %ld %ld \n", wts[0][0], wts[1][0], wts[2][0], wts[3][0]);

	q->weight_0_T = wts[0];
	q->weight_1_T = wts[1];
	q->bias_0 = wts[1];
	q->bias_1 = wts[2];
	q->predictor = cpu_prediction_model;
	q->ml_enabled = true;
	sysctl_lake_enable_linnos = true;
	pr_warn("Attached!\n");
	return 0;
}

static int dettach_queue(int idx) {
	struct block_device *dev;
	struct request_queue *q;

	pr_warn("Dettaching queue on %s\n", devices[idx]);
	dev = blkdev_get_by_path(devices[idx], FMODE_READ|FMODE_WRITE, THIS_MODULE);
	if(IS_ERR(dev)) {
		pr_warn("Error getting dev by path (%s): %ld\n", devices[idx], PTR_ERR(dev));
		return -1;
	}
	q = bdev_get_queue(dev);

	q->ml_enabled = false;
	sysctl_lake_enable_linnos = false;
	usleep_range(100,200);
	q->predictor = 0;
	q->weight_0_T = 0;
	q->weight_1_T = 0;
	q->bias_0 = 0;
	q->bias_1 = 0;
	pr_warn("Dettached!\n");
	return 0;
}

static int run_hook(void)
{
	const char *devs;
	int i, err;

	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]) {
		err = attach_to_queue(i);
		if (err) return err;
	}

	//sysctl_lake_linnos_debug = 1;

	return 0;
}

/**
 * Program main
 */
static int __init hook_init(void)
{
	sysctl_lake_linnos_debug = SET_SYSCTL_DEBUG;
	return run_hook();
}

static void __exit hook_fini(void)
{
	const char *devs;
	int i, err;
	for(devs = devices[0], i=0 ; devs != 0 ; devs = devices[++i]){
		err = dettach_queue(i);
		if (err) return;
	}
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
