#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include <linux/sched.h>

#include "mllb_comon.h"
#include "mllb_cpu.h"

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Kernel module of a mllb program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");

/*
 *  Test: run both linux and MLLB cpu
 */
__attribute__ ((unused))
static int run_both_rebalance(void) {
    return 0;
}

/*
 *  Test: just set the balancer to the default balancer
 */
__attribute__ ((unused))
static int run_test_default_rebalance(void) {
    int i;
    int n = 10;

    printk(KERN_INFO  "> Registering default fn\n");
    hack_mllb_register_fn(can_migrate_task_linux);

    printk(KERN_INFO  "> loopin\n");
    for (i=0 ; i < n ; i++) {
        msleep(1000);
    }

    printk(KERN_INFO  "> Unregistering default fn\n");
    hack_mllb_unregister_fn();
    msleep(5);

    return 0;
}

/*
 *  Test: set a test fn to be called by the balancer code
 */
__attribute__ ((unused))
static int run_test_register(void) {
    int i;
    int n = 10;

    printk(KERN_INFO  "> Registering fn\n");
    hack_mllb_register_test_fn(mllb_ping);

    printk(KERN_INFO  "> loopin\n");
    for (i=0 ; i < n ; i++) {
        msleep(1000);
    }

    printk(KERN_INFO  "> Unregistering fn\n");
    hack_mllb_unregister_test_fn();
    msleep(5);

    return 0;
}

static int __init mllb_init(void) {
    return run_test_default_rebalance();
	//return run_test_register();
}

static void __exit mllb_fini(void) {
    //cleanup
}

module_init(mllb_init);
module_exit(mllb_fini);