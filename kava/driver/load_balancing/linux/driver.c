//#include "helpers.h"
//#include "consts.h"
#include <linux/module.h>  /* Needed by all modules */
#include <linux/kernel.h>  /* Needed for KERN_ALERT */
#include <linux/delay.h>
#include <linux/ktime.h>
#include "mllb_helper.h"

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Kernel module of a mllb program in kava");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");

extern void hack_mllb_register_fn(void (*fn_ptr)(void));
extern void hack_mllb_unregister_fn(void);

static int run_gpu(void) {
    int i;
    int n = 10;

    printk(KERN_INFO  "> Registering fn\n");
    hack_mllb_register_fn(mllb_ping);

    printk(KERN_INFO  "> loopin\n");
    for (i=0 ; i < n ; i++) {
        msleep(1000);
    }

    printk(KERN_INFO  "> Unregistering fn\n");
    hack_mllb_unregister_fn();
    msleep(5);

    return 0;
}

static int __init mllb_init(void)
{
	return run_gpu();
}

static void __exit mllb_fini(void)
{
    //cleanup
}

module_init(mllb_init);
module_exit(mllb_fini);