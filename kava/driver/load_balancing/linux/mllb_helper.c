#include "mllb_helper.h"
#include <linux/module.h>  /* Needed by all modules */
#include <linux/kernel.h>  /* Needed for KERN_ALERT */

MODULE_LICENSE("GPL");

void mllb_ping(void) {
    printk(KERN_CRIT "  ! MLLB ping!\n");
}
EXPORT_SYMBOL(mllb_ping);