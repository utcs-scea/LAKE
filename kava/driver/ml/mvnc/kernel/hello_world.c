#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

#include "mvnc_nw.h"

#define description_string "Kernel implementation of hello world in mvnc."
#define maintainer_string "Bodun Hu"

MODULE_AUTHOR(maintainer_string);
MODULE_DESCRIPTION(description_string);
MODULE_VERSION("0.01");
MODULE_LICENSE("GPL");

static int __init mvnc_hello_world_init(void) {
    printk(KERN_INFO "Start mvcs hello world.\n");
    struct ncDeviceHandle_t *deviceHandle;
    int loglevel = 2;
    ncStatus_t retCode = ncGlobalSetOption(NC_RW_LOG_LEVEL, &loglevel, sizeof(loglevel));

    printk(KERN_INFO "Finished ncGlobalSetOption.\n");
    // Init device handle
    retCode = ncDeviceCreate(0, &deviceHandle);
    if (retCode != NC_OK) {
        printk(KERN_INFO "Error - No NCS devices found.\n");
        printk(KERN_INFO "    ncStatus value: %d\n", retCode);
        return retCode;
    }

    // Open Device
    retCode = ncDeviceOpen(deviceHandle);
    if (retCode != NC_OK) {
        printk(KERN_INFO "Error - ncDeviceOpen failed\n");
        printk(KERN_INFO "    ncStatus value: %d\n", retCode);
        return retCode;
    }

    printk("Hello NCS ! Device opened normally.\n");

    retCode = ncDeviceClose(deviceHandle);
    deviceHandle = NULL;
    if (retCode != NC_OK) {
        printk(KERN_INFO "Error - could not clse NCS device.\n");
        printk(KERN_INFO "    ncStatus value: %d\n", retCode);
        return retCode;
    }

    printk(KERN_INFO "NCS device working.\n");

    return 0;
}

static void __exit mvnc_hello_world_exit(void) {
    printk(KERN_INFO "MVNC hello world finished.\n");
}

module_init(mvnc_hello_world_init);
module_exit(mvnc_hello_world_exit);
