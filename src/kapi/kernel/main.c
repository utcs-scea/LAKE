#include <linux/module.h>
#include <linux/delay.h>
#include "lake_kapi.h"
#include "kargs.h"
#include "cuda.h"

static int __init lake_kapi_init(void)
{
    int err;
	err = lake_init_socket();
    init_kargs_kv();
    if (err < 0) {
		printk(KERN_ERR "Err in init_socket %d\n", err);
        return -1;
	}

    pr_info("[lake] Registered CUDA kapi\n");
    
    return 0;
}

static void __exit lake_kapi_exit(void)
{
    destroy_kargs_kv();
	lake_destroy_socket();
}

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("LAKE CUDA kapi");
MODULE_LICENSE("GPL");
module_init(lake_kapi_init)
module_exit(lake_kapi_exit)
