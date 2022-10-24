#include <linux/sched/signal.h>
#include <linux/slab.h>
#include <linux/time.h>
#include <linux/ktime.h>
#include "hello.h"
#include <linux/delay.h>

//#include <linux/bpf.h>
//#include <linux/btf_ids.h>

static int run_hello(void)
{
	int i, j;
    int *ptr[100];

	for(i = 0; i < 100; i++) {
		ptr[i] = (int *)kava_alloc(100*sizeof(int));
		if(ptr[i] == 0) {
			pr_warn("err on alloc\n");
			return -1;
		}
		for (j = 0; j < 100 ; j++)
			ptr[i][j] = i;
	}
	
	for(i = 0; i < 100; i++) {
		kava_free(ptr[i]);
	}
	
	// pr_warn("init");
	// int *ptr = kava_alloc(100*sizeof(int));
	// pr_warn("alloc");
	// if(ptr == 0) {
	// 	pr_warn("err on alloc\n");
	// 	return -1;
	// }
	// kava_free(ptr);

	return 0;
}


/**
 * Program main
 */
static int __init hello_init(void)
{
	return run_hello();
}

static void __exit hello_fini(void)
{
}

module_init(hello_init);
module_exit(hello_fini);

MODULE_AUTHOR("Henrique Fingler");
MODULE_DESCRIPTION("Example kernel module of using CUDA in lake");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
