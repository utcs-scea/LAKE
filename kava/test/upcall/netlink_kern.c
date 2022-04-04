#define pr_fmt(fmt) "%s:%d:: " fmt, __func__, __LINE__
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/random.h>
#include <linux/time.h>
#include <linux/uaccess.h>
#include <asm/uaccess.h>

#include <net/sock.h>
#include <linux/netlink.h>
#include <linux/skbuff.h>

#include "upcall_impl.h"

struct upcall_handle {
    struct sock *nl_sk;
    struct sk_buff *skb_out;
    int pid;
};

static DECLARE_WAIT_QUEUE_HEAD(__is_connected);
static volatile int is_connected = 0;

upcall_handle_t handle;

static void nl_recvmsg_handler(struct sk_buff *skb)
{
    struct nlmsghdr *nlh;

    if (is_connected)
        return;

    nlh = (struct nlmsghdr *)skb->data;
    handle->pid = nlh->nlmsg_pid;
    printk(KERN_INFO "Register user process PID = %d\n", handle->pid);
    is_connected = 1;
    wake_up_interruptible(&__is_connected);
}

struct netlink_kernel_cfg nl_cfg = {
    .input = nl_recvmsg_handler,
};

upcall_handle_t init_upcall(void)
{
    handle = kmalloc(sizeof(struct upcall_handle), GFP_KERNEL);
    memset(handle, 0, sizeof(struct upcall_handle));

    handle->skb_out = NULL;
    handle->nl_sk = netlink_kernel_create(&init_net, NETLINK_USER, &nl_cfg);

    if (!is_connected) {
        wait_event_interruptible(__is_connected, is_connected == 1);
    }

    return handle;
}

void close_upcall(upcall_handle_t handle)
{
    netlink_kernel_release(handle->nl_sk);
    kfree(handle);
}

void _do_upcall(upcall_handle_t handle,
                uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
                void *buf, size_t size)
{
    struct nlmsghdr *nlh;
    struct base_buffer base_buf = {
        .r0 = r0, .r1 = r1, .r2 = r2, .r3 = r3,
        .buf_size = size,
    };
    int ret;
#if PRINT_TIME_K_TO_U
    struct timespec ts;
#endif

    handle->skb_out = nlmsg_new(sizeof(struct base_buffer), 0);
    BUG_ON(!handle->skb_out && "Failed to allocate new skb\n");

    nlh = nlmsg_put(handle->skb_out, 0, 0, NLMSG_DONE,
                    sizeof(struct base_buffer), 0);
    NETLINK_CB(handle->skb_out).dst_group = 0;
    memcpy(nlmsg_data(nlh), &base_buf, sizeof(struct base_buffer));

#if PRINT_TIME_K_TO_U
    getnstimeofday(&ts);
    pr_info("Upcall called: sec=%lu, usec=%lu\n", ts.tv_sec, ts.tv_nsec / 1000);
#endif

    //ret = nlmsg_unicast(handle->nl_sk, handle->skb_out, handle->pid);
    ret = netlink_unicast(handle->nl_sk, handle->skb_out, handle->pid, 0);
    if (ret < 0) {
        printk(KERN_INFO "Error while notifying user\n");
    }
}

static int __init test_upcall_init(void)
{
    create_test_device();
    handle = init_upcall();
    return 0;
}

static void __exit test_upcall_fini(void)
{
    close_upcall(handle);
    close_test_device();
}

module_init(test_upcall_init);
module_exit(test_upcall_fini);

MODULE_AUTHOR("Hangchen Yu");
MODULE_DESCRIPTION("Upcall benchmarking module (netlink)");
MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(1) "."
               __stringify(0) "."
               __stringify(0) "."
               "0");
