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

upcall_handle_t handle;

static void nl_recvmsg_handler(struct sk_buff *skb)
{
    struct nlmsghdr *nlh;
    struct shared_region *recv_buf;
    size_t size;
    int ret;

    nlh = (struct nlmsghdr *)skb->data;
    recv_buf = (struct shared_region *)nlmsg_data(nlh);
    size = recv_buf->size;
    handle->pid = nlh->nlmsg_pid;

    handle->skb_out = nlmsg_new(recv_buf->size, 0);
    BUG_ON(!handle->skb_out && "Failed to allocate new skb\n");

    nlh = nlmsg_put(handle->skb_out, 0, 0, NLMSG_DONE, size, 0);
    NETLINK_CB(handle->skb_out).dst_group = 0;
    memcpy(nlmsg_data(nlh), recv_buf, size);

    //ret = nlmsg_unicast(handle->nl_sk, handle->skb_out, handle->pid);
    ret = netlink_unicast(handle->nl_sk, handle->skb_out, handle->pid, 0);
    BUG_ON(ret < 0 && "Error while notifying user\n");
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
