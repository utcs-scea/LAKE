#include <linux/netlink.h>
#include <linux/module.h>
#include <linux/ctype.h>
#include <linux/mm.h>
#include <net/sock.h>
#include <linux/xarray.h>
#include <linux/completion.h>
#include <linux/slab.h>

#include "netlink.h"

static struct sock *sk = NULL;
DEFINE_XARRAY_ALLOC(cmds_xa); 
static struct kmem_cache *cmd_cache;
static pid_t worker_pid = -1;

struct cmd_data {
    struct completion cmd_done;
    //u32 xa_idx;
};

int lake_send_cmd(void *buf, size_t size, char sync)
{
    int err;
    struct sk_buff *skb_out;
    struct nlmsghdr *nlh;
    struct cmd_data *cmd;
    u32 xa_idx;

    //create a cmd struct
    cmd = (struct cmd_data*) kmem_cache_alloc(cmd_cache, GFP_KERNEL);
    if(IS_ERR(cmd)) {
        pr_alert("Error allocating from cache: %ld\n", PTR_ERR(cmd));
        kmem_cache_destroy(cmd_cache);
        return -ENOMEM;
    }
    //init completion so we can wait on it
    init_completion(&cmd->cmd_done);

    //insert cmd into xarray, getting idx  (void*)cmd
    err = xa_alloc(&cmds_xa, &xa_idx, xa_mk_value(1), XA_LIMIT(0, 1024), GFP_KERNEL); //xa_limit_31b
    //err = xa_store(&cmds_xa, 1, xa_mk_value(1), GFP_KERNEL);
    if (err < 0) {
        pr_alert("Error allocating xa_alloc: %d\n", err);
        return err;
    }

    //create netlink cmd
    skb_out = nlmsg_new(size, 0);
    nlh = nlmsg_put(skb_out, 0, xa_idx, MSG_LAKE_KAPI_REQ, size, 0);
    NETLINK_CB(skb_out).dst_group = 0;
    memcpy(nlmsg_data(nlh), buf, size);

    err = netlink_unicast(sk, skb_out, worker_pid, 0);
    if (err < 0) {
        pr_err("Failed to send netlink skb to API server, error=%d\n", err);
        nlmsg_free(skb_out);
        return err;
    }
    pr_err("msg sent\n");
    nlmsg_free(skb_out);

    // sync if requested
    if (sync == 1) {
        wait_for_completion(&cmd->cmd_done);
    }

    return err;
}

static void netlink_recv_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh = (struct nlmsghdr*) skb->data;
    //TODO: get ret
    // ret = (struct cmd_data*) nlmsg_data(nlh);
    struct cmd_data *cmd;
    u32 xa_idx = nlh->nlmsg_seq;

    if (unlikely(worker_pid == -1)) {
        worker_pid = nlh->nlmsg_pid;
        printk(KERN_INFO "Setting worker PID to %d\n", worker_pid);
        return;
    }

    //find cmd in xa
    cmd = (struct cmd_data*) xa_load(&cmds_xa, xa_idx);
    if (!cmd) {
        pr_alert("Error looking up cmd %u at xarray\n", xa_idx);
    }

    //if there's anyone waiting, free them
    complete(&cmd->cmd_done);
    //free from cache
    kmem_cache_free(cmd_cache, cmd);
    //erase from xarray
    xa_erase(&cmds_xa, xa_idx);
}

static void null_constructor(void *argument) {
}

int lake_init_socket(void) {
    static struct netlink_kernel_cfg netlink_cfg = {
        .input = netlink_recv_msg,
    };

    sk = netlink_kernel_create(&init_net, NETLINK_LAKE_PROT, &netlink_cfg);
    if (!sk) {
        pr_err("Error creating netlink socket\n");
        return -ENOMEM;
    }

    //init slab cache (xarray requires 4-alignment)
    cmd_cache = kmem_cache_create("lake_cmd_cache", sizeof(struct cmd_data), 4, 0, null_constructor);
    if(IS_ERR(cmd_cache)) {
        pr_alert("Error creating cache: %ld\n", PTR_ERR(cmd_cache));
        return -ENOMEM;
    }
    return 0;
}

void lake_destroy_socket(void) {
    //TODO: wait a bit
    netlink_kernel_release(sk);
    xa_destroy(&cmds_xa);
    kmem_cache_destroy(cmd_cache);
}