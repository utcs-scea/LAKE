#include <linux/netlink.h>
#include <linux/module.h>
#include <linux/ctype.h>
#include <linux/mm.h>
#include <net/sock.h>
#include <linux/xarray.h>
#include <linux/completion.h>
#include <linux/slab.h>

#include "netlink.h"

static struct nl_sock *sk = NULL;
static struct xarray cmds_xa;
static struct kmem_cache *cmd_cache;

struct cmd_data {
    struct completion cmd_done;
    //u32 xa_idx;
};

void lake_send_cmd(void *buf, size_t size, char sync)
{
    int err;
    struct nl_msg *msg;
    struct nlmsghdr *nlh;
    struct cmd_data *cmd;
    u32 xa_idx;

    msg = nlmsg_alloc_simple(type, flags);
    if (!msg)
            return -NLE_NOMEM;
   
    if (buf && size) {
        err = nlmsg_append(msg, buf, size, NLMSG_ALIGNTO);
        if (err < 0)
             goto errout;
    }
    nl_complete_msg(sk, msg);
   
    //boilerplate done, do our job now
    //create a cmd struct
    cmd = (struct cmd_data*) kmem_cache_alloc(cmd_cache, GFP_KERNEL);
    if(IS_ERR(cmd)) {
        pr_alert("Error allocating from cache: %ld\n", PTR_ERR(cmd));
        kmem_cache_destroy(cmd_cache);
        return -ENOMEM;
    }
    //init completion so we can wait on it
    init_completion(&cmd->cmd_done);

    //insert cmd into xarray, getting idx
    err = xa_alloc(cmds_xa, &xa_idx, (void*)cmd, xa_limit_32b, GFP_KERNEL); //XA_LIMIT(0, 512)
    if (err < 0) {
        pr_alert("Error allocating xa_alloc: %d\n", err);
        return -ENOMEM;
    }

    //fill in our own seq with the xarray idx
    nlh = nlmsg_hdr(msg);
    nlh->nlmsg_seq = xa_idx;

    //done tinkering, send message
    err = nl_send(sk, msg);
errout:
    nlmsg_free(msg);

    // sync if requested
    if (sync == 1) {
        wait_for_completion(&cmd->cmd_done);
    }

    return err;
}

static int netlink_recv_msg(struct nl_msg *msg, void *arg)
{
    struct nlmsghdr *nlh;
    struct cmd_data *cmd;
    u32 xa_idx;

    nlh = nlmsg_hdr(msg);
    xa_idx = nlh->nlmsg_seq;

    //find cmd in xa
    cmd = (struct cmd_data*) xa_load(cmds_xa, xa_idx);
    if (!cmd) {
        pr_alert("Error looking up cmd %u at xarray\n", xa_idx);
        return -ENOENT;
    }

    //if there's anyone waiting, free them
    complete(cmd->cmd_done);
    //free from cache
    kmem_cache_free(cmd_cache, cmd);
    //erase from xarray
    xa_erase(cmds_xa, xa_idx);
}

static void null_constructor(void *argument) {
}

int lake_init_socket() {
    sk = nl_socket_alloc();
    nl_socket_modify_cb(sk, NL_CB_VALID, NL_CB_CUSTOM, netlink_recv_msg, NULL);
    nl_socket_disable_seq_check(sk);
    nl_connect(sk, NETLINK_LAKE_PROT);

    //init xarray for cmds
    xa_init(&cmds_xa);

    //init slab cache (xarray requires 4-alignment)
    cmd_cache = kmem_cache_create("lake_cmd_cache", sizeof(struct example_struct), 4, 0, null_constructor);
    if(IS_ERR(cmd_cache)) {
        pr_alert("Error creating cache: %ld\n", PTR_ERR(cmd_cache));
        return -ENOMEM;
    }
}

void lake_destroy_socket() {
    //TODO: wait a bit
    nl_socket_free(sk);
}