/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */



#include <linux/netlink.h>
#include <linux/module.h>
#include <linux/ctype.h>
#include <linux/mm.h>
#include <net/sock.h>
#include <linux/xarray.h>
#include <linux/completion.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/ktime.h>
#include "netlink.h"
#include "commands.h"

static struct sock *sk = NULL;
//DEFINE_XARRAY_ALLOC(cmds_xa); 
DEFINE_XARRAY(cmds_xa); 
static struct kmem_cache *cmd_cache;
static pid_t worker_pid = -1;
static int max_counter = (1<<10);

//static atomic_t seq_counter = ATOMIC_INIT(0);
static DEFINE_SPINLOCK(counter_lock);
u32 id_counter = 0;

//we can get away with an atomic here
static CUresult last_cu_err = 0;
//static atomic_t last_cu_err = ATOMIC_INIT(0);


struct cmd_data {
    struct completion cmd_done;
    struct lake_cmd_ret ret;
    char sync;
}; //__attribute__ ((aligned (8)));


// ret is only filled in case sync is CMD_SYNC
void lake_send_cmd(void *buf, size_t size, char sync, struct lake_cmd_ret* ret)
{
    int err;
    struct sk_buff *skb_out;
    struct nlmsghdr *nlh;
    struct cmd_data *cmd;
    int xa_idx;

    //create a cmd struct
    //cmd = (struct cmd_data*) kmalloc(sizeof(struct cmd_data), GFP_KERNEL);
    spin_lock(&counter_lock);
    cmd = (struct cmd_data*) kmem_cache_alloc(cmd_cache, GFP_KERNEL);
    spin_unlock(&counter_lock);
    if(IS_ERR(cmd)) {
        pr_warn("Error allocating from cache: %ld\n", PTR_ERR(cmd));
        //kmem_cache_destroy(cmd_cache);
        ret->res = CUDA_ERROR_OUT_OF_MEMORY;
    }

    //init completion so we can wait on it
    init_completion(&cmd->cmd_done);
    cmd->sync = sync;

    spin_lock(&counter_lock);
    //xa_idx = atomic_add_return(1, &seq_counter);
    xa_idx = id_counter++;
    if(unlikely(xa_idx >= max_counter))
        //atomic_set(&seq_counter, 0);
        id_counter = 0;
    spin_unlock(&counter_lock);

    if(xa_err(xa_store(&cmds_xa, xa_idx, (void*)cmd, GFP_KERNEL))) {
        pr_warn("Error allocating xa_alloc: %d\n", err);
        ret->res = CUDA_ERROR_OPERATING_SYSTEM;
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
        ret->res = CUDA_ERROR_OPERATING_SYSTEM;
    }

    // sync if requested
    if (sync == CMD_SYNC) {
        //XXX
        //wait_for_completion(&cmd->cmd_done);
        while (1) {
            err = wait_for_completion_interruptible(&cmd->cmd_done);
            if (err == 0) break;
        }
        memcpy(ret, (void*)&cmd->ret, sizeof(struct lake_cmd_ret));
        // if we sync, its like the cmd never existed, so clear every trace
        kmem_cache_free(cmd_cache, cmd);
        xa_erase(&cmds_xa, xa_idx);
    }

    if (likely(last_cu_err == 0))
        ret->res = CUDA_SUCCESS;
    else
        ret->res = last_cu_err;
}

static void netlink_recv_msg(struct sk_buff *skb)
{
    struct nlmsghdr *nlh = (struct nlmsghdr*) skb->data;
    //struct lake_cmd_ret *ret = (struct lake_cmd_ret*) nlmsg_data(nlh);
    void *ret = NLMSG_DATA(nlh);
    struct cmd_data *cmd;
    u32 xa_idx = nlh->nlmsg_seq;

    if (unlikely(worker_pid == -1)) {
        worker_pid = nlh->nlmsg_pid;
        printk(KERN_INFO "Setting worker PID to %d\n", worker_pid);
        return;
    }

    //find cmd in xa
    xa_lock(&cmds_xa);
    cmd = (struct cmd_data*) xa_load(&cmds_xa, xa_idx);
    xa_unlock(&cmds_xa);
    if (!cmd) {
        pr_warn("Error (0) looking up cmd %u at xarray\n", xa_idx);
        xa_erase(&cmds_xa, xa_idx);
        return;
    }
    memcpy(&(cmd->ret), ret, sizeof(struct lake_cmd_ret));

    //if the cmd is async, no one will read this cmd, so clear
    if (cmd->sync == CMD_ASYNC) {
        if(unlikely(cmd->ret.res > 0)) {
            last_cu_err = cmd->ret.res;
        }
        //erase from xarray
        xa_erase(&cmds_xa, xa_idx);
        //free from cache
        kmem_cache_free(cmd_cache, cmd);
    }
    else {
        //if there's anyone waiting, free them
        //if the cmd is sync, whoever we woke up will clean up
        complete(&cmd->cmd_done);
    }
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

    //init slab cache (xarray requires at least 4-alignment)
    cmd_cache = kmem_cache_create("lake_cmd_cache", sizeof(struct cmd_data), 8, 0, null_constructor);
    if(IS_ERR(cmd_cache)) {
        pr_warn("Error creating cache: %ld\n", PTR_ERR(cmd_cache));
        return -ENOMEM;
    }
    return 0;
}

void lake_destroy_socket(void) {
    unsigned long idx = 0;
    void* entry;
    //TODO: set a halt flag

    //free up all cache entries so the kernel doesnt yell at us
    xa_for_each(&cmds_xa, idx, entry) {
        if (entry)
            kmem_cache_free(cmd_cache, entry);
    }

    //kmem_cache_destroy(cmd_cache);
    xa_destroy(&cmds_xa);
    netlink_kernel_release(sk);
}