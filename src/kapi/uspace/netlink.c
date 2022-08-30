#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <inttypes.h>
#include <linux/netlink.h>
#include <netlink/netlink.h>
#include <netlink/msg.h>
#include <signal.h>
#include "netlink.h"
#include "commands.h"
#include "lake_kapi.h"

static struct nl_sock *sk = NULL;

static void lake_send_cmd(uint32_t seqn, void* buf, size_t len) {
    int err;
    struct nl_msg *msg;
    struct nlmsghdr *nlh;

    msg = nlmsg_alloc_simple(MSG_LAKE_KAPI_REP, 0);
    if (!msg)
        printf("error on nlmsg_alloc_simple\n");
   
    if(buf && len) {
        err = nlmsg_append(msg, buf, len, NLMSG_ALIGNTO);
        if (err < 0)
            printf("error on nlmsg_append %d\n", err);
    }

    nl_complete_msg(sk, msg);
   
    nlh = nlmsg_hdr(msg);
    nlh->nlmsg_seq = seqn;
    err = nl_send(sk, msg);
    if(err < 0)
        printf("error on nl_send %d\n", err);

    nlmsg_free(msg);
}

static int netlink_recv_msg(struct nl_msg *msg, void *arg) {
    struct nlmsghdr *nlh;
    nlh = nlmsg_hdr(msg);
    uint32_t seq = nlh->nlmsg_seq;
    printf("received msg with seq %u\n", seq);
    void* data = nlmsg_data(nlh);
    struct lake_cmd_ret cmd_ret;
    lake_handle_cmd(data, &cmd_ret);
    printf("command handled, replying\n");
    lake_send_cmd(seq, &cmd_ret, sizeof(cmd_ret));
    printf("reply sent\n");
}

void lake_destroy_socket() {
    nl_socket_free(sk);
}

void lake_recv() {
    nl_recvmsgs_default(sk);
}

int lake_init_socket() {
    int err;

    sk = nl_socket_alloc();
    nl_socket_modify_cb(sk, NL_CB_VALID, NL_CB_CUSTOM, netlink_recv_msg, NULL);
    nl_socket_disable_seq_check(sk);
    
    //XXX this looks like too much
    nl_socket_set_buffer_size(sk, 2*1024*1024, 2*1024*1024);
    nl_socket_set_msg_buf_size(sk, 2*1024*1024);

    nl_socket_disable_auto_ack(sk);
    nl_socket_set_passcred(sk, 0);
    nl_socket_recv_pktinfo(sk, 0);
    nl_socket_set_nonblocking(sk);

    while(1) {
        err = nl_connect(sk, NETLINK_LAKE_PROT);
        if (err < 0) {
            printf("Error connecting to netlink (%d), sleeping..\n", err);
            //exit(1);
            sleep(1);
        }
        else break;
    }

    //ping so kernel can get our pid
    lake_send_cmd(0, 0, 0);
    printf("Netlink connected, message sent to kernel\n");
}