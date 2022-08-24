#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <linux/netlink.h>
#include <netlink/netlink.h>
#include "netlink.h"


static struct nl_sock *sk = NULL;


static int netlink_recv_msg(struct nl_msg *msg, void *arg)
{
    struct nlmsghdr *nlh;
    nlh = nlmsg_hdr(msg);
    uint32_t seq = nlh->nlmsg_seq;

    printf("received msg with seq %u\n", xa);

    char* data = nlmsg_data(nlh);

    uint32_t cmd_id = *((uint32_t*) data)

    //TODO: handle cmd_id

    

}


int init_socket() {
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

    err = nl_connect(sk, NETLINK_LAKE_PROT);
    if (err < 0) {
        printf("Error connecting to netlink: %d\n", err);
        exit(1);
    }
}