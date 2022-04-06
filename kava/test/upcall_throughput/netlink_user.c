#include <linux/netlink.h>
#include <sys/socket.h>
#include <sys/time.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <assert.h>

#include "upcall.h"
#include "upcall_impl.h"

struct nlmsghdr *nlh_send;
struct msghdr msg_send;
struct nlmsghdr *nlh_recv;
struct msghdr msg_recv;

struct sockaddr_nl src_addr, dest_addr;

static struct nlmsghdr *create_nlmsg(struct msghdr *msg, size_t data_len) {
    struct nlmsghdr *nlh;
    struct iovec *iov;

    nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(data_len));
    assert(nlh != NULL);
    memset(nlh, 0, NLMSG_SPACE(data_len));
    nlh->nlmsg_pid = getpid();
    nlh->nlmsg_flags = 0;
    nlh->nlmsg_flags &= ~NLM_F_ACK;

    iov = (struct iovec *)malloc(sizeof(struct iovec));
    iov->iov_base = (void *)nlh;
    msg->msg_iov = iov;
    msg->msg_iovlen = 1;
    nlh->nlmsg_len = iov->iov_len = NLMSG_SPACE(data_len);

    return nlh;
}

int init_upcall_2(size_t message_size)
{
    int sock_fd;
    size_t buf_len = 0, set_buf_len = 2097152;
    socklen_t buf_len_size = sizeof(size_t);

    sock_fd = socket(PF_NETLINK, SOCK_RAW, NETLINK_USER);
    if (sock_fd < 0)
        return sock_fd;

    /* Bind source address */
    memset(&src_addr, 0, sizeof(src_addr));
    src_addr.nl_family = AF_NETLINK;
    src_addr.nl_pid = getpid(); /* self pid */
    bind(sock_fd, (struct sockaddr *)&src_addr, sizeof(src_addr));

    /* Check information */
    setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &set_buf_len, sizeof(size_t));
    setsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &set_buf_len, sizeof(size_t));

    getsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &buf_len, &buf_len_size);
    pr_info("Default socket recv buffer size %ld\n", buf_len);
    getsockopt(sock_fd, SOL_SOCKET, SO_SNDBUF, &buf_len, &buf_len_size);
    pr_info("Default socket send buffer size %ld\n", buf_len);

    /* Set destination address */
    memset(&dest_addr, 0, sizeof(dest_addr));
    dest_addr.nl_family = AF_NETLINK;
    dest_addr.nl_pid = 0; /* Linux kernel */

    /* Initialize message structure */
    nlh_send = create_nlmsg(&msg_send, message_size);
    msg_send.msg_name = (void *)&dest_addr;
    msg_send.msg_namelen = sizeof(dest_addr);

    nlh_recv = create_nlmsg(&msg_recv, message_size);

    return sock_fd;
}

void wait_upcall_2(int fd, void *src, void *dst, size_t size)
{
    ssize_t ret;
    memcpy(NLMSG_DATA(nlh_send), src, size);
    ret = sendmsg(fd, &msg_send, 0);
    assert(ret > 0);

    recvmsg(fd, &msg_recv, 0);
    assert(ret > 0);
    memcpy(dst, NLMSG_DATA(nlh_recv), size);
}

void close_upcall(int fd)
{
    free(msg_send.msg_iov);
    free(nlh_send);
    free(msg_recv.msg_iov);
    free(nlh_recv);

    close(fd);
}

int main(int argc, char *argv[]) {
    int fd;
    struct timeval tv_start, tv_end;
    int i;
    int test_num, message_size;
    struct shared_region *src, *dst;

    parse_input_args(argc, argv, &test_num, &message_size);
    fd = init_upcall_2(message_size);
    assert(fd > 0);

    src = (struct shared_region *)malloc(message_size);
    src->size = message_size;
    dst = (struct shared_region *)malloc(message_size);

    gettimeofday(&tv_start, NULL);
    for (i = 0; i < test_num; i++) {
        wait_upcall_2(fd, (void *)src, (void *)dst, message_size);
    }
    gettimeofday(&tv_end, NULL);

    double total_time = (tv_end.tv_sec - tv_start.tv_sec) + (tv_end.tv_usec - tv_start.tv_usec) / 1000000.0;
    pr_info("Total time: %lf sec, throughput = %lf MB/s\n",
            total_time,  message_size * test_num / total_time / 1024 / 1024);

    free(src);
    free(dst);
    close_upcall(fd);
    return 0;
}
