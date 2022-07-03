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

struct msghdr msg;

static struct nlmsghdr *create_nlmsg(size_t data_len) {
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
    msg.msg_iov = iov;
    msg.msg_iovlen = 1;
    nlh->nlmsg_len = iov->iov_len = NLMSG_SPACE(data_len);

    return nlh;
}

int init_upcall(void)
{
    int sock_fd;
    size_t buf_len = 0, set_buf_len = NL_MSG_LEN_MAX;
    socklen_t buf_len_size = sizeof(size_t);
    struct nlmsghdr *nlh;
    struct sockaddr_nl src_addr, dest_addr;
    ssize_t ret;

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
    nlh = create_nlmsg(0);
    msg.msg_name = (void *)&dest_addr;
    msg.msg_namelen = sizeof(dest_addr);

    /* Notify kernel handler */
    ret = sendmsg(sock_fd, &msg, 0);
    assert(ret > 0);
    free(msg.msg_iov);
    free(nlh);

    return sock_fd;
}

void wait_upcall(int fd, void **buf, size_t *size)
{
    struct nlmsghdr *nlh;
    struct timeval tv_recv;

    assert(buf && size);
    assert(*buf == NULL || *size >= sizeof(struct base_buffer));

    if (*buf == NULL) {
        *buf = malloc(sizeof(struct base_buffer));
    }
    *size = sizeof(struct base_buffer);

    /* Prepare receive message */
    nlh = create_nlmsg(sizeof(struct base_buffer));

    //recvmsg(fd, &msg, 0);

    while (1) {
        int x;
        x = recvmsg(fd, &msg, MSG_DONTWAIT);
        if (x > 0) break;
    }
 
#if PRINT_TIME_K_TO_U
    gettimeofday(&tv_recv, NULL);
    pr_info("Upcall received: sec=%lu, usec=%lu\n", tv_recv.tv_sec, tv_recv.tv_usec);
#endif

    memcpy(*buf, NLMSG_DATA(nlh), *size);
    free(msg.msg_iov);
    free(nlh);
}

void close_upcall(int fd)
{
    close(fd);
}

int main() {
    int fd = init_upcall();
    assert(fd > 0);
    struct base_buffer *buf = NULL;
    size_t size;

    while (1) {
        wait_upcall(fd, (void **)&buf, &size);
    }

    free(buf);
    close_upcall(fd);
    return 0;
}
