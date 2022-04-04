#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/types.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <assert.h>

#include <signal.h>

#include "upcall.h"
#include "upcall_impl.h"

int __klib_interrupt = 0;

struct base_buffer recv_buf;

void signal_func(int signo, siginfo_t *info, void *context)
{
    recv_buf.r0 = (uint64_t)info->si_utime;
    recv_buf.r1 = (uint64_t)info->si_stime;
    recv_buf.r2 = (uint64_t)info->si_ptr;
    recv_buf.r3 = (uint64_t)info->si_addr;
    recv_buf.buf_size = (uint64_t)info->si_call_addr;
    __klib_interrupt = 1;
}

int init_upcall(void)
{
    char dev_path[128];
    int dev_fd;
    int pid;
    int ret;
    struct sigaction sig;

    sprintf(dev_path, "/dev/%s", UPCALL_TEST_DEV_NAME);
    dev_fd = open(dev_path, O_RDWR);
    if (dev_fd <= 0) {
        pr_err("Failed to open upcall device %s\n", dev_path);
        return dev_fd;
    }
    pr_info("Upcall device %s is opened\n", dev_path);

    /* Send PID to kernel */
    pid = getpid();
    pr_info("Worker PID is %d\n", pid);
    ret = ioctl(dev_fd, KAVA_SET_USER_PID, pid);
    assert(ret == 0);

    sig.sa_sigaction = signal_func;
    sig.sa_flags = SA_SIGINFO;
    sigaction(UPCALL_TEST_SIG, &sig, NULL);

    return dev_fd;
}

void wait_upcall(int fd, void **buf, size_t *size)
{
    sigset_t mask, oldmask;
    int ret;
    struct timeval tv_recv;

    assert(buf && size);
    assert(*buf == NULL || *size >= sizeof(struct base_buffer));

    if (*buf == NULL) {
        *buf = malloc(sizeof(struct base_buffer));
    }
    *size = sizeof(struct base_buffer);

    /* Set up the mask of signals to temporarily block */
    sigemptyset(&mask);
    sigaddset(&mask, UPCALL_TEST_SIG);

    /* Wait for a signal to arrive */
    sigprocmask(SIG_BLOCK, &mask, &oldmask);
    while (!__klib_interrupt)
        sigsuspend(&oldmask);
    sigprocmask(SIG_UNBLOCK, &mask, NULL);

#if PRINT_TIME_K_TO_U
    gettimeofday(&tv_recv, NULL);
    pr_info("Upcall received: sec=%lu, usec=%lu\n", tv_recv.tv_sec, tv_recv.tv_usec);
#endif

    /* Get data */
    //printf("received value %lu\n", recv_buf.r0);
    memcpy(*buf, (void *)&recv_buf, sizeof(struct base_buffer));
    memset(&recv_buf, 0, sizeof(struct base_buffer));

    /* Notify kernel */
    ret = ioctl(fd, KAVA_ACK_SINGAL);
    assert(ret == 0);
    __klib_interrupt = 0;
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

    close_upcall(fd);
    return 0;
}
