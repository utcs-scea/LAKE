#include <sys/time.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <assert.h>

#include "upcall.h"
#include "upcall_impl.h"

int init_upcall(void)
{
    char dev_path[128];
    int dev_fd;

    sprintf(dev_path, "/dev/%s", UPCALL_TEST_DEV_NAME);
    dev_fd = open(dev_path, O_RDONLY);
    if (dev_fd <= 0) {
        pr_err("Failed to open upcall device %s\n", dev_path);
        return dev_fd;
    }
    pr_info("Upcall device %s is opened\n", dev_path);

    return dev_fd;
}

void wait_upcall(int fd, void **buf, size_t *size)
{
    int ret;
    struct timeval tv_recv;
    assert(buf && size);
    assert(*buf == NULL || *size >= sizeof(struct base_buffer));

    if (*buf == NULL) {
        *buf = malloc(sizeof(struct base_buffer));
    }
    *size = sizeof(struct base_buffer);

    ret = read(fd, *buf, sizeof(struct base_buffer));

#if PRINT_TIME_K_TO_U
    gettimeofday(&tv_recv, NULL);
    pr_info("Upcall received: sec=%lu, usec=%lu\n", tv_recv.tv_sec, tv_recv.tv_usec);
#endif
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
