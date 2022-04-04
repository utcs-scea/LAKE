#include <sys/time.h>

#include <sys/mman.h>
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
#include <pthread.h>

#include "upcall.h"
#include "upcall_impl.h"

void *shared_mem;
pthread_mutex_t lock;

int init_upcall(void)
{
    char dev_path[128];
    int dev_fd;

    sprintf(dev_path, "/dev/%s", UPCALL_TEST_DEV_NAME);
    dev_fd = open(dev_path, O_RDWR);
    if (dev_fd <= 0) {
        pr_err("Failed to open upcall device %s\n", dev_path);
        return dev_fd;
    }
    pr_info("Upcall device %s is opened\n", dev_path);

    /* Mmap kernel-space memory */
    shared_mem = (struct base_buffer *)mmap(0, MMAP_NUM_PAGES * getpagesize(),
                    PROT_READ | PROT_WRITE, MAP_SHARED, dev_fd, 0);
    if (shared_mem == MAP_FAILED) {
        pr_err("Failed to mmap shared memory");
        close(dev_fd);
        return -1;
    }

    pthread_mutex_init(&lock, NULL);

    return dev_fd;
}

void wait_upcall(int fd, void **buf, size_t *size)
{
    int ret;
    struct shared_region *shm;
    struct timeval tv_recv;

    assert(buf && size);
    assert(*buf == NULL || *size >= sizeof(struct base_buffer));

    if (!shared_mem) return;

    if (*buf == NULL) {
        *buf = malloc(sizeof(struct base_buffer));
    }
    *size = sizeof(struct base_buffer);

    pthread_mutex_lock(&lock);

    /* Spin wait */
    shm = (struct shared_region *)shared_mem;
    while (shm->doorbell == 0);

#if PRINT_TIME_K_TO_U
    gettimeofday(&tv_recv, NULL);
    pr_info("Upcall received: sec=%lu, usec=%lu\n", tv_recv.tv_sec, tv_recv.tv_usec);
#endif

    memcpy(*buf, (void *)&shm->data, sizeof(struct base_buffer));
    shm->doorbell = 0;

    pthread_mutex_unlock(&lock);
}

void close_upcall(int fd)
{
    pthread_mutex_destroy(&lock);
    munmap(shared_mem, MMAP_NUM_PAGES * getpagesize());
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
