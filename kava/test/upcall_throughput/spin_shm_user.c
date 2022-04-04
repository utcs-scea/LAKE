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

void wait_upcall_2(int fd, void *src, void *dst, size_t size)
{
    int ret;
    struct shared_region *shm = (struct shared_region *)shared_mem;

    if (!shm) return;

    pthread_mutex_lock(&lock);

    /* Send to shared memory. */
    memcpy(shm, src, size);
    shm->doorbell_kern = 1;

    /* Spin wait for response. */
    while (shm->doorbell_user == 0);
    memcpy(dst, shm, size);
    shm->doorbell_user = 0;

    pthread_mutex_unlock(&lock);
}

void close_upcall(int fd)
{
    pthread_mutex_destroy(&lock);
    munmap(shared_mem, MMAP_NUM_PAGES * getpagesize());
    close(fd);
}

int main(int argc, char *argv[]) {
    int fd;
    struct timeval tv_start, tv_end;
    int i;
    int test_num, message_size;
    struct shared_region *src, *dst;

    parse_input_args(argc, argv, &test_num, &message_size);
    fd = init_upcall();
    assert(fd > 0);

    src = (struct shared_region *)malloc(message_size);
    src->doorbell_kern = 0;
    src->doorbell_user = 0;
    src->size = message_size;
    dst = (struct shared_region *)malloc(message_size);

    gettimeofday(&tv_start, NULL);
    for (i = 0; i < test_num; i++) {
        /* Copy memory from src to shared region, and then from shared region to dst. */
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
