#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>
#include "lake_shm.h"

static char *kshm_base = NULL;
static int kshm_fd = 0;
static long kshm_size = 0;

void *lake_shm_address(long offset)
{
    return (void *)(kshm_base + offset);
}

int lake_shm_init(void)
{
    char dev_name[64];
    sprintf(dev_name, "/dev/%s", KAVA_SHM_DEV_NAME);
    kshm_fd = open(dev_name, O_RDWR);

    if (kshm_fd <= 0) {
        printf("Shared memory driver (%s) is not installed: %s\n",
                dev_name, strerror(errno));
        return errno;
    }

    int ret = ioctl(kshm_fd, KAVA_SHM_GET_SHM_SIZE, &kshm_size);
    if (ret) {
        printf("Failed IOCTL to shared memory driver\n");
        return ret;
    }
    else {
        printf("Request shared memory size: %lx\n", kshm_size);
    }

    kshm_base = (char *)mmap(NULL, kshm_size, PROT_READ | PROT_WRITE, MAP_SHARED, kshm_fd, 0);
    if (kshm_base == MAP_FAILED) {
        printf("Failed to mmap shared memory regionn\n");
        return (int)(uintptr_t)kshm_base;
    }
    else {
        printf("mmap shared memory region to 0x%lx, size=0x%lx\n",
                (uintptr_t)kshm_base, kshm_size);
    }

    return 0;
}

void lake_shm_fini(void)
{
    if (kshm_fd > 0 && kshm_base) {
        munmap(kshm_base, kshm_size);
        kshm_base = NULL;
    }

    if (kshm_fd > 0) {
        close(kshm_fd);
        kshm_fd = 0;
    }
}
