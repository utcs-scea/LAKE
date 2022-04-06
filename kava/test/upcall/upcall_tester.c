#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/ioctl.h>

#include "upcall_impl.h"

int main() {
    int dev_fd;
    char name[20];
    int r;

    sprintf(name, "/dev/%s", UPCALL_DEV_NAME);
    dev_fd = open(name, O_RDWR);
    if (dev_fd < 0) {
        pr_err("Failed to open upcall device: %s\n", strerror(errno));
        return 0;
    }

    r = ioctl(dev_fd, KAVA_TEST_START_UPCALL, 5);
    assert(r == 0 && "Failed to stop worker");

    close(dev_fd);
    return 0;
}
