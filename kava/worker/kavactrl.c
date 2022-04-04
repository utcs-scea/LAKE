#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include "control.h"
#include "debug.h"

int main() {
    int dev_fd;
    char dev_name[32];
    int r;

    sprintf(dev_name,"/dev/%s", KAVA_DEV_NAME);
    dev_fd = open(dev_name, O_RDWR);
    if (dev_fd < 0) {
        pr_err("Failed to open kAvA device: %s\n", strerror(errno));
        return 0;
    }

    r = ioctl(dev_fd, KAVA_IOCTL_STOP_WORKER);
    assert(r == 0 && "Failed to stop worker");

    close(dev_fd);
    return 0;
}
