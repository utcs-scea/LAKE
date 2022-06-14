#ifndef __COMMON_H__
#define __COMMON_H__


#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <linux/userfaultfd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdio.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <poll.h>
#include <fcntl.h>
#include <mutex>
#include <list>

#define PAGE_SZ 4096

inline int inode_from_fd(int fd) {
    struct stat file_stat;  
    int ret;  
    ret = fstat (fd, &file_stat);  
    if (ret < 0) {  
    } 
    return file_stat.st_ino;
}

#endif