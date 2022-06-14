#include "common.h"
#include "reads.h"

read_ptr real_read_225 = 0;
pread_ptr real_pread_225 = 0;
pread_ptr real_pread64_225 = 0;

void reads_contructor() {
    real_read_225 = (read_ptr) dlvsym(RTLD_NEXT, "read", "GLIBC_2.2.5");
    if (real_read_225 == NULL) {
        printf("Failed getting real_read_225\n");
        exit(1);
    }

    real_pread_225 = (pread_ptr) dlvsym(RTLD_NEXT, "pread", "GLIBC_2.2.5");
    if (real_pread_225 == NULL) {
        printf("Failed getting real_pread_225\n");
        exit(1);
    }

    real_pread64_225 = (pread_ptr) dlvsym(RTLD_NEXT, "pread64", "GLIBC_2.2.5");
    if (real_pread64_225 == NULL) {
        printf("Failed getting real_pread64_225\n");
        exit(1);
    }
}

/*
 *   pread64
 */
ssize_t pread64_225(int fd, void *buf, size_t count, off_t offset) {
    ssize_t ret;
    uint64_t position = lseek(fd, 0, SEEK_CUR);
    position += offset;

    ret = real_pread64_225(fd, buf, count, offset);

    uint32_t first, last;
    first = position/PAGE_SZ;
    last = (position+ret-1)/PAGE_SZ;
    int inode = inode_from_fd(fd);

    for (int i = first ; i <= last; i++) {
        fprintf(stderr, ":pr64,%d,%u\n", inode, i);
    }
    return ret;
}


/*
 *  pread
 */
ssize_t pread_225(int fd, void *buf, size_t nbyte, off_t offset) {
    ssize_t ret;
    uint64_t position = lseek(fd, 0, SEEK_CUR);
    position += offset;

    ret = real_pread_225(fd, buf, nbyte, offset);

    uint32_t first, last;
    first = position/PAGE_SZ;
    last = (position+ret-1)/PAGE_SZ;
    int inode = inode_from_fd(fd);

    for (int i = first ; i <= last; i++) {
        fprintf(stderr, ":pr,%d,%u\n", inode, i);
    }
    return ret;
}

/*
 *  read
 */
ssize_t read_225(int fd, void *buf, size_t nbyte) {
    ssize_t ret;
    uint64_t position = lseek(fd, 0, SEEK_CUR);
    ret = real_read_225(fd, buf, nbyte);

    uint32_t first, last;
    first = position/PAGE_SZ;
    last = (position+ret-1)/PAGE_SZ;
    //int inode = inode_from_fd(fd);

    for (int i = first ; i <= last; i++) {
        //fprintf(stderr, ":read, %d, %u\n", inode, i);
        fprintf(stderr, ":r,%d,%u\n", fd, i);
    }
    return ret;
}

__asm__(".symver pread_225, pread@GLIBC_2.2.5");
__asm__(".symver pread64_225, pread64@GLIBC_2.2.5");
__asm__(".symver read_225, read@GLIBC_2.2.5");