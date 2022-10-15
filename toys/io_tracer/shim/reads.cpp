#include "common.h"
#include "reads.h"
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

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
    struct timeval t2;
    gettimeofday(&t2, NULL);
    //printf(stderr, ":r,%d,%u\n", fd, i);
    fprintf(stderr, "pread64: ts=%.2f, offset=%llu \n", t2.tv_sec*1e6 + t2.tv_usec, offset);
    ret = real_pread64_225(fd, buf, count, offset);
    return ret;
}


/*
 *  pread
 */
ssize_t pread_225(int fd, void *buf, size_t nbyte, off_t offset) {
    ssize_t ret;
    struct timeval t2;
    gettimeofday(&t2, NULL);
    //printf(stderr, ":r,%d,%u\n", fd, i);
    fprintf(stderr, "pread: ts=%.2f, offset=%llu \n", t2.tv_sec*1e6 + t2.tv_usec, offset);
    ret = real_pread_225(fd, buf, nbyte, offset);
    
    return ret;
}

/*
 *  read
 */
ssize_t read_225(int fd, void *buf, size_t nbyte) {
    ssize_t ret;
    struct timeval t2;
    gettimeofday(&t2, NULL);
    //printf(stderr, ":r,%d,%u\n", fd, i);
    fprintf(stderr, "read: ts=%.2f, offset=%llu \n", t2.tv_sec*1e6 + t2.tv_usec, 0);
    ret = real_read_225(fd, buf, nbyte);
    return ret;
}

void __attribute__((constructor)) initialize(void) {
    reads_contructor();
}


__asm__(".symver pread_225, pread@GLIBC_2.2.5");
__asm__(".symver pread64_225, pread64@GLIBC_2.2.5");
__asm__(".symver read_225, read@GLIBC_2.2.5");