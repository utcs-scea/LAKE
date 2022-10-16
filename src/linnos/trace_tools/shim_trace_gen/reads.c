#define _GNU_SOURCE 
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <pthread.h>
#include "reads.h"
#include "atomic.h"
#include "tracer.h"

read_ptr real_read_225 = 0;
pread_ptr real_pread_225 = 0;
pread_ptr real_pread64_225 = 0;

pwrite_ptr real_pwrite_225 = 0;
pwrite_ptr real_pwrite64_225 = 0;

#define CAPTURE_READ_CALL 0

#define PRINT_DEBUG 1

void reads_contructor() {
#if CAPTURE_READ_CALL
    real_read_225 = (read_ptr) dlvsym(RTLD_NEXT, "read", "GLIBC_2.2.5");
    if (real_read_225 == NULL) {
        printf("Failed getting real_read_225\n");
        exit(1);
    }
#endif

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


    real_pwrite_225 = (pwrite_ptr) dlvsym(RTLD_NEXT, "pwrite", "GLIBC_2.2.5");
    if (real_pwrite_225 == NULL) {
        printf("Failed getting real_pwrite_225\n");
        exit(1);
    }

    real_pwrite64_225 = (pwrite_ptr) dlvsym(RTLD_NEXT, "pwrite64", "GLIBC_2.2.5");
    if (real_pwrite64_225 == NULL) {
        printf("Failed getting real_pwrite64_225\n");
        exit(1);
    }
}

void __attribute__((constructor)) initialize(void) {
    reads_contructor();
    tracer_constructor();
}

/*
 *   pread64
 */
ssize_t pread64_225(int fd, void *buf, size_t count, off_t offset) {
    struct timeval t2;
    gettimeofday(&t2, NULL);
#if PRINT_DEBUG
    fprintf(stderr, "pread: ts=%.2f, offset=%lu \n", t2.tv_sec*1e6 + t2.tv_usec, offset);
#endif
    tracer_append(offset, count, 0);
    return real_pread64_225(fd, buf, count, offset);
}

/*
 *  pread
 */
ssize_t pread_225(int fd, void *buf, size_t nbyte, off_t offset) {
    struct timeval t2;
    gettimeofday(&t2, NULL);
#if PRINT_DEBUG
    fprintf(stderr, "pread: ts=%.2f, offset=%lu \n", t2.tv_sec*1e6 + t2.tv_usec, offset);
#endif
    tracer_append(offset, nbyte, 0);
    return real_pread_225(fd, buf, nbyte, offset);
}

/*
 *  read
 */
ssize_t read_225(int fd, void *buf, size_t nbyte) {
    uint64_t offset = lseek(fd, 0, SEEK_CUR);
    struct timeval t2;
    gettimeofday(&t2, NULL);
#if PRINT_DEBUG
    fprintf(stderr, "read: ts=%.2f, offset=%lu \n", t2.tv_sec*1e6 + t2.tv_usec, offset);
#endif
    tracer_append(offset, nbyte, 0);
    return real_read_225(fd, buf, nbyte);
}

ssize_t pwrite_225(int fd, const void *buf, size_t count, off_t offset) {
    struct timeval t2;
    gettimeofday(&t2, NULL);
#if PRINT_DEBUG
    fprintf(stderr, "pwrite: ts=%.2f, offset=%lu \n", t2.tv_sec*1e6 + t2.tv_usec, offset);
#endif
    tracer_append(offset, count, 1);
    return real_pwrite_225(fd, buf, count, offset);
}

ssize_t pwrite64_225(int fd, const void *buf, size_t count, off_t offset) {
    struct timeval t2;
    gettimeofday(&t2, NULL);
#if PRINT_DEBUG
    fprintf(stderr, "pwrite64: ts=%.2f, offset=%lu \n", t2.tv_sec*1e6 + t2.tv_usec, offset);
#endif
    tracer_append(offset, count, 1);
    return real_pwrite64_225(fd, buf, count, offset);
}

#if CAPTURE_READ_CALL
__asm__(".symver read_225, read@GLIBC_2.2.5");
#endif
__asm__(".symver pread_225, pread@GLIBC_2.2.5");
__asm__(".symver pread64_225, pread64@GLIBC_2.2.5");
__asm__(".symver pwrite_225, pwrite@GLIBC_2.2.5");
__asm__(".symver pwrite64_225, pwrite64@GLIBC_2.2.5");


//destructor doesn't work if the process terminates abnormaly (ctrl c)

// void __attribute__((destructor)) destruct(void) {
//     // write the array to file
//     fprintf(stderr, "In desctructor!");
//     FILE *f = fopen("/home/itarte/test.csv", "w");
//     int64_t num = atomic_read(&array_index);
    
//     char *temp_string = (char*) malloc(1024 * sizeof(char));
//     for (int i = 0 ; i < num ; i++) {
//         sprintf(temp_string, ", %u\n", timestamps[i], offset[i]);
//         fputs(temp_string, f);
//     }

//     fclose(f);

//     fprintf(stderr, "Wrote %u elements\n", num);
// }


