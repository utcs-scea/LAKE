#include "common.h"
#include "reads.h"
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include<signal.h>

#define MAX_ENTRIES 5000
read_ptr real_read_225 = 0;
pread_ptr real_pread_225 = 0;
pread_ptr real_pread64_225 = 0;
uint64_t *timestamps;
uint64_t *offset;
int64_t array_index = 0;

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

int64_t atomic_read(int64_t* ptr) {
    return __atomic_load_n (ptr,  __ATOMIC_SEQ_CST);
}

void atomic_add(int64_t* ptr, int val) {
    __atomic_add_fetch(ptr, val, __ATOMIC_SEQ_CST);
}

int64_t atomic_fetch_inc(int64_t* ptr) {
    return __atomic_fetch_add(ptr, 1, __ATOMIC_SEQ_CST);
}

void append_offset(uint64_t off) {
    //fail fast
    int idx = atomic_read(&array_index);
    if(idx >= MAX_ENTRIES-1)
        return;

    //add one and get our index, need to check bounds again
    idx = atomic_fetch_inc(&array_index) - 1; //we inc before fetch
    if(idx >= MAX_ENTRIES-1)
        return;

    offset[idx] = off;
    struct timeval t2;
    gettimeofday(&t2, NULL);
    timestamps[idx] = t2.tv_sec*1e6 + t2.tv_usec;
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
    append_offset(offset);
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
    append_offset(offset);
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
    append_offset(0);
    ret = real_read_225(fd, buf, nbyte);
    return ret;
}

void handle_sigint(int sig)
{
    printf("Caught signal %d\n", sig);
    exit(0);
}

void writer_func(int sig) {
    fprintf(stderr, "Caught signal %d\n", sig);
    fprintf(stderr, "In desctructor!");
    FILE *f = fopen("/home/itarte/tame.csv", "w");
    int64_t num = atomic_read(&array_index);
    
    char *temp_string = (char*) malloc(1024 * sizeof(char));
    for (int i = 0 ; i < num ; i++) {
        sprintf(temp_string, "%llu, %llu\n", timestamps[i], offset[i]);
        fputs(temp_string, f);
    }

    fclose(f);

    fprintf(stderr, "Wrote %u elements\n", num);
    exit(0);
}

void __attribute__((constructor)) initialize(void) {
    reads_contructor();
    fprintf(stderr, "In constructor!");
    signal(SIGINT, writer_func);
    timestamps = (uint64_t*) malloc( 524288 * sizeof(uint64_t));
    if(!timestamps) {
        fprintf(stderr, "Can't allocate timestamps\n");
    }
    
    offset = (uint64_t*) malloc( 524288 * sizeof(uint64_t));
    if(!offset) {
        fprintf(stderr, "Can't allocate offset\n");
        free(timestamps);
    }
}

void __attribute__((destructor)) destruct(void) {
    // write the array to file
    fprintf(stderr, "In desctructor!");
    FILE *f = fopen("/home/itarte/test.csv", "w");
    int64_t num = atomic_read(&array_index);
    
    char *temp_string = (char*) malloc(1024 * sizeof(char));
    for (int i = 0 ; i < num ; i++) {
        sprintf(temp_string, "%llu, %llu\n", timestamps[i], offset[i]);
        fputs(temp_string, f);
    }

    fclose(f);

    fprintf(stderr, "Wrote %u elements\n", num);

    //free(timestamps);
    //free(offset);
    //free(temp_string);
}



__asm__(".symver pread_225, pread@GLIBC_2.2.5");
__asm__(".symver pread64_225, pread64@GLIBC_2.2.5");
__asm__(".symver read_225, read@GLIBC_2.2.5");