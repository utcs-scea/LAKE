#include "common.h"

/*
 *  meta data about current mappings
 */
struct file_mapping {
    uint64_t start;
    uint64_t end;
    int fd;
};

typedef void *(*mmap_ptr)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);