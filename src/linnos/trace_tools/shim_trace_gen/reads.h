#include <stdint.h>

typedef ssize_t (*read_ptr)(int fildes, void *buf, size_t nbyte);
typedef ssize_t (*pread_ptr)(int fd, void *buf, size_t count, off_t offset); 
typedef ssize_t (*pwrite_ptr)(int fd, const void *buf, size_t count, off_t offset); 

void reads_contructor();

extern read_ptr real_read_225;
extern pread_ptr real_pread_225;
extern pread_ptr real_pread64_225;

extern pwrite_ptr real_pwrite_225;
extern pwrite_ptr real_pwrite64_225;


ssize_t read_225(int fd, void *buf, size_t nbyte);
ssize_t pread_225(int fd, void *buf, size_t nbyte, off_t offset);
ssize_t pread64_225(int fd, void *buf, size_t count, off_t offset);

ssize_t pwrite_225(int fd, const void *buf, size_t count, off_t offset);
ssize_t pwrite64_225(int fd, const void *buf, size_t count, off_t offset);

