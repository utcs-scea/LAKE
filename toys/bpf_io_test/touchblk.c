#include <stdio.h>
#include <sys/stat.h> // For fstat()
#include <unistd.h>   // For lseek()
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//1mb = 256 blocks

int n = 2;
int read_size = 1;
int read_order[] = {30, 1024};

int main() {

    int fd = open("./dummy.dat",O_RDONLY);
    if (fd < 0) {
        printf("Open Failed");
        return 1;
    }

    char buf[read_size];
    for (int i = 0 ; i < n ; i++) {
        lseek(fd, read_order[i], SEEK_SET);
        int bytes_read = read(fd, buf, read_size);
        if (bytes_read <= 0) {
            printf("Did not read enough bytes\n");
            return -1;
        }
    }

    return 0;
}
