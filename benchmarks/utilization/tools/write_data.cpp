#include <fstream>
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

void dropCache()
{
    FILE *fp = fopen("/proc/sys/vm/drop_caches", "w");
    fprintf(fp, "3");
    fclose(fp);
}

long GB = 1024*1024*1024;
long MB = 1024*1024;
long bsize = 512*MB;

//long size = 2 * GB;
long size = 256 * MB;

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("need path to file\n");
        exit(1);
    }

    int fd;
    fd = open(argv[1], O_RDWR | O_CREAT| O_TRUNC, S_IROTH | S_IWOTH);
    if(fd == 0) {
        printf("Error!");   
        exit(1);             
    }
    
    char *data = (char *) malloc (bsize);
    ssize_t ret;
    for(int i = 0; i <= size; i+= bsize) {
        ret = write(fd, data, bsize);
        if (ret <= 0) {
            printf("error on write: %d\n", errno);
            exit(1);
        }
    }

    close(fd);
    free(data);
    dropCache();
    return 0;
}
