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
    //system("./drop_cache");
}

long GB = 1024*1024*1024;
long MB = 1024*1024;
long size = 2*GB;

int main()
{
    int fd;
    fd = open("temp.dat", O_RDWR, O_CREAT | O_SYNC  | O_TRUNC);

    if(fd == 0) {
        printf("Error!");   
        exit(1);             
    }
    char* buf = (char *) malloc (2 * MB);
    char *data = (char *) malloc (128 * MB);
    ssize_t ret;
    for(int i = 0; i < size; i+=128*MB) {
        ret = write(fd, data, 128*MB);
        if (ret <= 0) {
            printf("error on write: %d\n", errno);
            exit(1);
        }
    }

    close(fd);
    free(data);
    dropCache();

    fd = open("temp.dat", O_RDONLY);
    //read in 2MB chunks
    for(int i = 0; i < size; i+=2*MB) { 
        ret = read(fd, buf, 2*MB);
        if (ret <= 0) {
            printf("error on write: %d\n", errno);
            exit(1);
        }
    }

   return 0;
}
