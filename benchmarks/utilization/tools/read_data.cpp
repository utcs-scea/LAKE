#include <fstream>
#include <iostream>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

long GB = 1024*1024*1024;
long MB = 1024*1024;
long bsize = 2*MB;


//long size = 2 * GB;
long size = 512 * MB;

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("need path to file\n");
        exit(1);
    }

    int fd, ret;
    char* buf = (char *) malloc (2 * MB);
  
    fd = open(argv[1], O_RDONLY);
    if(fd == 0) {
        printf("Error!");   
        exit(1);             
    }

    //lseek(fd, SEEK_SET, 0);
    //read in 2MB chunks
    for(int i = 0; i < size; i+=bsize) { 
        ret = read(fd, buf, bsize);
        if (ret <= 0) {
            printf("error on read: %d  read %dMB\n", errno, i/MB);
            break;
        }
    }

   return 0;
}
