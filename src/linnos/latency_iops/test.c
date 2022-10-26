#include<stdio.h>
#include <fcntl.h>

#define DEVICE_NODE "/dev/vdc"

int main ()
{ 
    int fd = open(DEVICE_NODE, O_RDWR);
    if(fd < 0) {
        printf("Cannot open %s error : %d\n", DEVICE_NODE, fd);
        exit(1);
    }
}