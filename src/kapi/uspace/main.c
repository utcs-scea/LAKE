#include <stdio.h>
#include <unistd.h>
#include "lake_kapi.h"

int main() {
    printf("Starting uspace lake kapi with pid %d\n", getpid());
    lake_init_socket();


    lake_send_cmd(0);


    lake_destroy_socket();
}