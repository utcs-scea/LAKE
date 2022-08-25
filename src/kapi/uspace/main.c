#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include "lake_kapi.h"

volatile sig_atomic_t stop_running = 0;

void exit_handler(int dummy) {
    stop_running = 1;
    //TODO: sock is blocking, so it never quits the loop, just quit here
    sleep(1);
    lake_destroy_socket();
    lake_shm_fini();
    exit(0);
}

int main() {
    signal(SIGINT, exit_handler);
    printf("Starting uspace lake kapi with pid %d\n", getpid());
    lake_init_socket();
    lake_shm_init();

    while(!stop_running) {
        lake_recv();
    }
    printf("Quitting\n");
    //lake_destroy_socket();
    return 0;
}

