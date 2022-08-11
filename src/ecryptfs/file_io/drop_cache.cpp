#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>

// helper binary to drop cache as user

int main() {
    if (geteuid() != 0) {
        fprintf(stderr, "flush-cache: Not root\n");
        exit(EXIT_FAILURE);
    }

    FILE *fp = fopen("/proc/sys/vm/drop_caches", "w");
    if(!fp) {
        printf("error opening /proc/sys/vm/drop_caches\n");
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "3");
    fclose(fp);
    return 0;
}