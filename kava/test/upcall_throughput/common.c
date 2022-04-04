#include <stdlib.h>
#include "upcall_impl.h"

void parse_input_args(int argc, char **argv, int *test_num, int *message_size) {
    if (argc != 3) {
        pr_err("Usage: %s <test_num> <message_size (KB)>\n",
                argv[0]);
        exit(0);
    }

    *test_num = atoi(argv[1]);
    *message_size = atoi(argv[2]) << 10;
}
