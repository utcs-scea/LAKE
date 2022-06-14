#include "reads.h"
#include "mmap.h"

void __attribute__((constructor)) initialize(void) {

    reads_contructor();
    //mmap_contructor();
}
