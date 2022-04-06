#include "debug.h"
#include "test.h"
#include "shared_memory.h"

void test_alloc_and_free(allocator_t *allocator)
{
    int *a;
    void *b, *c, *d;
    size_t size = allocator->size;
    pr_info("[kshm-test] Shared memory start = %lx, size = %lx\n",
            (uintptr_t)allocator->start, size);

    a = (int *)kava_alloc(size / 3);
    if (!a) {
        pr_err("[kshm-test] Failed to allocate memory size = %lx\n", size / 3);
        return;
    }
    else {
        pr_info("[kshm-test] Allocate a new memory at %lx, size = %lx\n",
                (uintptr_t)a, size / 3);
    }
    *a = 12345;
    pr_info("[kshm-test] Read from memory at %lx = %d\n", (uintptr_t)a, *a);

    b = kava_alloc(size / 3);
    if (!b) {
        pr_err("[kshm-test] Failed to allocate memory size = %lx\n", size / 3);
        return;
    }
    else {
        pr_info("[kshm-test] Allocate a new memory at %lx, size = %lx\n",
                (uintptr_t)b, size / 3);
    }

    kava_free(b);
    pr_info("[kshm-test] Free memory at %lx\n", (uintptr_t)b);

    c = kava_alloc(size / 2);
    if (!c) {
        pr_err("[kshm-test] Failed to allocate memory size = %lx\n", size / 3);
        return;
    }
    else {
        pr_info("[kshm-test] Allocate a new memory at %lx, size = %lx\n", (uintptr_t)c,
                size / 2);
    }

    d = kava_alloc(size / 3);
    if (d) {
        pr_err("[kshm-test] Unexpected success on allocating memory size = %lx\n",
                size / 3);
        return;
    }
    else {
        pr_info("[kshm-test] Try to allocate a new memory at %lx, size = %lx\n", (uintptr_t)d,
                size / 3);
    }

    kava_free(a);
    pr_info("[kshm-test] Free memory at %lx\n", (uintptr_t)b);
    kava_free(c);
    pr_info("[kshm-test] Free memory at %lx\n", (uintptr_t)c);
}
