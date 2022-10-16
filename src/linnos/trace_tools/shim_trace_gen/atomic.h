#ifndef __SHIM_ATOMIC_H
#define __SHIM_ATOMIC_H

#include <stdint.h>

inline int64_t atomic_read(int64_t* ptr) {
    return __atomic_load_n (ptr,  __ATOMIC_SEQ_CST);
}

inline void atomic_add(int64_t* ptr, int val) {
    __atomic_add_fetch(ptr, val, __ATOMIC_SEQ_CST);
}

inline int64_t atomic_fetch_inc(int64_t* ptr) {
    return __atomic_fetch_add(ptr, 1, __ATOMIC_SEQ_CST);
}


#endif