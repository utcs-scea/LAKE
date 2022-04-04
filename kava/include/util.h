#ifndef __KAVA_UTIL_H__
#define __KAVA_UTIL_H__


#ifdef __KERNEL__

#include <linux/types.h>
typedef __s64 intptr_t;

#else

#include <stdint.h>
#include <glib.h>
#include <glib/ghash.h>
#include <gmodule.h>

#endif

#ifndef TRUE
  #define TRUE 1
  #define FALSE 0
#endif

/**
 * These macros to be used to resolve unused and unreferenced compiler warnings.
 */
#define UNREFERENCED_PARAMETER(_Parameter_) (_Parameter_)
#define UNUSED_VARIABLE(_x_) UNREFERENCED_PARAMETER(_x_)

/**
 * Linkage.
 */
#ifdef __cplusplus
extern "C" {
#endif

#define EXPORTED_WEAKLY __attribute__ ((visibility ("default"))) __attribute__ ((weak))
#define EXPORTED __attribute__ ((visibility ("default")))

#ifdef __cplusplus
}
#endif

/**
 * Array macros.
 */
#define ARRAY_COUNT(_Array_) (sizeof(_Array_)/sizeof(_Array_[0]))

/**
 * Terminal colors.
 */
#define KNRM   "\x1B[0m"
#define KRED   "\x1B[31m"
#define KGRN   "\x1B[32m"
#define KYEL   "\x1B[33m"
#define KBLU   "\x1B[34m"
#define KMAG   "\x1B[35m"
#define KCYN   "\x1B[36m"
#define KWHT   "\x1B[37m"
#define KRESET "\x1B[0m"

/*
 * Generic macro wrappers (HT_MACRO_START, HT_MACRO_END)
 *
 * N.B. These macros must be structured as a do/while statement rather than
 * as a block (i.e. just {}) because the compiler can not deal properly
 * with if/else statements of the form:
 * if (...)
 *     MACRO_USING_HT_MACRO_START
 * else
 *     ...
 */
#define HT_MACRO_START do {
#define HT_MACRO_END   } while (0)

/*
 * Macros for error propagation.
 *
 * HT_CHK(_Status_) - Evaluates the _Status_ expression once on all builds.
 *                    If the resulting value is a failure code, goto Cleanup.
 *
 * HT_ERR(_Status_) - Evaluates the _Status_ expression once on all builds.
 *                    Always goes to Cleanup.
 *
 * HT_EXIT() - Goto Cleanup.
 */
#define HT_CHK(_Status_)    HT_MACRO_START if (HT_FAILURE(_Status_)) HT_EXIT(); HT_MACRO_END
#define HT_EXIT()           HT_MACRO_START goto Cleanup; HT_MACRO_END
#define HT_ERR(_Status_)    HT_MACRO_START _Status_; HT_EXIT(); HT_MACRO_END

/**
 * Units.
 */
#define KB(x) (x << 10)
#define MB(x) ((KB(x)) << 10)
#define GB(x) ((MB(x)) << 10)

#define US_TO_US(x)  ((long)x)
#define MS_TO_US(x)  (US_TO_US(x) * 1000L)
#define SEC_TO_US(x) (MS_TO_US(x) * 1000L)

/**
 * Hash functions.
 */
#ifndef __KERNEL__

void MurmurHash3_x86_32(const void *key, int len, uint32_t seed, void *out);

void MurmurHash3_x86_128(const void *key, int len, uint32_t seed, void *out);

void MurmurHash3_x64_128(const void *key, int len, uint32_t seed, void *out);

__attribute__ ((const))
guint kava_hash_mix64variant13(gconstpointer ptr);

__attribute__ ((pure))
static inline guint kava_hash_struct(gconstpointer ptr, int size) {
    guint ret;
    MurmurHash3_x86_32(ptr, size, 0xfbcdabc7 + size, &ret);
    return ret;
}

__attribute__ ((const))
static inline guint kava_hash_pointer(gconstpointer ptr) {
    return kava_hash_mix64variant13(ptr);
}

static inline uint64_t rdtsc(void) {
    unsigned int lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

#endif

#endif
