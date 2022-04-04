#ifndef __KAVA_DEBUG_H__
#define __KAVA_DEBUG_H__

#ifndef __KERNEL__
#include <assert.h>
#include <stdio.h>
#endif

#ifndef KAVA_RELEASE
#define DEBUG
#else
#undef DEBUG
#endif

/* debug print */
#ifdef DEBUG
    #ifdef __KERNEL__
    #define DEBUG_PRINT(fmt, args...) printk(KERN_INFO fmt, ## args)
    #else
    #define DEBUG_PRINT(fmt, args...) fprintf(stderr, fmt, ## args)
    #endif
#else
    #define DEBUG_PRINT(fmt, args...)
#endif

#ifdef DEBUG
    #ifdef __KERNEL__
    #define DEBUG_PRINT_COMMAND(chan, cmd) kava_print_cmd(NULL, chan, cmd)
    #else
    #define DEBUG_PRINT_COMMAND(chan, cmd) kava_print_cmd(stderr, chan, cmd)
    #endif
#else
#define DEBUG_PRINT_COMMAND(chan, cmd)
#endif

#ifndef __KERNEL__

#ifndef pr_fmt
#define pr_fmt(fmt) fmt
#endif

#define pr_err(fmt, ...) \
        printf(pr_fmt(fmt), ##__VA_ARGS__)
#define pr_warning(fmt, ...) \
        printf(pr_fmt(fmt), ##__VA_ARGS__)
#define pr_warn pr_warning
#define pr_info(fmt, ...) \
        printf(pr_fmt(fmt), ##__VA_ARGS__)
#endif

#undef pr_fmt
#define pr_fmt(fmt) "[kava] %s:%d:: " fmt, __func__, __LINE__

/* Assertions */

#ifdef NDEBUG
#define abort_with_reason(reason) abort()
#else
#define abort_with_reason(reason) __assert_fail(reason, __FILE__, __LINE__, __FUNCTION__)
#endif
#define KAVA_CHECK_RET(code) if (G_UNLIKELY(!(code))) abort_with_reason("Function returned failure: " __STRING(code))

#ifndef KAVA_RELEASE
#define KAVA_DEBUG_ASSERT(code) assert(code)
#else
#define KAVA_DEBUG_ASSERT(code)
#endif

#define __INTERNAL_PRAGMA(t) _Pragma(#t)
#define __PRAGMA(t) __INTERNAL_PRAGMA(t)

/// A combined compile time warning and runtime abort_with_reason.
#define ABORT_TODO(t) __PRAGMA(GCC warning __STRING(TODO: t)) abort_with_reason(__STRING(TODO: t))

#endif
