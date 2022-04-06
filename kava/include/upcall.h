#ifndef __KAVA_UPCALL_H__
#define __KAVA_UPCALL_H__

#ifdef __KERNEL__

#include <linux/types.h>

/**
 * Upcaller
 */

typedef struct upcall_handle* upcall_handle_t;

/**
 * do_upcall - Issue an upcall to user-space process
 * @handle: internal upcall handle
 * @r0: first scalar argument
 * @r1: second scalar argument
 * @r2: third scalar argument
 * @r3: fourth scalar argument
 * @buf: bulk data buffer
 * @size: size of the bulk data
 */
extern void _do_upcall(upcall_handle_t handle,
                    uint64_t r0, uint64_t r1, uint64_t r2, uint64_t r3,
                    void *buf, size_t size);

#define do_upcall4(handle, r0, r1, r2, r3, buf, size) _do_upcall(handle, r0, r1, r2, r3, buf, size)
#define do_upcall3(handle, r0, r1, r2, buf, size)     do_upcall4(handle, r0, r1, r2, 0, buf, size)
#define do_upcall2(handle, r0, r1, buf, size)         do_upcall3(handle, r0, r1, 0, buf, size)
#define do_upcall1(handle, r0, buf, size)             do_upcall2(handle, r0, 0, buf, size)
#define do_upcall0(handle, buf, size)                 do_upcall1(handle, 0, buf, size)
#define do_upcall(handle)                             do_upcall0(handle, NULL, 0)

/**
 * init_upcall - Initialize upcall interfaces and data structures
 *
 * This function may be blocking for the connection from user-space process.
 * It returns an internal handle struct containing necessary data.
 */
upcall_handle_t init_upcall(void);

/**
 * close_upcall - Close upcall channel
 * @handle: internal upcall handle
 */
void close_upcall(upcall_handle_t handle);

#else

#include <sys/types.h>

/**
 * Upcallee
 */

/**
 * init_upcall - Initialize upcall interfaces and data structures
 *
 * It returns an file descriptor representing the upcall channel.
 */
int init_upcall(void);

/**
 * wait_upcall - Wait notification from kernel space
 * @fd: file descriptor of opened upcall channel
 * @buf: the address of a buffer pointer. The receive buffer will be allocated
 * if the buffer pointer is NULL
 * @size: the size of the received buffer. Cannot be NULL
 *
 * This function is always blocking and returns when a notification arrives.
 */
void wait_upcall(int fd, void **buf, size_t *size);

/**
 * close_upcall - Close upcall channel
 * @fd: file descriptor of opened upcall channel
 */
void close_upcall(int fd);

#endif

#endif
