
#include <linux/atomic.h>
#include <linux/vmalloc.h>
#include <linux/fs.h>
#include <asm/segment.h>
#include <asm/uaccess.h>
#include <linux/buffer_head.h>
#include "queue_depth.h"

typedef void (*append_qdepth_fn_type)(u32);
extern append_qdepth_fn_type append_qdepth_fn;

//this is 4MB for timestamps and 1MB for qds
#define MAX_ENTRIES 524288
//#define MAX_ENTRIES 4096


u64 *timestamps;
u16 *qds;
static atomic_t qd_index = ATOMIC_INIT(0);
char *temp_string;

int qd_init(void) {
    timestamps = vmalloc(MAX_ENTRIES*sizeof(u64));
    if(!timestamps) {
        pr_warn("Can't allocate timestamps\n");
        return -2;
    }
    
    qds = vmalloc(MAX_ENTRIES*sizeof(u32));
    if(!qds) {
        pr_warn("Can't allocate qds\n");
        vfree(timestamps);
        return -2;
    }

    append_qdepth_fn = append_qdepth;

    return 0;
}

void append_qdepth(u32 queue_depth) {
    //fail fast
    int idx = atomic_read(&qd_index);
    if(idx >= MAX_ENTRIES-1)
        return;

    //add one and get our index, need to check bounds again
    idx = atomic_inc_return(&qd_index) - 1; //we inc before fetch
    if(idx >= MAX_ENTRIES-1)
        return;

    qds[idx] = queue_depth;
    u64 now = ktime_get_ns()/1000;
    timestamps[idx] = now;
}

void qd_writeout(void) {
    struct file* fp = filp_open("/disk/hfingler/qdepth/qd.txt", O_CREAT | O_WRONLY, S_IRWXU);
    u32 n = atomic_read(&qd_index);
    u32 i;
    u64 offset = 0;

    append_qdepth_fn = 0;

    if(IS_ERR(fp)) {
        pr_warn("filp_open error!!.\n");
        return;
    }

    temp_string = vmalloc(1024);

    for (i=0 ; i < n ; i++) {
        sprintf(temp_string, "%llu, %u\n", timestamps[i], qds[i]);
        kernel_write(fp, temp_string, strlen(temp_string), &offset);
    }

    pr_warn("Wrote %u elements\n", n);

    vfree(timestamps);
    vfree(qds);
    vfree(temp_string);
}



