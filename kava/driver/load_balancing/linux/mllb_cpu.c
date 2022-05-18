#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <asm/fpu/api.h>
#include <linux/ktime.h>

#include "consts.h"
#include "mllb_common.h"

#define _ReLU(x)        (x > 0 ?  x : 0)
#define ftox(f)         (*(unsigned *)&((float){f}))

struct matrix {
    int nrow;
    int ncol;
    float *values;
};

static int matmul(struct matrix *X, struct matrix *Y, struct matrix *Z) {
    int i, j, k;
    int ij, ik, kj;
    for(i = 0; i < X->nrow; i++)
        for(j = 0; j < Y->ncol; j++)
            for(k = 0; k < X->ncol; k++) {
                ij = i*Z->ncol + j;
                ik = i*X->ncol + k;
                kj = k*Y->ncol + j;
                Z->values[ij] += (X->values[ik] * Y->values[kj]);
            }
    return 0;
}

static int matadd(struct matrix *X, struct matrix *Y, struct matrix *Z) {
    int i;
    for (i = 0; i < X->nrow * X->ncol; i++) {
        Z->values[i] = X->values[i] + Y->values[i];
    }
    return 0;
}

static void ReLU(struct matrix *X) {
    int i;
    for (i = 0; i < X->nrow * X->ncol; i++) {
        X->values[i] = _ReLU(X->values[i]);
    }
}

static int forward_pass(struct matrix *input) {
    float output;
    float o1[10] = {0};
    float o2[10] = {0};
    int ret, i, temp;

    struct matrix W1 = {NR_FEAT, 10, w1};
    struct matrix out1 = {1, 10, o1};
    struct matrix B1 = {1, 10, b1};
    struct matrix W2 = {10, 1, w2};
    struct matrix out2 = {1, 1, o2};
    struct matrix B2 = {1, 1, b2};

    kernel_fpu_begin();

    //printk("feats [%d]: ", NR_FEAT);
    for (i = 0 ; i < NR_FEAT ; i++) {
        if (i==12) 
            input->values[12] = input->values[12] / input->values[15];
        else {
            temp = *((int*)&(input->values[i]));
            input->values[i] = (float) temp;
        }
        //printk("%08x, ", ftox(input->values[i]));
    }

    matmul(input, &W1, &out1);
    matadd(&out1, &B1, &out1);
    ReLU(&out1);
    matmul(&out1, &W2, &out2);
    matadd(&out2, &B2, &out2);
    output = out2.values[0];
    //printk("forward_pass output: %08x", ftox(output));
    ret = output > 0.5 ? 1 : 0;

    kernel_fpu_end();

    return ret;
}

int can_migrate_task_mllb_cpu(struct task_struct *p, struct lb_env *env) {
    //plus one because we get an extra value from the kernel and need to do one float op
    float input[NR_FEAT+1];
    struct matrix m;
    m.nrow = 1;
    m.ncol = NR_FEAT;
    m.values = input;

    hack_mllb_lb_struct_to_feature_vec(p, env, (int*)input);

    return forward_pass(&m);
}
EXPORT_SYMBOL(can_migrate_task_mllb_cpu);

static int count = 0;
static int print_at = 200;
u64 mllb_sum, linux_sum;

int can_migrate_task_both_cpu(struct task_struct *p, struct lb_env *env) {
    u64 mllb_st, mllb_end, linux_st, linux_end;
    int ret, linux_ret;
    //plus one because we get an extra value from the kernel and need to do one float op
    float input[NR_FEAT+1];
    struct matrix m;
    mllb_st = ktime_get_ns();
    m.nrow = 1;
    m.ncol = NR_FEAT;
    m.values = input;
    hack_mllb_lb_struct_to_feature_vec(p, env, (int*)input);
    ret = forward_pass(&m);
    mllb_end = ktime_get_ns();

    linux_st = ktime_get_ns();
    linux_ret = can_migrate_task_linux(p, env);
    linux_end = ktime_get_ns();
    
    mllb_sum = mllb_end - mllb_st;
    linux_sum = linux_end - linux_st;

    if (++count == print_at) {
        count = 0;
        printk("linux avg, mllb_avg\n");
        printk("%lld, %lld\n", linux_sum/print_at, mllb_sum/print_at);
    }

    printk("returns: linux=%d  mllb=%d\n", linux_ret, ret);

    return linux_ret;
}

EXPORT_SYMBOL(can_migrate_task_both_cpu);