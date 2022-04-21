#include "consts.h"
#include <stddef.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <asm/fpu/api.h>

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
    int ret;

    struct matrix W1 = {NR_FEAT, 10, w1};
    struct matrix out1 = {1, 10, o1};
    struct matrix B1 = {1, 10, b1};
    struct matrix W2 = {10, 1, w2};
    struct matrix out2 = {1, 1, o2};
    struct matrix B2 = {1, 1, b2};

    kernel_fpu_begin();

    input->values[12] = input->values[12] / input->values[15];

    matmul(input, &W1, &out1);
    matadd(&out1, &B1, &out1);
    ReLU(&out1);
    matmul(&out1, &W2, &out2);
    matadd(&out2, &B2, &out2);
    output = out2.values[0];
    printk("forward_pass output: %08x", ftox(output));
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