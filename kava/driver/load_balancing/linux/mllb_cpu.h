#ifndef _MLLB_CPU_H_
#define _MLLB_CPU_H_

#include "mllb_common.h"

int can_migrate_task_mllb_cpu(struct task_struct *p, struct lb_env *env);
int can_migrate_task_both_cpu(struct task_struct *p, struct lb_env *env);

#endif