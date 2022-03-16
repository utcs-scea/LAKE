



Kernel code that uses load balancing:

https://github.com/torvalds/linux/blob/ac5a9bb6b4fa22135b3e371ac9787de120e18c8d/kernel/sched/fair.c#L7877


struct list_head *tasks = &env->src_rq->cfs_tasks;
...
while (!list_empty(tasks)) {
    ...
    if (!can_migrate_task(p, env))
			goto next;