#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <bcc/proto.h>
#include <linux/bpf.h>
#include <linux/kernel.h>
#include <linux/sched/topology.h>
#include <linux/percpu.h>
#include <linux/sched/mm.h>
#include <linux/percpu.h>
#include <linux/bitmap.h>
#include <linux/lockdep.h>
#include <linux/migrate.h>
#include <linux/mm.h>
#include <linux/smp.h>
#include <linux/percpu-defs.h>
#include <uapi/linux/bpf.h>
#include <linux/jump_label.h>
#include <linux/nodemask.h>
#include <linux/sched/idle.h>

#include "sched.h"
#include "dump_lb.h"


/* ====== NUMA related ====== */
/* Shared or private faults. */
#define NR_NUMA_HINT_FAULT_TYPES 2

/* Memory and CPU locality */
#define NR_NUMA_HINT_FAULT_STATS (NR_NUMA_HINT_FAULT_TYPES * 2)

/* Averaged statistics, and temporary buffers. */
#define NR_NUMA_HINT_FAULT_BUCKETS (NR_NUMA_HINT_FAULT_STATS * 2)


#undef offsetof
#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)
#undef container_of
#define container_of(ptr, type, member) ({				\
    ((type *)((void *)(ptr) - offsetof(type, member))); })


#define KPROBE(fn)  _KPROBE(fn)
#define _KPROBE(fn) kprobe__##fn
#define KRETPROBE(fn)  _KRETPROBE(fn)
#define _KRETPROBE(fn) kretprobe__##fn


static inline bool cfs_bandwidth_used(void)
{
    /* return static_key_false(&__cfs_bandwidth_used); */
    return true;
}

/* check whether cfs_rq, or any parent, is throttled */
static inline int throttled_hierarchy(struct cfs_rq *cfs_rq)
{
    return cfs_bandwidth_used() && cfs_rq->throttle_count;
}

/*
 * Ensure that neither of the group entities corresponding to src_cpu or
 * dest_cpu are members of a throttled hierarchy when performing group
 * load-balance operations.
 */
static inline int throttled_lb_pair(struct task_group *tg,
				    int src_cpu, int dest_cpu)
{
	struct cfs_rq *src_cfs_rq, *dest_cfs_rq;

	src_cfs_rq = tg->cfs_rq[src_cpu];
	dest_cfs_rq = tg->cfs_rq[dest_cpu];

	return throttled_hierarchy(src_cfs_rq) ||
	       throttled_hierarchy(dest_cfs_rq);
}

static inline int task_faults_idx(enum numa_faults_stats s, int nid, int priv)
{
	return NR_NUMA_HINT_FAULT_TYPES * (s * nr_node_ids + nid) + priv;
}

static inline unsigned long group_faults(struct task_struct *p, int nid)
{
	if (!p->numa_group)
		return 0;

	return *(p->numa_group->faults + task_faults_idx(NUMA_MEM, nid, 0)) +
		*(p->numa_group->faults + task_faults_idx(NUMA_MEM, nid, 1));
}

static inline unsigned long task_faults(struct task_struct *p, int nid)
{
	if (!p->numa_faults)
		return 0;

	return *(p->numa_faults + task_faults_idx(NUMA_MEM, nid, 0)) +
		*(p->numa_faults + task_faults_idx(NUMA_MEM, nid, 1));
}

/* runqueue on which this entity is (to be) queued */
static inline struct cfs_rq *cfs_rq_of(struct sched_entity *se)
{
	return se->cfs_rq;
}

static inline u64 rq_clock_task(struct rq *rq)
{
	lockdep_assert_held(&rq->lock);

	return rq->clock_task;
}

struct pid_cpu_key {
    int pid;
    int cpu;
};


BPF_HASH(can_migrate_instances, struct pid_cpu_key, struct can_migrate_context);
BPF_PERF_OUTPUT(can_migrate_events);


struct lb_env {
	struct sched_domain	*sd;

	struct rq		*src_rq;
	int			src_cpu;

	int			dst_cpu;
	struct rq		*dst_rq;

	struct cpumask		*dst_grpmask;
	int			new_dst_cpu;
	enum cpu_idle_type	idle;
	long			imbalance;
	/* The set of CPUs under consideration for load-balancing */
	struct cpumask		*cpus;

	unsigned int		flags;

	unsigned int		loop;
	unsigned int		loop_break;
	unsigned int		loop_max;

	enum fbq_type		fbq_type;
	struct list_head	tasks;
};


int KPROBE(can_migrate_task) (struct pt_regs *ctx, struct task_struct *p, struct lb_env *env)
{
    struct migrate_data data = {};
    struct rq *src_rq = env->src_rq;
    struct rq *dst_rq = env->dst_rq;
    struct can_migrate_context context = {};
    int idle = env->idle;
    int cpu = bpf_get_smp_processor_id();
    u32 pid = bpf_get_current_pid_tgid();
    u64 ts = bpf_ktime_get_ns();

    struct pid_cpu_key key = {pid, cpu};

    data.throttled = (throttled_lb_pair(p->sched_task_group, env->src_cpu, env->dst_cpu));

    data.p_running = p->on_cpu;

    data.ts = ts;

    data.src_cpu = env->src_cpu;
    data.dst_cpu = env->dst_cpu;
    data.imbalance = env->imbalance;
    data.cpu_idle = idle == CPU_IDLE;
    data.cpu_not_idle = idle == CPU_NOT_IDLE;
    data.cpu_newly_idle = idle == CPU_NEWLY_IDLE;

    data.curr_pid = pid;
    data.pid = p->pid;
    data.numa_preferred_nid = p->numa_preferred_nid;
    /* data.p_policy = p->policy; */
	/* data.fair_class = (p->policy == SCHED_NORMAL || p->policy == SCHED_BATCH); */

    data.src_nr_running = src_rq->nr_running;
    data.src_nr_numa_running = src_rq->nr_numa_running;
    data.src_nr_preferred_running = src_rq->nr_preferred_running;

    data.dst_nr_running = dst_rq->nr_running;
    data.src_load = src_rq->cfs.avg.load_avg;
    data.dst_load = dst_rq->cfs.avg.load_avg;

	data.delta = rq_clock_task(env->src_rq) - p->se.exec_start;

    data.nr_balance_failed = env->sd->nr_balance_failed;
    data.cache_nice_tries = env->sd->cache_nice_tries;

    if (dst_rq->nr_running && (&p->se == cfs_rq_of(&p->se)->next || &p->se == cfs_rq_of(&p->se)->last))
        data.buddy_hot = 1;
    else
        data.buddy_hot = 0;

    int n;
    for (n = 0; n < NR_NODES; n++) {
        data.p_numa_faults[n] = task_faults(p, n);
    }
    data.total_numa_faults = p->total_numa_faults;

    /* data.f_test[0] = (unsigned long)*(p->numa_faults + 1); */
    /* data.f_test[1] = (unsigned long)(p->numa_faults[1]); */
    /* data.f_test[2] = (unsigned long)(p->total_numa_faults); */
    /* data.f_test[3] = (unsigned long)(p->numa_faults[0]); */

    data.perf_count_0 = 0;
    data.perf_count_1 = 0;

    context.ts = ts;
    context.cpu = cpu;
    context.p = p;
    context.env = env;
    context.data = data;
    
    /* can_migrate_instances.update(&pid, &context); */
    can_migrate_instances.update(&key, &context);

    return 0;
}

int KRETPROBE(can_migrate_task) (struct pt_regs *ctx)
{
    struct task_struct *p;
    struct lb_env *env;
    struct migrate_data *data_p;
    struct can_migrate_context *context;
    int cpu = bpf_get_smp_processor_id();
    u32 pid = bpf_get_current_pid_tgid();
    int ret = PT_REGS_RC(ctx);

    struct pid_cpu_key key = {pid, cpu};

    context = (struct can_migrate_context *)can_migrate_instances.lookup(&key);
    if (!context)
        return 0;

    data_p = &context->data;
    env = context->env;

    data_p->test_aggressive = 1;    // NO test flag in old kernel

    data_p->can_migrate = ret;
    can_migrate_events.perf_submit(ctx, data_p, sizeof(*data_p));

    can_migrate_instances.delete(&key);
    return 0;
}


/* Dumps rq data including tasks */
/* NOT USED */
static void dump_rq(struct rq_data *data, struct rq *rq) {

    struct cfs_rq *cfs;
    struct task_struct *curr_task;
    struct list_head *head, *pos;
    int i;
    struct task_struct *task;

    cfs = &rq->cfs;
    curr_task = rq->curr;
    data->curr_pid = curr_task->pid;
    data->cpu = rq->cpu;
    bpf_probe_read_str(&(data->comm), sizeof(data->comm), curr_task->comm);
    data->h_nr_running = cfs->h_nr_running;
    data->runnable_weight = cfs->runnable_weight;
    data->pid_cnt = 0;

    head = &(rq->cfs_tasks);
    pos = head;
    int j = 0;
    for (i = 0; i < NR_LOOPS; i++) {
        pos = pos->next;
        if (pos == head)
            break;
        task = container_of(pos, struct task_struct, se.group_node);
        data->pids[j] = task->pid;
        j++;
        pos = pos->next;
        if (pos == head)
            break;
        task = container_of(pos, struct task_struct, se.group_node);
        data->pids[j] = task->pid;
        j++;
    }
    data->pid_cnt = j;
}
