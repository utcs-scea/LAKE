#include <linux/types.h>
#include <linux/cpumask.h>

#define NR_LOOPS 7
#define DUMPS_PER_LOOP 2
#define NR_PIDS  (NR_LOOPS * DUMPS_PER_LOOP)  

#define NR_NODES 2

struct migrate_data {
    u64 ts;
    u64 instance_ts;
    int cpu;
    int dst_cpu;
    int src_cpu;
    int pid;
    int curr_pid;
    long imbalance;
    int numa_preferred_nid;
    unsigned int src_nr_running;
    unsigned int src_nr_numa_running;
    unsigned int src_nr_preferred_running;
    unsigned int dst_nr_running;
    unsigned long src_load;
    unsigned long dst_load;
    int cpu_idle;
    int cpu_not_idle;
    int cpu_newly_idle;
    int buddy_hot;
    int p_running;
    int throttled;
    unsigned int nr_balance_failed;
    unsigned int cache_nice_tries;
    s64 delta;
    u64 p_numa_faults[2];
    unsigned long total_numa_faults;
    int can_migrate;
    unsigned int test_aggressive;
    u64 perf_count_0;
    u64 perf_count_1;
};

struct can_migrate_context {
    u64 ts;
    int cpu;
    struct task_struct *p;
    struct lb_env *env;
    struct migrate_data data;
};


/* NOT USED */
/* for runqueue dump */
struct rq_data {
    u64 ts;
    u64 instance_ts;
    int cpu;
    int curr_pid;
    char comm[TASK_COMM_LEN];
    unsigned int h_nr_running;
    unsigned long runnable_weight;
    int pid_cnt;
    int pids[NR_PIDS];
    unsigned long weights[NR_PIDS];
};
/* for context switch */
struct cs_data {
    u64 ts;
    int cpu;
    int p_pid;
    int n_pid;
    long prev_state;
    char comm[TASK_COMM_LEN];
    unsigned int len;
    int curr_pid;
    unsigned long runnable_weight;
    int pid_cnt;
    int pids[NR_PIDS];
    unsigned long weights[NR_PIDS];
};

struct tsk_data {
    u64 ts;
    int cpu;
    int pid;
    char comm[TASK_COMM_LEN];
    int prio;
    unsigned long runnable_weight;
    char oldcomm[TASK_COMM_LEN];
};
