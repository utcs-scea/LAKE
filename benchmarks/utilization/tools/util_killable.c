#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>

struct cpustat {
    unsigned long t_user;
    unsigned long t_nice;
    unsigned long t_system;
    unsigned long t_idle;
    unsigned long t_iowait;
    unsigned long t_irq;
    unsigned long t_softirq;
};

void skip_lines(FILE *fp, int numlines)
{
    int cnt = 0;
    char ch;
    while((cnt < numlines) && ((ch = getc(fp)) != EOF))
    {
        if (ch == '\n')
            cnt++;
    }
    return;
}

void get_stats(struct cpustat *st, int cpunum)
{
    FILE *fp = fopen("/proc/stat", "r");
    int lskip = cpunum+1;
    skip_lines(fp, lskip);
    char cpun[255];
    fscanf(fp, "%s %ld %ld %ld %ld %ld %ld %ld", cpun, &(st->t_user), &(st->t_nice), 
        &(st->t_system), &(st->t_idle), &(st->t_iowait), &(st->t_irq),
        &(st->t_softirq));
    fclose(fp);
	return;
}

void print_stats(struct cpustat *st, char *name)
{
    printf("%s: %ld %ld %ld %ld %ld %ld %ld\n", name, (st->t_user), (st->t_nice), 
        (st->t_system), (st->t_idle), (st->t_iowait), (st->t_irq),
        (st->t_softirq));
}

double calculate_load(struct cpustat *prev, struct cpustat *cur)
{
    //int idle_prev = (prev->t_idle) + (prev->t_iowait);
    //int idle_cur = (cur->t_idle) + (cur->t_iowait);
    int idle_prev = (prev->t_idle) + (prev->t_user) + (prev->t_nice + (prev->t_iowait));
    int idle_cur = (cur->t_idle) + (cur->t_user) + (cur->t_nice) + (cur->t_iowait);

    //int nidle_prev = (prev->t_user) + (prev->t_nice) + (prev->t_system) + (prev->t_irq) + (prev->t_softirq);
    //int nidle_cur = (cur->t_user) + (cur->t_nice) + (cur->t_system) + (cur->t_irq) + (cur->t_softirq);
    int nidle_prev = (prev->t_system) + (prev->t_irq) + (prev->t_softirq); //+ (prev->t_iowait);
    int nidle_cur = (cur->t_system) + (cur->t_irq) + (cur->t_softirq); //+ (cur->t_iowait);
    
    int total_prev = idle_prev + nidle_prev;
    int total_cur = idle_cur + nidle_cur;

    double totald = (double) total_cur - (double) total_prev;
    double idled = (double) idle_cur - (double) idle_prev;
    double cpu_perc = (1000 * (totald - idled) / totald + 1) / 10;

    return cpu_perc;
}

double calculate_io(struct cpustat *prev, struct cpustat *cur)
{
    int idle_prev = (prev->t_idle) + (prev->t_user) + (prev->t_nice) +  + (prev->t_system) + (prev->t_irq) + (prev->t_softirq);
    int idle_cur = (cur->t_idle) + (cur->t_user) + (cur->t_nice) +  + (cur->t_system) + (cur->t_irq) + (cur->t_softirq);

    int nidle_prev = (prev->t_iowait); //+ (prev->t_iowait);
    int nidle_cur = (cur->t_iowait); //+ (cur->t_iowait);
    
    int total_prev = idle_prev + nidle_prev;
    int total_cur = idle_cur + nidle_cur;

    double totald = (double) total_cur - (double) total_prev;
    double idled = (double) idle_cur - (double) idle_prev;
    double cpu_perc = (1000 * (totald - idled) / totald + 1) / 10;

    return cpu_perc;
}

struct cpustat st0_0, st0_1;
void intHandler(int dummy) {
    get_stats(&st0_1, -1);
    printf("\nsys,%lf%%\n", calculate_load(&st0_0, &st0_1));
    printf("io,%lf%%\n", calculate_io(&st0_0, &st0_1));
    exit(0);
}

int main (void)
{
    signal(SIGINT, intHandler);
    get_stats(&st0_0, -1);
    //printf("sleeping\n");
    sleep(99999999);    
    return 0;
}
