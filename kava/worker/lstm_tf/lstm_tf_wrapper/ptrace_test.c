#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/user.h>
#include <sys/reg.h>
#include <stdio.h>
#include <sys/time.h>
/* #include <linux/user.h>   /1* For constants */

#define ELAPSED_TIME_MICRO_SEC(start, stop) ((stop.tv_sec - start.tv_sec) * 1000000 + (stop.tv_usec - start.tv_usec))

int main()
{   pid_t child;
    /* long orig_eax; */
    child = fork();
    if(child == 0) {
        ptrace(PTRACE_TRACEME, 0, NULL, NULL);
        execl("/bin/ls", "ls", NULL);
        /* printf("gg"); */
    }
    else {
        struct user_regs_struct regs;
        int i = 0;
        struct timeval micro_start, micro_stop;
        long total_time = 0;
        const int it = 20;
        long long test_array[it];
        gettimeofday(&micro_start, NULL);
        for (;i < it ;i++) {
            ptrace(PTRACE_SYSCALL, child, 0, 0);
            waitpid(child, 0, 0);

            /* wait(NULL); */
            /* orig_eax = ptrace(PTRACE_PEEKUSER, */
            /*                   child, 4 * ORIG_RAX, */
            /*                   NULL); */
            ptrace(PTRACE_GETREGS, child, 0, &regs);
            /* printf("The child made a " */
            /*        "system call %ld\n", orig_eax); */
            test_array[i] = regs.orig_rax;
            /* printf("The child made a " */
            /*         "syscall %lld\n", regs.orig_rax); */
            /* ptrace(PTRACE_CONT, child, NULL, NULL); */
            /* ptrace(PTRACE_SYSCALL, child, 0, 0); */

        }
        gettimeofday(&micro_stop, NULL);
        total_time += ELAPSED_TIME_MICRO_SEC(micro_start, micro_stop);
        printf("total tracing time %ld\n", total_time);
    }
    return 0;
}
