#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/mman.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#include "debug.h"
#include "channel_user.h"
#include "command_handler.h"
#include "worker.h"

struct kava_chan *chan;
int log_fd;

__sighandler_t original_sigint_handler = SIG_DFL;

void sigint_handler(int signo)
{
    if (chan)
        chan->chan_free(chan);
    if (log_fd > 0)
        close(log_fd);
    signal(signo, original_sigint_handler);
    raise(signo);
}

static struct kava_chan *channel_create()
{
    return chan;
}

#define REDIRECT_LOG 0

int main(int argc, char *argv[])
{
    cpu_set_t cpu_mask;
    CPU_ZERO(&cpu_mask);
    CPU_SET(7, &cpu_mask);
    CPU_SET(8, &cpu_mask);
    sched_setaffinity(0, sizeof(cpu_mask), &cpu_mask);

    enable_constructor();

#if REDIRECT_LOG
    /* Redirect to temporary log file */
    char log_file[128];
    log_fd = open("/tmp/kava_worker.log", O_WRONLY | O_TRUNC | O_CREAT, 0666);
    dup2(log_fd, STDOUT_FILENO);
    dup2(log_fd, STDERR_FILENO);
#endif

    if (argc != 3) {
        printf("Usage: %s <api_name> <dev_name>\n", argv[0]);
        return 0;
    }
    char *api_name = argv[1];
    char *dev_name = argv[2];


    /* setup signal handler */
    if ((original_sigint_handler = signal(SIGINT, sigint_handler)) == SIG_ERR)
        printf("failed to catch SIGINT\n");

    /**
     * Read configurations:
     *   1. channel mode
     */

    int fd;
    char sysfs_path[128], config[128];
    int bytes;

    /* Remove trailing whitespace */
    pr_info("API name: %s\n", api_name);

    /* Read channel mode */
    sprintf(sysfs_path, "/sys/kernel/kava_%s/channel", api_name);
    fd = open(sysfs_path, O_RDONLY);
    bytes = read(fd, config, 128);
    config[bytes] = '\0';
    close(fd);
    if (bytes > 0) {
        pr_info("Read channel mode: %s\n", config);
    }
    else {
        pr_info("Read channel mode failed: (%s, %d)\n", sysfs_path, bytes);
        return 0;
    }

    /* Create channel */
    if (!strcmp(config, kava_chan_name[KAVA_CHAN_FILE_POLL])) {
        chan = kava_chan_file_poll_new(dev_name);
    }
    else if (!strcmp(config, kava_chan_name[KAVA_CHAN_NL_SOCKET])) {
        chan = kava_chan_nl_socket_new(dev_name);
    }
    else {
        pr_err("Unsupported channel mode: %s\n", config);
        return 0;
    }

    if (!chan) {
        pr_err("Failed to create command channel\n");
        exit(EXIT_FAILURE);
    }

    kava_init_internal_cmd_handler();
    pr_info("Start polling commands\n");
    kava_init_cmd_handler_inline(channel_create);
    //kava_init_cmd_handler(channel_create);
    //kava_wait_for_cmd_handler();

    assert(!"Should never reach this end");

    return 0;
}

void worker_common_init(void)
{
    kava_shm_init();
}

void worker_common_fini(void)
{
    kava_shm_fini();
}
