#define _GNU_SOURCE

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <assert.h>
#include <inttypes.h>
#include <linux/fs.h>
#include <sys/ioctl.h>

#include "globals.h"
#include "io_ops.h"


int LARGEST_REQUEST_SIZE = (16*1024*1024); //blocks
int MEM_ALIGN = 4096*8; //bytes
int nr_workers[NR_DEVICE] = {16, 16, 16};
int printlatency = 1; //print every io latency
int respecttime = 1;
int block_size = 1; // by default, one sector (512 bytes)
int single_io_limit = (1024*1024); // the size limit of a single IOs request, used to break down large IOs

// int LARGEST_REQUEST_SIZE = (16*1024*1024); //blocks
// int MEM_ALIGN = 4096*8; //bytes
// int nr_workers[NR_DEVICE] = {64, 64, 64};
// int printlatency = 1; //print every io latency
// // int maxio = 100000000; //halt if number of IO > maxio, to prevent printing too many to metrics file
// int respecttime = 1;
// int block_size = 1; // by default, one sector (512 bytes)
// int single_io_limit = (1024*1024); // the size limit of a single IOs request, used to break down large IOs

// ANOTHER GLOBAL VARIABLES
int dev_idx_enum[NR_DEVICE] = {0, 1, 2};
int fd[NR_DEVICE];
int64_t DISKSZ[NR_DEVICE];
int64_t nr_tt_ios;
int64_t nr_ios[NR_DEVICE];
int64_t latecount = 0;
int64_t slackcount = 0;
uint64_t starttime;

int64_t *oft[NR_DEVICE];
int *reqsize[NR_DEVICE];
int *reqflag[NR_DEVICE];
float *timestamp[NR_DEVICE];

FILE *metrics; // current format: offset,size,type,latency(ms)
FILE *metrics_sub;

pthread_mutex_t lock; // only for writing to logfile, TODO
// pthread_mutex_t lock_sub; // only for writing to logfile, TODO

pthread_barrier_t sync_barrier;

int64_t jobtracker[NR_DEVICE] = {0, 0, 0};

/*=============================================================*/

static int64_t get_disksz(int devfd)
{
    int64_t sz;
    ioctl(devfd, BLKGETSIZE64, &sz);
    printf("Disk size is %"PRId64" MB\n", sz / 1024 / 1024);
    return sz;
}

static void prepare_metrics(char *logfile, const char* suffix)
{
    char filename[1024];

    if (printlatency == 1) {
        sprintf(filename, "%s_%s", logfile, suffix);
        metrics = fopen(filename, "w");
        if (!metrics) {
            printf("Error creating metrics(%s) file!\n", logfile);
            exit(1);
        }
    }
    if (printlatency == 1) {
        sprintf(filename, "%s_%s_sub", logfile, suffix);
        metrics_sub = fopen(filename, "w");
        if (!metrics_sub) {
            printf("Error creating sub metrics(%s) file!\n", logfile);
            exit(1);
        }
    }
}

int64_t read_trace(char ***req, char *tracefile)
{
    char line[1024];
    int64_t nr_lines = 0, i = 0;
    int ch;

    // first, read the number of lines
    FILE *trace = fopen(tracefile, "r");
    if (trace == NULL) {
        printf("Cannot open trace file: %s!\n", tracefile);
        exit(1);
    }

    while (!feof(trace)) {
        ch = fgetc(trace);
        if (ch == '\n') {
            nr_lines++;
        }
    }
    printf("there are [%lu] IOs in total in trace:%s\n", nr_lines, tracefile);

    rewind(trace);

    // then, start parsing
    if ((*req = malloc(nr_lines * sizeof(char *))) == NULL) {
        fprintf(stderr, "memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }

    while (fgets(line, sizeof(line), trace) != NULL) {
        line[strlen(line) - 1] = '\0';
        if (((*req)[i] = malloc((strlen(line) + 1) * sizeof(char))) == NULL) {
            fprintf(stderr, "memory allocation error (%d)!\n", __LINE__);
            exit(1);
        }

        strcpy((*req)[i], line);
        i++;
    }

    printf("%s,nr_lines=%lu,i=%lu\n", __func__, nr_lines, i);
    fclose(trace);

    return nr_lines;
}

void parse_io(char **reqs, int idx_dev)
{
    int64_t i = 0;
    int zero_count = 0, trash;

    oft[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(int64_t));
    reqsize[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(int));
    reqflag[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(int));
    timestamp[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(float));
    // reqdev[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(int));

    if (oft[idx_dev] == NULL || reqsize[idx_dev] == NULL || reqflag[idx_dev] == NULL ||
            timestamp[idx_dev] == NULL) {
        printf("memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }

    for (i = 0; i < nr_ios[idx_dev]; i++) {
        sscanf(reqs[i], "%f %d %ld %d %d", 
            &timestamp[idx_dev][i],
            &trash,
            &oft[idx_dev][i],
            &reqsize[idx_dev][i],
            &reqflag[idx_dev][i]);

        assert(oft[idx_dev][i] >= 0);
        if (zero_count == 0) zero_count++;

        //printf("read  %.2f,%ld,%d,%d\n", timestamp[idx_dev][i], oft[idx_dev][i], reqsize[idx_dev][i], reqflag[idx_dev][i]);
    }

    printf("\"zero\" I/O count: %d\n", zero_count);
}

void *pr_progress()
{
    int64_t progress, np;
    int64_t cur_late_cnt, cur_slack_cnt;

    while (1) {
        progress = atomic_read(&jobtracker[0])+atomic_read(&jobtracker[1])+atomic_read(&jobtracker[2]);
        cur_late_cnt = atomic_read(&latecount);
        cur_slack_cnt = atomic_read(&slackcount);

        np = (progress > nr_tt_ios) ? nr_tt_ios : progress;
        printf("Progress: %.2f%% (%lu/%lu), Late rate: %.2f%% (%lu), "
                "Slack rate: %.2f%% (%lu)\r",
                100 * (float)np / nr_tt_ios, progress, nr_tt_ios,
                100 * (float)cur_late_cnt / nr_tt_ios, cur_late_cnt,
                100 * (float)cur_slack_cnt / nr_tt_ios, cur_slack_cnt);
        fflush(stdout);

        if (progress > nr_tt_ios) {
            break;
        }

        sleep(2);
    }
    printf("\n\n All done!\n");

    return NULL;
}

void do_replay(void* (*io_func)(void*))
{
    pthread_t track_thread; //progress
    pthread_t *tid[NR_DEVICE]; // workers
    struct timeval t1, t2;
    float totaltime;
    int i, t, err;
    int thread_count=0;
    for(i = 0; i < NR_DEVICE ; i++) {
        thread_count += nr_workers[i];
    }

    err = pthread_barrier_init(&sync_barrier, NULL, thread_count+1);
    if (err != 0) {
        printf("Error creating barrier\n");
        exit(1);
    }

    // thread creation
    for (i=0; i<NR_DEVICE; i++) {
        tid[i] = malloc(nr_workers[i] * sizeof(pthread_t));
        if (tid[i] == NULL) {
            printf("Error malloc thread,LOC(%d)!\n", __LINE__);
            exit(1);
        }
    }

    assert(pthread_mutex_init(&lock, NULL) == 0);
    // assert(pthread_mutex_init(&lock_sub, NULL) == 0);

    for (i=0; i<NR_DEVICE; i++) {
        for (t = 0; t < nr_workers[i]; t++) {
            assert(pthread_create(&(tid[i][t]), NULL, io_func, &(dev_idx_enum[i])) == 0);
        } 
    }
    //assert(pthread_create(&track_thread, NULL, pr_progress, NULL) == 0);

    sleep(1);
    printf("Starting IO replay\n");
    gettimeofday(&t1, NULL);
    starttime = t1.tv_sec * 1000000 + t1.tv_usec;
    //hit the barrier so they all start
    pthread_barrier_wait(&sync_barrier);

    // wait for all threads to finish
    for (int i=0; i<NR_DEVICE; i++) {
        for (t = 0; t < nr_workers[i]; t++) {
            pthread_join(tid[i][t], NULL);
        }
    }
    
    gettimeofday(&t2, NULL);
    //pthread_join(track_thread, NULL); //progress

    // calculate something
    totaltime = (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_usec - t1.tv_usec) / 1e3;
    printf("==============================\n");
    printf("Total run time: %.3f ms\n", totaltime);

    if (respecttime == 1) {
        printf("Late rate: %.2f%%\n", 100 * (float)atomic_read(&latecount) / nr_tt_ios);
        printf("Slack rate: %.2f%%\n", 100 * (float)atomic_read(&slackcount) / nr_tt_ios);
    }

    assert(pthread_mutex_destroy(&lock) == 0);
    // assert(pthread_mutex_destroy(&lock_sub) == 0);

    //run statistics
    //system("python statistics.py");

    //reset
    for(i = 0 ; i < NR_DEVICE ; i++) {
        jobtracker[i] = 0;
    }

}

int parse_device(char *dev_list) {
    int count = 0;
    char *token;

    token = strtok(dev_list, "-");
    while (token) {

        fd[count] = open(token, O_DIRECT | O_RDWR);
        if (fd[count] < 0) {
            printf("Cannot open %s\n", token);
            exit(1);
        }
        printf("Opened %s\n", token);
        token = strtok(NULL, "-");
        count++;
    }
    return count;
}

int main (int argc, char **argv)
{
    char device[256];
    char tracefile[NR_DEVICE][256], logfile[256];
    char **request[NR_DEVICE];

    if (argc != 7) {
        printf("Usage: ./replayer <baseline|failover> /dev/tgt0-/dev/tgt1-/dev/tgt2 tracefile0 tracefile1 tracefile2 logfile\n");
        exit(1);
    } else {
        sprintf(device, "%s", argv[2]);
        printf("Disk ==> %s\n", device);
        for (int i=0; i<NR_DEVICE; i++) {
            sprintf(tracefile[i], "%s", argv[3+i]);
            printf("Trace #%d ==> %s\n", i, tracefile[i]);
        }
        sprintf(logfile, "%s", argv[3+NR_DEVICE]);
        printf("Logfile ==> %s\n", logfile);
    }

    // this fills the fd array with open fds
    if (parse_device(device)!=NR_DEVICE) {
        printf("Nan, number of devices must be %d\n", NR_DEVICE);
        exit(1);
    }

    for (int i=0; i<NR_DEVICE; i++) {
        DISKSZ[i] = get_disksz(fd[i]);
    }

    for (int i=0; i<NR_DEVICE; i++) {
        // read the trace before everything else
        nr_ios[i] = read_trace(&(request[i]), tracefile[i]);
        nr_tt_ios += nr_ios[i];
        // store trace related fields into our global arraries
        // this parses strings and fills the arrays oft, reqsize, reqflag, timestamp
        parse_io(request[i], i);
    }

    if(!strcmp(argv[1], "baseline")) {
        printf("About to run baseline, which means the linnos hook should NOT be loaded\n");
        // printf("Press any key to continue\n");  
        // getchar();  
        prepare_metrics(logfile, "baseline");
        do_replay(perform_io_baseline);
    } else if(!strcmp(argv[1], "failover")) {
        printf("About to run failover, which means the linnos hook SHOULD be loaded\n");
        // printf("Press any key to continue\n");  
        // getchar();
        prepare_metrics(logfile, "failover");
        do_replay(perform_io_failover);
    } else {
        printf("Invalid type to run");
        return 1;
    }
    

    fclose(metrics);
    fclose(metrics_sub);

    return 0;
}
