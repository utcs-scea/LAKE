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

//#include "atomic.h"

#define NR_DEVICE 3
#define MAX_FAIL 2

int64_t atomic_read(int64_t* ptr) {
    return __atomic_load_n (ptr,  __ATOMIC_SEQ_CST);
}

void atomic_add(int64_t* ptr, int val) {
    __atomic_add_fetch(ptr, val, __ATOMIC_SEQ_CST);
}

int64_t atomic_fetch_inc(int64_t* ptr) {
    return __atomic_fetch_add(ptr, 1, __ATOMIC_SEQ_CST);
}

enum {
    READ = 1,
    WRITE = 0,
};


int LARGEST_REQUEST_SIZE = (16*1024); //blocks
int MEM_ALIGN = 4096; //bytes
int nr_workers[NR_DEVICE] = {1, 1, 1};
int printlatency = 1; //print every io latency
int respecttime = 1;
int block_size = 1; // by default, one sector (512 bytes)
int single_io_limit = (1024); // the size limit of a single IOs request, used to break down large IOs


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
void *buff;

int64_t *oft[NR_DEVICE];
int *reqsize[NR_DEVICE];
int *reqflag[NR_DEVICE];
float *timestamp[NR_DEVICE];

FILE *metrics; // current format: offset,size,type,latency(ms)
FILE *metrics_sub;

pthread_mutex_t lock; // only for writing to logfile, TODO
// pthread_mutex_t lock_sub; // only for writing to logfile, TODO

int64_t jobtracker[NR_DEVICE] = {0, 0, 0};

/*=============================================================*/

static int64_t get_disksz(int devfd)
{
    int64_t sz;

    ioctl(devfd, BLKGETSIZE64, &sz);
    printf("Disk size is %"PRId64" MB\n", sz / 1024 / 1024);

    return sz;
}

void prepare_metrics(char *logfile)
{
    // if (printlatency == 1 && nr_tt_ios > maxio) {
    //     printf("Coperd,too many IOs in the trace file (%ld)!\n", nr_tt_ios);
    //     exit(1);
    // }

    if (printlatency == 1) {
        metrics = fopen(logfile, "w");
        if (!metrics) {
            printf("Coperd,Error creating metrics(%s) file!\n", logfile);
            exit(1);
        }
    }
    if (printlatency == 1) {
        sprintf(logfile, "%s_sub", logfile);
        metrics_sub = fopen(logfile, "w");
        if (!metrics_sub) {
            printf("Coperd,Error creating metrics(%s) file!\n", logfile);
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
    printf("Coperd,there are [%lu] IOs in total in trace:%s\n", nr_lines, tracefile);

    rewind(trace);

    // then, start parsing
    if ((*req = malloc(nr_lines * sizeof(char *))) == NULL) {
        fprintf(stderr, "Coperd,memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }

    while (fgets(line, sizeof(line), trace) != NULL) {
        line[strlen(line) - 1] = '\0';
        if (((*req)[i] = malloc((strlen(line) + 1) * sizeof(char))) == NULL) {
            fprintf(stderr, "Coperd,memory allocation error (%d)!\n", __LINE__);
            exit(1);
        }

        strcpy((*req)[i], line);
        i++;
    }

    printf("Coperd,%s,nr_lines=%lu,i=%lu\n", __func__, nr_lines, i);
    fclose(trace);

    return nr_lines;
}

void parse_io(char **reqs, int idx_dev)
{
    char *one_io;
    int64_t i = 0;
    int zero_count = 0;

    oft[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(int64_t));
    reqsize[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(int));
    reqflag[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(int));
    timestamp[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(float));
    // reqdev[idx_dev] = malloc(nr_ios[idx_dev] * sizeof(int));

    if (oft[idx_dev] == NULL || reqsize[idx_dev] == NULL || reqflag[idx_dev] == NULL ||
            timestamp[idx_dev] == NULL) {
        printf("Coperd,memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }

    one_io = malloc(1024);
    if (one_io == NULL) {
        fprintf(stderr, "Coperd,memory allocation error (%d)!\n", __LINE__);
        exit(1);
    }

    for (i = 0; i < nr_ios[idx_dev]; i++) {
        memset(one_io, 0, 1024);

        strcpy(one_io, reqs[i]);

        // 1. request arrival time in "ms"
        timestamp[idx_dev][i] = atof(strtok(one_io, " "));
        // 2. device number (not needed)
        strtok(NULL, " ");
        // reqdev[i] = atoi(strtok(NULL, " "));
        // 3. block number (offset)
        oft[idx_dev][i] = atoll(strtok(NULL, " ")) % DISKSZ[idx_dev];
        // 4. request size in blocks
        reqsize[idx_dev][i] = atoi(strtok(NULL, " "));
        // make sure the request does not overflow
        if (oft[idx_dev][i]+reqsize[idx_dev][i] >= DISKSZ[idx_dev]) {
            oft[idx_dev][i] -= reqsize[idx_dev][i];
        }
        assert(oft[idx_dev][i] >= 0);
        if (zero_count == 0) zero_count++;
        // 5. request flags: 0 for write and 1 for read
        reqflag[idx_dev][i] = atoi(strtok(NULL, " "));

        // printf("%.2f,%ld,%d,%d\n", timestamp[i], oft[i], reqsize[i],reqflag[i]);
    }

    free(one_io);
    printf("\"zero\" I/O count: %d\n", zero_count);
}

void *perform_io(void *input)
{
    int dev_index = *(int *)input;
    int64_t cur_idx;
    int mylatecount = 0;
    int myslackcount = 0;
    struct timeval t1, t2;
    useconds_t sleep_time;
    int io_limit__, size__, ret;
	int64_t offset__;
	char *req_str[2] = {"write", "read"};

    // printf("I am dealing with #%d\n", dev_index);
    // return NULL;

    int64_t *oft__;
    int *reqsize__;
    int *reqflag__;
    float *timestamp__;

    oft__ = oft[dev_index];
    reqsize__ = reqsize[dev_index];
    reqflag__ = reqflag[dev_index];
    timestamp__ = timestamp[dev_index];

    int max_len = 1, cur_len;
    int *size_sub_arr, *lat_sub_arr, *fail_sub_arr;
    int64_t *offset_sub_arr;
    float *sub_sub_arr;

    size_sub_arr = (int *)malloc(sizeof(int)*max_len);
    lat_sub_arr = (int *)malloc(sizeof(int)*max_len);
    offset_sub_arr = (int64_t *)malloc(sizeof(int64_t)*max_len);
    sub_sub_arr = (float *)malloc(sizeof(float)*max_len);
    fail_sub_arr = (int *)malloc(sizeof(int)*max_len);

    while (1) {
        cur_idx = atomic_fetch_inc(&(jobtracker[dev_index]));
        if (cur_idx >= nr_ios[dev_index]) {
            break;
        }

        myslackcount = 0;
        mylatecount = 0;
        io_limit__ = (single_io_limit > 0) ? single_io_limit : reqsize__[cur_idx];
        size__ = reqsize__[cur_idx];
        offset__ = oft__[cur_idx];

        cur_len = (size__+single_io_limit-1)/single_io_limit;
        if (cur_len > max_len) {
            size_sub_arr = (int *)realloc(size_sub_arr, sizeof(int)*cur_len);
            lat_sub_arr = (int *)realloc(lat_sub_arr, sizeof(int)*cur_len);
            offset_sub_arr = (int64_t *)realloc(offset_sub_arr, sizeof(int64_t)*cur_len);
            sub_sub_arr = (float *)realloc(sub_sub_arr, sizeof(float)*cur_len);
            fail_sub_arr = (int *)realloc(fail_sub_arr, sizeof(int)*cur_len);
            max_len = cur_len;
        }

        // respect time part
        if (respecttime == 1) {
            gettimeofday(&t1, NULL); //get current time
            int64_t elapsedtime = t1.tv_sec * 1e6+ t1.tv_usec - starttime;
            if (elapsedtime < (int64_t)(timestamp__[cur_idx] * 1000)) {
                sleep_time = (useconds_t)(timestamp__[cur_idx] * 1000) - elapsedtime;
                if (sleep_time > 100000) {
                    myslackcount++;
                }
                usleep(sleep_time);
            } else { // I am late
                mylatecount++;
            }
        }
		
        // do the job
		//printf("IO %lu: size: %d; offset: %lu\n", cur_idx, size__, offset__);
        gettimeofday(&t1, NULL); //reset the start time to before start doing the job
        /* the submission timestamp */
        float submission_ts = (t1.tv_sec * 1e6 + t1.tv_usec - starttime) / 1000;
        int lat, tot_lat, i, nr_fail;
        tot_lat = 0;
        i = 0;
		while (size__ > 0) {
			
            nr_fail = 0;

            gettimeofday(&t1, NULL);

            // ret = reqflag__[cur_idx]==WRITE?pwrite(fd[dev_index], buff, io_limit__>size__?size__:io_limit__, offset__):
            //                          pread(fd[dev_index], buff, io_limit__>size__?size__:io_limit__, offset__);

            // The special forwarding policy
            // if (ret <= 0) {
            //     int fwd_dev;
            //     if (dev_index == 0) {
            //         fwd_dev = 2;    // nvme0 -> sde
            //     } else if (dev_index == 1) {
            //         fwd_dev = 2;    // sdd -> sde
            //     } else {
            //         fwd_dev = 1;    // sde -> sdd
            //     }

            //     while (ret <= 0) {
            //         ret = reqflag__[cur_idx]==WRITE?pwrite(fd[fwd_dev], buff, io_limit__>size__?size__:io_limit__, offset__):
            //                                 pread(fd[fwd_dev], buff, io_limit__>size__?size__:io_limit__, offset__);
            //     }
            //     nr_fail = 1;
            // }

   //          // try on original drive
			// ret = reqflag__[cur_idx]==WRITE?pwrite(fd[dev_index], buff, io_limit__>size__?size__:io_limit__, offset__):
			// 							pread(fd[dev_index], buff, io_limit__>size__?size__:io_limit__, offset__);
   //          if (ret > 0) {
   //              goto success;
   //          }

   //          // first time failover
   //          ret = reqflag__[cur_idx]==WRITE?pwrite(fd[(dev_index+1)%nr_dev], buff, io_limit__>size__?size__:io_limit__, offset__):
   //                                      pread(fd[(dev_index+1)%nr_dev], buff, io_limit__>size__?size__:io_limit__, offset__);
   //          if (ret > 0) {
   //              nr_fail = 1;
   //              goto success;
   //          }
            // nr_fail = 2;

            for (; nr_fail < MAX_FAIL; nr_fail++) {
                ret = reqflag__[cur_idx]==WRITE?pwrite(fd[(dev_index+nr_fail)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, offset__):
                                         pread(fd[(dev_index+nr_fail)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, offset__);
                if (ret > 0) {
                    goto success;
                }
            }

            // //second time failover (must pass)
            // while (ret <= 0) {
            //     ret = reqflag__[cur_idx]==WRITE?pwrite(fd[(dev_index+MAX_FAIL)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, offset__):
            //                             pread(fd[(dev_index+MAX_FAIL)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, offset__);
            // }

            // for (int i=0; i<10000; i++) {
            //     ret = reqflag__[cur_idx]==WRITE?pwrite(fd[(dev_index+MAX_FAIL)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, offset__):
            //                             pread(fd[(dev_index+MAX_FAIL)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, offset__); 
            //     if (ret > 0) {
            //         goto success;
            //     }
            // }
            // ret = io_limit__>size__?size__:io_limit__;
            // // printf("ERR: reached retry limit at request [%ld, %d]\n", cur_idx, dev_index);

            for (int i=0; i<1; i++) {
                ret = reqflag__[cur_idx]==WRITE?pwrite(fd[(dev_index+MAX_FAIL)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, offset__):
                                        pread(fd[(dev_index+MAX_FAIL)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, offset__); 
                if (ret > 0) {
                    goto success;
                }
            }
            ret = reqflag__[cur_idx]==WRITE?pwrite(fd[(dev_index+MAX_FAIL)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, 0):
                                        pread(fd[(dev_index+MAX_FAIL)%NR_DEVICE], buff, io_limit__>size__?size__:io_limit__, 0); 
            if (ret <= 0) {
                printf("ERR: final try not successful\n");
                exit(1);
            }
success:
            gettimeofday(&t2, NULL);
            lat = (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_usec - t1.tv_usec);
            tot_lat += lat;

            size_sub_arr[i] = ret;
            lat_sub_arr[i] = lat;
            offset_sub_arr[i] = offset__;
            sub_sub_arr[i] = (t1.tv_sec * 1e6 + t1.tv_usec - starttime) / 1000;
            fail_sub_arr[i] = nr_fail;
            i++;

            // prepare for the next sub-IO
			size__ -= ret;
			offset__ += ret;
		}

        /* Coperd: I/O latency in us */
        // int lat = (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_usec - t1.tv_usec);
        if (printlatency == 1) {
            /*
             * Coperd: keep consistent with fio latency log format:
             * 1: timestamp in ms
             * 2: latency in us
             * 3: r/w type [0 for w, 1 for r] (this is opposite of fio)
             * 4: I/O size in bytes
             * 5: offset in bytes
             */
            pthread_mutex_lock(&lock);
            fprintf(metrics, "%ld,%d,%.3f,%d,%d,%d,%ld,%.3f,%d\n", cur_idx, dev_index, timestamp__[cur_idx], tot_lat,
                    reqflag__[cur_idx], reqsize__[cur_idx], oft__[cur_idx], submission_ts, ret>0?1:ret);
            // fflush(metrics);
            for (i=0; i<cur_len; i++) {
                fprintf(metrics_sub, "%ld,%d,%.3f,%d,%d,%d,%ld,%.3f,%d,%d\n", cur_idx, dev_index, timestamp__[cur_idx], lat_sub_arr[i],
                        reqflag__[cur_idx], size_sub_arr[i], offset_sub_arr[i], sub_sub_arr[i], ret>0?1:ret, fail_sub_arr[i]);
            }
            // fflush(metrics_sub);
            pthread_mutex_unlock(&lock);
        }

        atomic_add(&latecount, mylatecount);
        atomic_add(&slackcount, myslackcount);
    }

    return NULL;
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

void do_replay(void)
{
    pthread_t track_thread; //progress
    pthread_t *tid[NR_DEVICE]; // workers
    struct timeval t1, t2;
    float totaltime;
    int t;

    printf("Coperd,Start doing IO replay...\n");

    // thread creation
    for (int i=0; i<NR_DEVICE; i++) {
        tid[i] = malloc(nr_workers[i] * sizeof(pthread_t));
        if (tid[i] == NULL) {
            printf("Coperd,Error malloc thread,LOC(%d)!\n", __LINE__);
            exit(1);
        }
    }

    assert(pthread_mutex_init(&lock, NULL) == 0);
    // assert(pthread_mutex_init(&lock_sub, NULL) == 0);

    gettimeofday(&t1, NULL);
    starttime = t1.tv_sec * 1000000 + t1.tv_usec;
    for (int i=0; i<NR_DEVICE; i++) {
        for (t = 0; t < nr_workers[i]; t++) {
            assert(pthread_create(&(tid[i][t]), NULL, perform_io, &(dev_idx_enum[i])) == 0);
        } 
    }
    assert(pthread_create(&track_thread, NULL, pr_progress, NULL) == 0);

    // wait for all threads to finish
    for (int i=0; i<NR_DEVICE; i++) {
        for (t = 0; t < nr_workers[i]; t++) {
            pthread_join(tid[i][t], NULL);
        }
    }
    
    pthread_join(track_thread, NULL); //progress

    gettimeofday(&t2, NULL);

    // calculate something
    totaltime = (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_usec - t1.tv_usec) / 1e3;
    printf("==============================\n");
    printf("Total run time: %.3f ms\n", totaltime);

    if (respecttime == 1) {
        printf("Late rate: %.2f%%\n", 100 * (float)atomic_read(&latecount) / nr_tt_ios);
        printf("Slack rate: %.2f%%\n", 100 * (float)atomic_read(&slackcount) / nr_tt_ios);
    }

    fclose(metrics);
    assert(pthread_mutex_destroy(&lock) == 0);
    // assert(pthread_mutex_destroy(&lock_sub) == 0);

    //run statistics
    //system("python statistics.py");
}

int parse_device(char *dev_list) {

    int count = 0;
    char *token;

    token = strtok(dev_list, "-");
    while (token) {

        fd[count] = open(token, O_DIRECT | O_RDWR);
        if (fd[count] < 0) {
            printf("Coperd,Cannot open %s\n", token);
            exit(1);
        }

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

    if (argc != 6) {
        printf("Usage: ./replayer /dev/tgt0 tracefile0 tracefile1 tracefile2 logfile\n");
        exit(1);
    } else {
        sprintf(device, "%s", argv[1]);
        printf("Disk ==> %s\n", device);
        for (int i=0; i<NR_DEVICE; i++) {
            sprintf(tracefile[i], "%s", argv[2+i]);
            printf("Trace #%d ==> %s\n", i, tracefile[i]);
        }
        sprintf(logfile, "%s", argv[2+NR_DEVICE]);
        printf("Logfile ==> %s\n", logfile);
    }

    // start the disk part
    if (parse_device(device)!=NR_DEVICE) {
        printf("Nan, number of devices must be %d\n", NR_DEVICE);
        exit(1);
    }

    for (int i=0; i<NR_DEVICE; i++) {
        DISKSZ[i] = get_disksz(fd[i]);
    }

    if (posix_memalign(&buff, MEM_ALIGN, LARGEST_REQUEST_SIZE * block_size)) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    for (int i=0; i<NR_DEVICE; i++) {
        // read the trace before everything else
        nr_ios[i] = read_trace(&(request[i]), tracefile[i]);
        nr_tt_ios += nr_ios[i];
        // store trace related fields into our global arraries
        parse_io(request[i], i);
    }

    prepare_metrics(logfile);

    // do the replay here
    do_replay();

    free(buff);

    return 0;
}
