#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "io_ops.h"
#include "globals.h"


int64_t atomic_read(int64_t* ptr) {
    return __atomic_load_n (ptr,  __ATOMIC_SEQ_CST);
}

void atomic_add(int64_t* ptr, int val) {
    __atomic_add_fetch(ptr, val, __ATOMIC_SEQ_CST);
}

int64_t atomic_fetch_inc(int64_t* ptr) {
    return __atomic_fetch_add(ptr, 1, __ATOMIC_SEQ_CST);
}

static int sleep_until(float ts) {
    struct timeval t1;
    uint32_t sleep_time;
    gettimeofday(&t1, NULL); //get current time
    //how much time since we started
    int64_t elapsedtime = t1.tv_sec * 1e6+ t1.tv_usec - starttime;
    //next op is in the future, so sleep
    if (elapsedtime < (int64_t)(ts * 1000)) {
        sleep_time = (uint32_t)(ts * 1000) - elapsedtime;
        if (sleep_time > 100000) {
            //myslackcount++;
            return 1;
        }
        usleep(sleep_time);
        return 0;
    } else { //next op is in the past
        //mylatecount++;
        return -1;
    }
}

void *perform_io_failover(void *input)
{
    int dev_index = *(int *)input;
    int64_t cur_idx;
    int mylatecount = 0;
    int myslackcount = 0;
    struct timeval t1, t2;
    
    int request_io_size_limit, request_io_size, ret;
	int64_t request_offset;
	char *req_str[2] = {"write", "read"};

    int64_t *dev_trace_offsets = oft[dev_index];
    int *dev_trace_io_sizes = reqsize[dev_index];
    int *dev_trace_req_type = reqflag[dev_index];
    float *dev_trace_timestamps = timestamp[dev_index];

    int max_split_io_count = 4;
    int request_io_count;
    int *size_sub_arr, *lat_sub_arr, *fail_sub_arr;
    int64_t *offset_sub_arr;
    float *sub_sub_arr;

    void *buff;

    size_sub_arr = (int *)malloc(sizeof(int)*max_split_io_count);
    lat_sub_arr = (int *)malloc(sizeof(int)*max_split_io_count);
    offset_sub_arr = (int64_t *)malloc(sizeof(int64_t)*max_split_io_count);
    sub_sub_arr = (float *)malloc(sizeof(float)*max_split_io_count);
    fail_sub_arr = (int *)malloc(sizeof(int)*max_split_io_count);

    if (posix_memalign(&buff, MEM_ALIGN, LARGEST_REQUEST_SIZE * block_size)) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    //hit barrier when we are ready
    pthread_barrier_wait(&sync_barrier);

    while (1) {
        //get the index of an io in the trace
        cur_idx = atomic_fetch_inc(&(jobtracker[dev_index]));
        if (cur_idx >= nr_ios[dev_index]) {
            break;
        }

        myslackcount = 0;
        mylatecount = 0;
        //this is always true, single_io_limit is 1024 
        request_io_size_limit = (single_io_limit > 0) ? single_io_limit : dev_trace_io_sizes[cur_idx];
        request_io_size = dev_trace_io_sizes[cur_idx];
        request_offset = dev_trace_offsets[cur_idx];

        //floor of how many IOs we are going to do
        request_io_count = (request_io_size+single_io_limit-1)/single_io_limit;
        if (request_io_count > max_split_io_count) {
            size_sub_arr = (int *)realloc(size_sub_arr, sizeof(int)*request_io_count);
            lat_sub_arr = (int *)realloc(lat_sub_arr, sizeof(int)*request_io_count);
            offset_sub_arr = (int64_t *)realloc(offset_sub_arr, sizeof(int64_t)*request_io_count);
            sub_sub_arr = (float *)realloc(sub_sub_arr, sizeof(float)*request_io_count);
            fail_sub_arr = (int *)realloc(fail_sub_arr, sizeof(int)*request_io_count);
            max_split_io_count = request_io_count;
        }

        // respect time part
        if (respecttime == 1) {
            int ret = sleep_until(dev_trace_timestamps[cur_idx]);

            if(ret == 1) myslackcount++;
            if(ret == -1) mylatecount++;
        }
		
        // do the job
		//printf("IO %lu: size: %d; offset: %lu\n", cur_idx, request_io_size, request_offset);
        gettimeofday(&t1, NULL); //reset the start time to before start doing the job
        /* the submission timestamp */
        float submission_ts = (t1.tv_sec * 1e6 + t1.tv_usec - starttime) / 1000;
        int current_sub_io = 0, lat, total_latency = 0;
        
		while (request_io_size > 0) {
            int nr_fail;
            for (nr_fail = 0 ; nr_fail < MAX_FAIL; nr_fail++) {
                int ret;
                uint32_t this_io_size = request_io_size_limit > request_io_size ? request_io_size : request_io_size_limit;
                if(dev_trace_req_type == WRITE) {
                    ret = pwrite( fd[(dev_index+nr_fail)%NR_DEVICE], 
                            buff, 
                            this_io_size, 
                            request_offset);
                } else {
                    ret = pread( fd[(dev_index+nr_fail)%NR_DEVICE], 
                        buff, 
                        this_io_size, 
                        request_offset);
                }

                if (ret > 0) {
                    goto success;
                }
            }

            if (ret <= 0) {
                printf("ERR: final try not successful\n");
                exit(1);
            }
success:
            gettimeofday(&t2, NULL);
            lat = (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_usec - t1.tv_usec);
            total_latency += lat;

            size_sub_arr[current_sub_io] = ret;
            lat_sub_arr[current_sub_io] = lat;
            offset_sub_arr[current_sub_io] = request_offset;
            sub_sub_arr[current_sub_io] = (t1.tv_sec * 1e6 + t1.tv_usec - starttime) / 1000;
            fail_sub_arr[current_sub_io] = nr_fail;
            current_sub_io++;

            // prepare for the next sub-IO
			request_io_size -= ret;
			request_offset += ret;
		}


        if (printlatency == 1) {
            /*
             * log format:
             * 1: timestamp in ms
             * 2: latency in us
             * 3: r/w type [0 for w, 1 for r] (this is opposite of fio)
             * 4: I/O size in bytes
             * 5: offset in bytes
             */
            pthread_mutex_lock(&lock);
            fprintf(metrics, "%.3f,%d,%d,%d,%ld,%.3f\n", dev_trace_timestamps[cur_idx], lat,
                    dev_trace_req_type[cur_idx], dev_trace_io_sizes[cur_idx], dev_trace_offsets[cur_idx],
                    submission_ts);
            
            for (int i = 0; i < request_io_count; i++) {
                fprintf(metrics_sub, "%ld,%d,%.3f,%d,%d,%d,%ld,%.3f,%d,%d\n", cur_idx, dev_index, dev_trace_timestamps[cur_idx], lat_sub_arr[i],
                        dev_trace_req_type[cur_idx], size_sub_arr[i], offset_sub_arr[i], sub_sub_arr[i], ret>0?1:ret, fail_sub_arr[i]);
            }
            pthread_mutex_unlock(&lock);
        }

        atomic_add(&latecount, mylatecount);
        atomic_add(&slackcount, myslackcount);
    }

    free(buff);
    return NULL;
}


void *perform_io_baseline(void *input)
{
    int dev_index = *(int *)input;
    int64_t cur_idx;
    int mylatecount = 0;
    int myslackcount = 0;
    struct timeval t1, t2;
    
    int request_io_size_limit, request_io_size, ret;
	int64_t request_offset;
	char *req_str[2] = {"write", "read"};

    int64_t *dev_trace_offsets = oft[dev_index];
    int *dev_trace_io_sizes = reqsize[dev_index];
    int *dev_trace_req_type = reqflag[dev_index];
    float *dev_trace_timestamps = timestamp[dev_index];

    int max_split_io_count = 4;
    int request_io_count;
    int *size_sub_arr, *lat_sub_arr, *fail_sub_arr;
    int64_t *offset_sub_arr;
    float *sub_sub_arr;

    void *buff;

    size_sub_arr = (int *)malloc(sizeof(int)*max_split_io_count);
    lat_sub_arr = (int *)malloc(sizeof(int)*max_split_io_count);
    offset_sub_arr = (int64_t *)malloc(sizeof(int64_t)*max_split_io_count);
    sub_sub_arr = (float *)malloc(sizeof(float)*max_split_io_count);
    fail_sub_arr = (int *)malloc(sizeof(int)*max_split_io_count);

    if (posix_memalign(&buff, MEM_ALIGN, LARGEST_REQUEST_SIZE * block_size)) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    //hit barrier when we are ready
    pthread_barrier_wait(&sync_barrier);

    while (1) {
        //get the index of an io in the trace
        cur_idx = atomic_fetch_inc(&(jobtracker[dev_index]));
        if (cur_idx >= nr_ios[dev_index]) {
            break;
        }

        myslackcount = 0;
        mylatecount = 0;
        //this is always true, single_io_limit is 1024 
        request_io_size_limit = (single_io_limit > 0) ? single_io_limit : dev_trace_io_sizes[cur_idx];
        request_io_size = dev_trace_io_sizes[cur_idx];
        request_offset = dev_trace_offsets[cur_idx];

        //floor of how many IOs we are going to do
        request_io_count = (request_io_size+single_io_limit-1)/single_io_limit;
        if (request_io_count > max_split_io_count) {
            size_sub_arr = (int *)realloc(size_sub_arr, sizeof(int)*request_io_count);
            lat_sub_arr = (int *)realloc(lat_sub_arr, sizeof(int)*request_io_count);
            offset_sub_arr = (int64_t *)realloc(offset_sub_arr, sizeof(int64_t)*request_io_count);
            sub_sub_arr = (float *)realloc(sub_sub_arr, sizeof(float)*request_io_count);
            fail_sub_arr = (int *)realloc(fail_sub_arr, sizeof(int)*request_io_count);
            max_split_io_count = request_io_count;
        }

        // respect time part
        if (respecttime == 1) {
            int is_slack = sleep_until(dev_trace_timestamps[cur_idx]);
            if (is_slack) myslackcount++;
            else mylatecount++;
        }
		
        // do the job
		printf("IO %lu: size: %d; offset: %lu\n", cur_idx, request_io_size, request_offset);
        gettimeofday(&t1, NULL); //reset the start time to before start doing the job
        /* the submission timestamp */
        float submission_ts = (t1.tv_sec * 1e6 + t1.tv_usec - starttime) / 1000;
        int current_sub_io = 0, lat, total_latency = 0;
        
		while (request_io_size > 0) {
            uint32_t this_io_size = request_io_size_limit > request_io_size ? request_io_size : request_io_size_limit;
            if(dev_trace_req_type == WRITE) {
                ret = pwrite(fd[(dev_index)%NR_DEVICE], 
                        buff, 
                        this_io_size, 
                        request_offset);
            } else {
                ret = pread( fd[(dev_index)%NR_DEVICE], 
                    buff, 
                    this_io_size, 
                    request_offset);
            }

success:
            gettimeofday(&t2, NULL);
            lat = (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_usec - t1.tv_usec);
            total_latency += lat;

            size_sub_arr[current_sub_io] = ret;
            lat_sub_arr[current_sub_io] = lat;
            offset_sub_arr[current_sub_io] = request_offset;
            sub_sub_arr[current_sub_io] = (t1.tv_sec * 1e6 + t1.tv_usec - starttime) / 1000;
            fail_sub_arr[current_sub_io] = 0;
            current_sub_io++;

            // prepare for the next sub-IO
			request_io_size -= ret;
			request_offset += ret;
		}


        if (printlatency == 1) {
            /*
             * log format:
             * 1: timestamp in ms
             * 2: latency in us
             * 3: r/w type [0 for w, 1 for r] (this is opposite of fio)
             * 4: I/O size in bytes
             * 5: offset in bytes
             */
            pthread_mutex_lock(&lock);
            //fprintf(metrics, "%ld,%d,%.3f,%d,%d,%d,%ld,%.3f,%d\n", cur_idx, dev_index, dev_trace_timestamps[cur_idx], total_latency,
            //        dev_trace_req_type[cur_idx], dev_trace_io_sizes[cur_idx], dev_trace_offsets[cur_idx], submission_ts, ret>0?1:ret);
            
            if(dev_trace_req_type[cur_idx] != 0 && dev_trace_req_type[cur_idx] != 1) {
                printf("wtf, type is wrong %d\n", dev_trace_req_type[cur_idx]);
            }

            fprintf(metrics, "%.3f,%d,%d,%d,%ld,%.3f\n", dev_trace_timestamps[cur_idx], lat,
                    dev_trace_req_type[cur_idx], dev_trace_io_sizes[cur_idx], dev_trace_offsets[cur_idx],
                    submission_ts);

            for (int i = 0; i < request_io_count; i++) {
                fprintf(metrics_sub, "%ld,%d,%.3f,%d,%d,%d,%ld,%.3f,%d,%d\n", cur_idx, dev_index, dev_trace_timestamps[cur_idx], lat_sub_arr[i],
                        dev_trace_req_type[cur_idx], size_sub_arr[i], offset_sub_arr[i], sub_sub_arr[i], ret>0?1:ret, fail_sub_arr[i]);
            }
            pthread_mutex_unlock(&lock);
        }

        atomic_add(&latecount, mylatecount);
        atomic_add(&slackcount, myslackcount);
    }

    free(buff);
    return NULL;
}