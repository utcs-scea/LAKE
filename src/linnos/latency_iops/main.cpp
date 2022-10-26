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
#include <pthread.h>
#include <pthread.h>
#include <unistd.h>
#include <thread>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <map>


uint8_t N_THREADS = 1;

#define LARGEST_REQUEST_BYTES (64*1024*1024)
#define MEM_ALIGN 4096
int KB = 1024; int MB = 1024*1024; int GB = 1024*1024*1024;

int iops[] = { 10, 100, 1000, 2000, 5000, 10000};
int sizes[] = {4*KB, 16*KB, 64*KB, 256*KB, 1024*KB};
std::multimap<int, int> stats;
int fds[] = {0,0,0};
uint32_t lates[] = {0,0,0};
std::vector<int> init;
std::vector<std::vector<int>> io_latency_1(3, init);
int io_latency[3][100000] = {0};
int last_index_latency[] = {-1, -1, -1};
int num_devices = 2;

uint32_t runtime_us = 100000;
    
struct Thread_arg {
    uint32_t device;
    int iops_idx;
    int size_idx;
    pthread_barrier_t *sync_barrier;
};

void* replayer_fn(void* arg) {
    Thread_arg *targ = (Thread_arg*) arg;
    uint32_t device = targ->device;
    int size_idx = targ->size_idx;
    int iops_idx = targ->iops_idx;
    uint32_t iop = iops[iops_idx];
    uint32_t size = sizes[size_idx];
    char *buf;
    bool is_late;
    int fd = fds[device];
    int ret;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> distr(128*MB, 1.9*GB); // define the range

    if (posix_memalign((void**)&buf, MEM_ALIGN, LARGEST_REQUEST_BYTES)) {
        fprintf(stderr, "memory allocation failed\n");
        exit(1);
    }

    //start together
    pthread_barrier_wait(targ->sync_barrier);
    auto start = std::chrono::steady_clock::now();
    double period_us = (1.0/(double)iop)*1000000; //period in us

    while (1) {
        auto now = std::chrono::steady_clock::now();
        uint32_t elaps = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();

        if (elaps >= runtime_us) {
            //printf("Thread for dev %d done\n", device);
            break;
        }
        auto begin = std::chrono::steady_clock::now();
        int offset = distr(gen);
        if (offset % 512 != 0) {
            offset = offset/512 * 512 + 512;
        }
        ret = pread(fd, buf, size, offset);
        if (ret < 0) {
            printf("Error on pread %d\n", errno);
            break;
        }
        auto end = std::chrono::steady_clock::now();        
        uint32_t latency =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        //TODO store latency somewhere so that later we can calculate avg and stddev
        io_latency_1[device].push_back(latency);
        // if(last_index_latency[device] < 100000) {
        //     last_index_latency[device]++;
        //     io_latency[device][last_index_latency[device]] = latency;
        // }

        if(latency > period_us) {
            //if we are late, we cant even sleep
            lates[device]++;
        } else {
            //sleep for period_us - latency
            usleep(period_us - latency);
        }
    }
    free(buf);
    return 0;
}

template<typename T>
T variance(const std::vector<T> &vec) {
    const size_t sz = vec.size();
    if (sz == 1) {
        return 0.0;
    }

    // Calculate the mean
    const T mean = std::accumulate(vec.begin(), vec.end(), 0.0) / sz;

    // Now calculate the variance
    auto variance_func = [&mean, &sz](T accumulator, const T& val) {
        return accumulator + ((val - mean)*(val - mean) / (sz - 1));
    };

    return std::accumulate(vec.begin(), vec.end(), 0.0, variance_func);
}

void print_stats(int io_idx, int size_idx) {
    int s = sizes[size_idx];
    int io = iops[io_idx];
    printf("\n Size = %d", s);
    printf("\n IOPS = %d", io);
    for(int dev = 0; dev < num_devices; dev++) {
        std::cout << "\ndevice = "<<fds[dev];
        const size_t sz = io_latency_1[dev].size();
        double mean = std::accumulate(io_latency_1[dev].begin(), io_latency_1[dev].end(), 0.0) / sz;
        std::cout << "\tmean = "<<mean;
        stats.insert(std::pair<int, int>(io, mean));
        std::cout << "\tVariance = "<<variance(io_latency_1[dev]);
        io_latency_1[dev].clear();
    }
    std::cout <<"\n Total lates =";
    for(int dev = 0; dev < num_devices; dev++) { 
        std::cout << "\ndevice = "<<fds[dev]<<"\t"<<lates[dev];
        lates[dev] = 0;
    }
}

int main (int argc, char **argv)
{   
    std::string dev_names[num_devices] = {"/dev/vdc", "/dev/vdb"};
    for (int i=0; i < num_devices ; i++) {
        //TODO open fds, check replayer.hpp for "open" to know how to
        fds[i] = open(dev_names[i].c_str(), O_DIRECT | O_RDWR);
        if (fds[i] < 0) {
            printf("Cannot open %s\n", dev_names[i].c_str());
            exit(1);
        }
        printf("Opened device %s at idx %d\n", dev_names[i].c_str(), i);
    }


    Thread_arg args[num_devices];
    // //XXX, change to 6 to run all experiments
    // //reset state
    // for (int idx=0; idx < 2 ; idx++) {
    for(int size_idx = 0 ; size_idx < 5; size_idx++) {
        for(int io_idx = 0; io_idx < 6; io_idx++) {
            pthread_barrier_t sync_barrier;
            int err = pthread_barrier_init(&sync_barrier, NULL, num_devices + 1);
            if (err != 0) {
                printf("Error creating barrier\n");
                exit(1);
            }
            pthread_t threads[num_devices];
            Thread_arg targs[num_devices];

            for (int dev=0; dev < num_devices ; dev++) {
                targs[dev].device = dev;
                targs[dev].sync_barrier = &sync_barrier;
                targs[dev].size_idx = size_idx;
                targs[dev].iops_idx = io_idx;
                pthread_create(&threads[dev], NULL, replayer_fn, (void*)&targs[dev]);
            }
            pthread_barrier_wait(&sync_barrier);
            //wait for workers
            for (int dev=0; dev < num_devices; dev++) {
                pthread_join(threads[dev], 0);
            }
            print_stats(io_idx, size_idx);
        }
    }
    std::cout <<"Printing stats"<<std::endl;
    std::multimap<int, int>::iterator itr;
    for (itr = stats.begin(); itr != stats.end(); ++itr) {
        std::cout << ',' << itr->first ;
    }
    std::cout <<std::endl;
    for (itr = stats.begin(); itr != stats.end(); ++itr) {
        std::cout << ',' << itr->second;
    }
    return 0;
}
