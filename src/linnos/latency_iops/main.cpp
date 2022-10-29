/*
 * Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
 * Copyright (C) 2022-2024 Henrique Fingler
 * Copyright (C) 2022-2024 Isha Tarte
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


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

int iops[] = {100, 1000, 5000, 10000, 20000, 50000};
int sizes[] = {4*KB, 16*KB, 64*KB, 256*KB, 1024*KB};
int fds[] = {0,0,0};
uint32_t lates[] = {0,0,0};
uint32_t total[] = {0,0,0};
std::vector<int> init;
std::vector<std::vector<int>> io_latency_1(3, init);
const int num_devices = 3;

// arrays for stats
double lat_stat_avg[5][num_devices][6];
double lat_stat_std[5][num_devices][6];

int total_io[5][num_devices][6];
int late_io[5][num_devices][6];
double percent_late[5][num_devices][6];

uint32_t runtime_us = 10 * 100000;
    
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
        total[device]++;
        auto end = std::chrono::steady_clock::now();        
        uint32_t latency =  std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

        //TODO store latency somewhere so that later we can calculate avg and stddev
        io_latency_1[device].push_back(latency);

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
        lat_stat_avg[size_idx][dev][io_idx] = mean;
        lat_stat_std[size_idx][dev][io_idx] = sqrt(variance(io_latency_1[dev]));
        total_io[size_idx][dev][io_idx] = total[dev];
        late_io[size_idx][dev][io_idx] = lates[dev];
        percent_late[size_idx][dev][io_idx] = (double)lates[dev]/ (double) total[dev];
        std::cout << "\tVariance = "<<variance(io_latency_1[dev]);
        io_latency_1[dev].clear();
        lates[dev] = 0;
        total[dev] = 0;
    }
    
}

int main (int argc, char **argv)
{   
    std::string dev_names[num_devices] = {"/dev/nvme0n1", "/dev/nvme1n1", "/dev/nvme2n1"};
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
            int runs = 1;
            if(size_idx == 0)
                runs = 2;
            for(int r = 0; r < runs; r++) {
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
                usleep(100);
                print_stats(io_idx, size_idx);
            }
        }
    }
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < num_devices; j++) {
            std::cout <<"\n_"<<sizes[i]<<"_dev_"<<j<<" = [ ";
            for(int k = 0; k < 6; k++) {
                std::cout<< lat_stat_avg[i][j][k];
                if(k != 5)
                    std::cout<< ", ";
            }
            std::cout <<"]";
        }
    }

    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < num_devices; j++) {
            std::cout <<"\n_"<<sizes[i]<<"_dev_"<<j<<"e = [ ";
            for(int k = 0; k < 6; k++) {
                std::cout<< lat_stat_std[i][j][k];
                if(k != 5)
                    std::cout<< ", ";
            }
            std::cout <<"]";
        }
    }

    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < num_devices; j++) {
            std::cout <<"\n_"<<sizes[i]<<"_dev_"<<j<<"percent_late_io = [ ";
            for(int k = 0; k < 6; k++) {
                std::cout<< percent_late[i][j][k];
                if(k != 5)
                    std::cout<< ", ";
            }
            std::cout <<"]";
        }
    }
    return 0;
}
