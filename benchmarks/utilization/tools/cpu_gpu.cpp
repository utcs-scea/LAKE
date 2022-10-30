#include <cstdio>
#include <nvml.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <stdlib.h>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <cctype>
#include <algorithm>
#include <string>

//https://gist.githubusercontent.com/sakamoto-poteko/44d6cd19552fa7721b99/raw/4098c76ec7258c2d548cff47a0b8d6c5a6286e4e/nvml.cpp

const int interval_ms = 1000;

static std::atomic<float> last_cpu(0);
static std::atomic<float> last_gpu(0);
static std::atomic<float> last_api(0);
static std::atomic<uint64_t> total_cpu1(0);
static std::atomic<uint64_t> total_cpu2(0);

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

double calculate_load(struct cpustat *prev, struct cpustat *cur)
{
    //int idle_prev = (prev->t_idle) + (prev->t_iowait);
    //int idle_cur = (cur->t_idle) + (cur->t_iowait);
    int idle_prev = (prev->t_idle) + (prev->t_user) + (prev->t_nice) + (prev->t_iowait);
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

    //printf("sum %d\n", cur->t_user + cur->t_nice + cur->t_system + cur->t_idle);
    uint64_t tc = cur->t_user + cur->t_nice + cur->t_system + cur->t_idle;
    uint64_t tmp = total_cpu1.load();
    total_cpu1.store(tc);
    total_cpu2.store(tmp);

    //printf("diff %d\n", tc - tmp);

    return cpu_perc;
}

void cpu_thread() {
    double util;
    struct cpustat st0_0, st0_1;
    get_stats(&st0_0, -1);
    get_stats(&st0_1, -1);
    while(1) {
        util = calculate_load(&st0_0, &st0_1);
        last_cpu.store(util);
        get_stats(&st0_0, -1);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        get_stats(&st0_1, -1);
    }
}

void pid_thread() {
    std::array<char, 128> buffer;
    std::string result;
    std::string cmd = "pidof lake_uspace";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result = buffer.data();
        std::cout << result;
    }

    result.erase(std::remove_if(result.begin(), result.end(), 
                  [&](char ch) 
                  { return std::iscntrl(static_cast<unsigned char>(ch));}), 
                  result.end());

    //not found..
    if(result == "") {
        return;
    }

    char pidstat[1024];
    sprintf(pidstat, "/proc/%s/stat", result.c_str());

    float t1, t2;

    while(1) {
        FILE *fp = fopen(pidstat, "r");
        char cpun[255];
        int pid;
        uint64_t ll, user, system;
        char c;
        fscanf(fp,
            "%d %s %c %d" //pid,command,state,ppid
            "%d %d %d %d %u %lu %lu %lu %lu"
            "%lu %lu", 
            &pid, cpun, &c, &pid,
            &pid, &pid, &pid, &pid, &pid, &ll, &ll, &ll, &ll,
            &user, &system);
        fclose(fp);

        //printf("+: %f\n", (user+system));
        t1 = user+system;
        uint64_t c1 = total_cpu1.load();
        uint64_t c2 = total_cpu2.load();
        double pct = ((t1+t2)*100) / (c1 - c2);
        t2 = t1;
        last_api.store(pct);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

	return;
}

void gpu_thread() {
    nvmlReturn_t result;
    unsigned int device_count;

    result = nvmlInit();
    if (result != NVML_SUCCESS)
        exit(1);
    
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS)
        exit(1);

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS)
        exit(1);

    char device_name[NVML_DEVICE_NAME_BUFFER_SIZE];
    result = nvmlDeviceGetName(device, device_name, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result != NVML_SUCCESS)
        exit(1);
    //std::printf("Device %d: %s\n", 0, device_name);

    nvmlUtilization_st device_utilization;

    while(1) {
        result = nvmlDeviceGetUtilizationRates(device, &device_utilization);
        if (result != NVML_SUCCESS)
            exit(1);
        last_gpu.store(result);
        //printf("gpu %f\n", device_utilization.gpu);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    nvmlShutdown();
}

int main() {
    printf("Hello, World heh");
    // std::thread gpu_t(gpu_thread);
    // std::thread cpu_t(cpu_thread);
    std::thread pid_t(pid_thread);
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    float ts = 0;

    float c, g;        
    while(1) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        c = last_cpu.load();
        g = last_gpu.load();
        printf("%.2f,%.2f,%.2f\n", ts, c, g);
        ts += interval_ms;
    }
    return 0;
}