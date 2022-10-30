#include <cstdio>
#include <nvml.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <stdlib.h>

//https://gist.githubusercontent.com/sakamoto-poteko/44d6cd19552fa7721b99/raw/4098c76ec7258c2d548cff47a0b8d6c5a6286e4e/nvml.cpp

const int interval_ms = 1000;


static std::atomic<float> last_cpu;
static std::atomic<float> last_gpu;

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
    int idle_prev = (prev->t_idle) + (prev->t_iowait);
    int idle_cur = (cur->t_idle) + (cur->t_iowait);
    //int idle_prev = (prev->t_idle) + (prev->t_user) + (prev->t_nice + (prev->t_iowait));
    //int idle_cur = (cur->t_idle) + (cur->t_user) + (cur->t_nice) + (cur->t_iowait);

    int nidle_prev = (prev->t_user) + (prev->t_nice) + (prev->t_system) + (prev->t_irq) + (prev->t_softirq);
    int nidle_cur = (cur->t_user) + (cur->t_nice) + (cur->t_system) + (cur->t_irq) + (cur->t_softirq);
    //int nidle_prev = (prev->t_system) + (prev->t_irq) + (prev->t_softirq); //+ (prev->t_iowait);
    //int nidle_cur = (cur->t_system) + (cur->t_irq) + (cur->t_softirq); //+ (cur->t_iowait);
    
    int total_prev = idle_prev + nidle_prev;
    int total_cur = idle_cur + nidle_cur;

    double totald = (double) total_cur - (double) total_prev;
    double idled = (double) idle_cur - (double) idle_prev;
    double cpu_perc = (1000 * (totald - idled) / totald + 1) / 10;

    return cpu_perc;
}

void cpu_thread() {
    double util;
    struct cpustat st0_0, st0_1;
    get_stats(&st0_0, -1);
    get_stats(&st0_1, -1);
    while(1) {
        util = calculate_load(&st0_0, &st0_1);
        //printf("cpu %f\n", util);
        last_cpu.store((float)10);
        get_stats(&st0_0, -1);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        get_stats(&st0_1, -1);
    }
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
    std::printf("Device %d: %s\n", 0, device_name);

    nvmlUtilization_st device_utilization;

    while(1) {
        result = nvmlDeviceGetUtilizationRates(device, &device_utilization);
        if (result != NVML_SUCCESS)
            exit(1);
        last_gpu.store(2);
        //printf("gpu %f\n", device_utilization.gpu);
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }

    nvmlShutdown();
}

int main() {
    std::thread gpu_t(gpu_thread);
    std::thread cpu_t(cpu_thread);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
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