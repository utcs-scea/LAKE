#ifndef __REPLAYER_H__
#define __REPLAYER_H__

#include <stdint.h>
#include <atomic>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <fstream>
#include <string>
#include <fcntl.h>

#define LARGEST_REQUEST_BYTES (16*1024*1024)
#define MEM_ALIGN 4096
#define SINGLE_IO_LIMIT 1024*1024

inline uint64_t get_ns_ts() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

//aaargh, I should have used this when reading
struct TraceOp {
    double timestamp;
    uint64_t offset;
    uint64_t size;
    uint8_t op;
    uint8_t device;
};

class Trace {
private:
    uint64_t io_rejections=0;
    uint64_t unique_io_rejections=0;
    uint64_t never_completed_ios=0;

    uint32_t ndevices;
    uint32_t *nr_workers;
    std::atomic<uint64_t> *jobtracker;
    uint64_t *trace_line_count;
    uint64_t start_timestamp;
    std::atomic<uint64_t> late_ios;

    //the traces themselves
    double **req_timestamps;
    uint64_t **req_offsets;
    uint64_t **req_sizes;
    uint8_t **req_ops;
    int *dev_fds;

    //failover stuff
    std::atomic<uint64_t> fails;
    std::atomic<uint64_t> unique_fails;
    std::atomic<uint64_t> never_finished;

    /*log format:
    * 1: timestamp in ms
    * 2: latency in us
    * 3: r/w type [0 for w, 1 for r] (this is opposite of fio)
    * 4: I/O size in bytes
    * 5: offset in bytes
    * 6: IO submission time (not used)
    */
    std::ofstream outfile;  
    std::mutex io_mutex;


    void allocate_trace() {
        req_timestamps = new double*[ndevices];
        req_offsets = new uint64_t*[ndevices];
        req_sizes = new uint64_t*[ndevices];
        req_ops = new uint8_t*[ndevices];
    }

public:
    Trace(char* dev_string) {
        ndevices = 0;
        char *token;
        std::string dev_names[12];
        token = strtok(dev_string, "-");
        while (token) {
            dev_names[ndevices] = std::string(token);
            ndevices++;
            token = strtok(NULL, "-");
        }

        dev_fds = new int[ndevices];
        for (int i = 0 ; i < ndevices ; i++) {
            dev_fds[i] = open(dev_names[i].c_str(), O_DIRECT | O_RDWR | O_LARGEFILE);
            if (dev_fds[i] < 0) {
                printf("Cannot open %s\n", dev_names[i].c_str());
                exit(1);
            }
            printf("Opened device %s\n", dev_names[i].c_str());
        }

        nr_workers = new uint32_t[ndevices];
        jobtracker = new std::atomic<uint64_t>[ndevices];
        trace_line_count = new uint64_t[ndevices]{0};
        ndevices = ndevices;
        allocate_trace();
        std::atomic_init(&late_ios, (uint64_t)0);
        std::atomic_init(&fails, (uint64_t)0);
        std::atomic_init(&unique_fails, (uint64_t)0);
        std::atomic_init(&never_finished, (uint64_t)0);
    }

    ~Trace() {
        delete nr_workers;
        delete jobtracker;

        for (int i = 0 ; i < ndevices ; i++) {
            delete req_timestamps[i] ;
            delete req_offsets[i];
            delete req_sizes[i];
            delete req_ops[i];
        }

        delete trace_line_count;
        delete req_timestamps;
        delete req_offsets;
        delete req_sizes;
        delete req_ops;

        outfile.close();
    }

    uint8_t get_ndevices() {
        return ndevices;
    }

    uint8_t get_fd(uint8_t dev) {
        return dev_fds[dev];
    }

    int* get_fds() {
        return dev_fds;
    }

    void set_output_file(std::string filename) {
        outfile = std::ofstream(filename);
    }

    void parse_file(uint8_t device, char* trace_path) {
        trace_line_count[device] = 0;
        std::string cstr = std::string(trace_path);
        std::ifstream in(cstr);
        std::string line;
        while(std::getline(in, line)) {
            trace_line_count[device]++;
        }
        in.clear();
        in.seekg(0);

        printf("Trace of device %d has %lu lines\n", device, trace_line_count[device]);

        req_timestamps[device] = new double[trace_line_count[device]];
        req_offsets[device] = new uint64_t[trace_line_count[device]];
        req_sizes[device] = new uint64_t[trace_line_count[device]];
        req_ops[device] = new uint8_t[trace_line_count[device]];
        
        double timestamp;
        int trash;
        uint64_t offset, size;
        uint32_t op_type; //0 is read, 1 write
        uint64_t max_size=0;
        for (int i = 0 ; i < trace_line_count[device] ; i++) {
            std::getline(in, line);
            //printf("parsing %s\n", line.c_str());
            sscanf(line.c_str(), "%lf %d %lu %lu %u", 
                &timestamp, &trash, &offset, &size, &op_type);

            if (offset < 256*1e6) { //new gen handles this
                offset + 256*1e6; 
            }

            //in >> timestamp >> trash >> offset >> size >> op_type;
            req_timestamps[device][i] = timestamp;
            req_offsets[device][i] = offset;
            req_sizes[device][i] = size;
            req_ops[device][i] = op_type;
            //printf("%f, %lu, %lu, %d\n", timestamp, offset, size, op_type);
            if(size > max_size) {
                max_size = size;
            }
        }

        printf("Max size %lu MB\n", (uint64_t)(max_size/1e6));
    }

    /* 1: timestamp in ms
    * 2: latency in us
    * 3: r/w type [0 for w, 1 for r] (this is opposite of fio)
    * 4: I/O size in bytes
    * 5: offset in bytes
    * 6: IO submission time (not used)
    * 7: Device index
    */
   //ts comes in as us
    void write_output_line(uint64_t ts, uint32_t latency, uint8_t op,
            uint64_t size, uint64_t offset, uint64_t submission, 
            uint32_t device) {
        std::lock_guard<std::mutex> lk(io_mutex);
        char buf[1024]; 
        sprintf(buf, "%.3ld,%d,%d,%ld,%lu,%.3ld,%u", ts, latency, !op, 
                size, offset, submission, device);
        outfile << std::string(buf) << std::endl;
    }

    TraceOp get_line(uint8_t device) {
        uint64_t line_n = jobtracker[device].fetch_add(1, std::memory_order_seq_cst);

        TraceOp traceop; 
        traceop.timestamp = line_n >= trace_line_count[device] ? -1 : req_timestamps[device][line_n];
        traceop.offset = req_offsets[device][line_n],
        traceop.size = req_sizes[device][line_n],
        traceop.op = req_ops[device][line_n];
        return traceop;
    }

    void add_late_op() {
        late_ios.fetch_add(1, std::memory_order_seq_cst);
    }

    uint64_t get_late_op() {
        return late_ios.fetch_add(0, std::memory_order_seq_cst);
    }

    void print_stats() {
        uint64_t total_lines = 0;
        for (int i = 0 ; i < ndevices ; i++) {
            total_lines += trace_line_count[i];
        }
        printf("stats: %lu IOs failed\n", std::atomic_load(&fails));
        printf("That's about %f of all IOs issued\n", std::atomic_load(&fails)/(float)total_lines);
        printf("stats: %lu unique IOs failed\n", std::atomic_load(&unique_fails));
        printf("That's about %f of unique IOs\n", std::atomic_load(&unique_fails)/(float)total_lines);
        printf("stats: %lu IOs never finished\n", std::atomic_load(&never_finished));

    }

    void add_fail(){
        std::atomic_fetch_add(&fails, (uint64_t)1);
    }

    void add_unique_fail() {
        std::atomic_fetch_add(&unique_fails, (uint64_t)1);
    }

    void add_never_finished() {
        std::atomic_fetch_add(&never_finished, (uint64_t)1);
    }

};


#endif