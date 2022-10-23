import numpy as np
import sys

if len(sys.argv) != 7:
    print("Need argument: <file output> <readpct e.g. 0.7> <total time in sec> <avg_read_size in KB> <avg_write_size in KB> <arrival rate in us>")
    sys.exit(1)

KB = 1024
MB = 1024*1024
GB = 1024*1024*1024 #i like dumb and readable
S_TO_US = 1000*1000

#configs
TIME_US = int(sys.argv[3]) *S_TO_US  #seconds times us
MAX_BYTE_OFFSET = 500*GB

READ_PCT = float(sys.argv[2])

#AVG_SIZE_BYTES = 23*KB
#BYTES_STDDEV =  2*KB
#ARRIVAL_RATE_US = 200.0
#ARRIVAL_STDDEV = 10.0

#AVG_SIZE_BYTES = 1024*KB
#BYTES_STDDEV =  256*KB
#ARRIVAL_RATE_US = 100.0
#ARRIVAL_STDDEV = 10.0

AVG_SIZE_BYTES = int(sys.argv[4])*KB
BYTES_STDDEV =  int(int(sys.argv[4])/20) * KB #5%
ARRIVAL_RATE_US = float(sys.argv[6])
ARRIVAL_STDDEV = float(float(sys.argv[6])/20) 

AVG_READ_SIZE_BYTES = int(sys.argv[4])*KB
AVG_WRITE_SIZE_BYTES = int(sys.argv[5])*KB

def get_next_multiple(A, B):
    if (A % B):
        A = A + (B - A % B)
    return A

max_size = 0
step_size = 300
total_time = 0
done = False
now = 0
next_burst = 100 # in micro seconds
burst_request_num = 16
with open(sys.argv[1], "w") as fp:
    while not done:
        #timestamps_us = np.random.zipf(ARRIVE_RATE_US, step_size)
        #print(timestamps_us)
        #zipfian sucks
        #later: switch between burst and chill
        timestamps_us = np.random.normal(ARRIVAL_RATE_US, ARRIVAL_STDDEV, step_size)
        read_sizes = np.random.normal(AVG_READ_SIZE_BYTES, BYTES_STDDEV, step_size)
        write_sizes = np.random.normal(AVG_WRITE_SIZE_BYTES, BYTES_STDDEV, step_size)
        offsets = np.random.randint(0, MAX_BYTE_OFFSET, size=step_size)
        ops = np.random.choice([0, 1], size=step_size, p=[1 - READ_PCT, READ_PCT])

        for i in range(step_size):
            #clip stuff so we dont have negatives or close to +inf
            aligned_offset = get_next_multiple(abs(offsets[i]), 4096)
            aligned_offset = min(aligned_offset, MAX_BYTE_OFFSET)
            aligned_offset = max(aligned_offset, 128*MB) #dont write to lower offsets

            if ops[i] == 0:
                aligned_size = get_next_multiple(abs(write_sizes[i]), 4096)
                aligned_size = min(aligned_size, 10*AVG_WRITE_SIZE_BYTES)
            else:
                aligned_size = get_next_multiple(abs(read_sizes[i]), 4096)
                aligned_size = min(aligned_size, 10*AVG_READ_SIZE_BYTES)


            if aligned_size > max_size:
                max_size = aligned_size

            line = f"{total_time:.5f} 0 {int(aligned_offset)} {int(aligned_size)} {ops[i]}\n"
            fp.write(line)
            
            total_time += timestamps_us[i]

            if total_time >= next_burst:
                read_sizes_burst = np.random.normal(AVG_READ_SIZE_BYTES, BYTES_STDDEV, 16)
                offsets_burst = np.random.randint(0, MAX_BYTE_OFFSET, size=16)
                next_burst += 100
                for j in range(16):
                    aligned_size = get_next_multiple(abs(read_sizes_burst[j]), 4096)
                    aligned_size = min(aligned_size, 10*AVG_READ_SIZE_BYTES)
                    total_time += 0.1

                    aligned_offset = get_next_multiple(abs(offsets_burst[j]), 4096)
                    aligned_offset = min(aligned_offset, MAX_BYTE_OFFSET)
                    aligned_offset = max(aligned_offset, 128*MB) #dont write to lower offsets

                    line = f"{total_time:.5f} 0 {int(aligned_offset)} {int(aligned_size)} {ops[i]}\n"
                    fp.write(line)


            if total_time >= TIME_US:
                done = True
                break

print(f"Max size: {max_size/(1e6)} MB")
print(f"Max size: {max_size} B")



