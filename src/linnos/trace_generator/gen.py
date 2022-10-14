import numpy as np
import sys

if len(sys.argv) != 2:
    print("Need argument: <file output>")
    sys.exit(1)

KB = 1024
GB = 1024*1024*1024 #i like dumb and readable
US = 1000*1000

#configs
TIME_US = 500  #seconds times us
MAX_BYTE_OFFSET = 500*GB
AVG_SIZE_BYTES = 128*KB
BYTES_STDDEV =  16*KB
ARRIVAL_RATE_US = 40.0
ARRIVAL_STDDEV = 10.0

#AVG_SIZE_BYTES = 1024*KB
#BYTES_STDDEV =  256*KB
#ARRIVAL_RATE_US = 200.0
#ARRIVAL_STDDEV = 10.0

READ_PCT = 0.7

def get_next_multiple(A, B):
    if (A % B):
        A = A + (B - A % B)
    return A

max_size = 0
step_size = 100
total_time = 0
done = False
now = 0
with open(sys.argv[1], "w") as fp:
    while not done:
        #timestamps_us = np.random.zipf(ARRIVE_RATE_US, step_size)
        #print(timestamps_us)
        #zipfian sucks
        #later: switch between burst and chill
        timestamps_us = np.random.normal(ARRIVAL_RATE_US, ARRIVAL_STDDEV, 100)
        
        sizes = np.random.normal(AVG_SIZE_BYTES, BYTES_STDDEV, step_size)
        offsets = np.random.randint(0, MAX_BYTE_OFFSET, size=step_size)
        ops = np.random.choice([0, 1], size=step_size, p=[READ_PCT, 1-READ_PCT])

        for i in range(step_size):
            #clip stuff so we dont have negatives or close to +inf
            aligned_offset = get_next_multiple(abs(offsets[i]), 4096)
            aligned_offset = min(aligned_offset, MAX_BYTE_OFFSET)
            aligned_size = get_next_multiple(abs(sizes[i]), 4096)
            aligned_size = min(aligned_size, 10*AVG_SIZE_BYTES)

            if aligned_size > max_size:
                max_size = aligned_size

            line = f"{total_time:.5f} 0 {int(aligned_offset)} {int(aligned_size)} {ops[i]}\n"
            fp.write(line)
            
            total_time += timestamps_us[i]
            if total_time >= TIME_US:
                done = True
                break

print(f"Max size: {max_size/(1e6)} MB")
print(f"Max size: {max_size} B")



