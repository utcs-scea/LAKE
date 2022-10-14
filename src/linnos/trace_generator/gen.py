import numpy as np
import sys

if len(sys.argv) != 2:
    print("Need argument: <file output>")
    sys.exit(1)

KB = 1024
GB = 1024*1024*1024 #i like dumb and readable
US = 1000*1000

#configs
TIME_US = 50  #in seconds
MAX_BYTE_OFFSET = 500*GB
AVG_SIZE_BYTES = 1024*KB
BYTES_STDDEV =  256*KB
ARRIVAL_RATE_US = 40.0
ARRIVAL_STDDEV = 20.0

READ_PCT = 0.7

step_size = 10
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
            #timestamp is in ms
            line = f"{total_time:.5f} 0 {offsets[i]} {int(sizes[i])} {ops[i]}\n"
            fp.write(line)
            
            total_time += timestamps_us[i]
            if total_time >= TIME_US:
                done = True
                break







