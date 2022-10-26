import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics

if len(sys.argv) != 7:
    print("Need argument: <file output> <readpct e.g. 0.7> <total time in sec> avg_read/min/max/stdev avg_write/min/max/stdev <arrival rate in us>")
    sys.exit(1)

KB = 1024
MB = 1024*1024
GB = 1024*1024*1024 #i like dumb and readable
S_TO_US = 1000*1000

#configs
MAX_BYTE_OFFSET = 500*GB
READ_PCT = float(sys.argv[2])
TIME_US = int(sys.argv[3]) *S_TO_US  #seconds times us

avg_rd, min_rd, max_rd, st_rd = [int(x)*KB for x in sys.argv[4].split("/")]
avg_wt, min_wt, max_wt, st_wt = [int(x)*KB for x in sys.argv[5].split("/")]

print(f"rd avg {avg_rd}")
print(f"wt avg {avg_wt}")
stdev_rd =  (math.log(max_rd) - math.log(avg_rd))/3
stdev_wt =  (math.log(max_wt) - math.log(avg_wt))/3

lower, upper = min_rd, max_rd
mu, sigma = avg_rd, st_rd
Xrd = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

lower, upper = min_wt, max_wt
mu, sigma = avg_wt, st_wt
Xwt = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)

# AVG_READ_SIZE_BYTES = int(avg_rd)
# AVG_WRITE_SIZE_BYTES = int(avg_wt)
# READ_BYTES_STDDEV =  int(int(AVG_READ_SIZE_BYTES)/20)  #5%
# WRITE_BYTES_STDDEV =  int(int(AVG_WRITE_SIZE_BYTES)/20) #5%

ARRIVAL_RATE_US = float(sys.argv[6])

def get_next_multiple(A, B):
    if (A % B):
        A = A + (B - A % B)
    return A

max_size = 0
step_size = 300
total_time = 0
done = False
with open(sys.argv[1], "w") as fp:
    while not done:
        #timestamps_us = np.random.normal(ARRIVAL_RATE_US, ARRIVAL_STDDEV, step_size)
        timestamps_us = np.random.exponential(ARRIVAL_RATE_US, step_size)
        
        #read_sizes = np.full(step_size, AVG_READ_SIZE_BYTES)
        #write_sizes = np.full(step_size, AVG_WRITE_SIZE_BYTES)
        # read_sizes = Xrd.rvs(step_size)
        # write_sizes = Xwt.rvs(step_size)
        read_sizes = np.random.lognormal(math.log(avg_rd), stdev_rd, step_size)
        write_sizes = np.random.lognormal(math.log(avg_wt), stdev_wt, step_size)
        read_sizes[read_sizes > max_rd] = max_rd
        write_sizes[write_sizes > max_wt] = max_wt

        offsets = np.random.randint(0, MAX_BYTE_OFFSET, size=step_size)
        ops = np.random.choice([0, 1], size=step_size, p=[READ_PCT, 1-READ_PCT])

        for i in range(step_size):
            aligned_offset = get_next_multiple(abs(offsets[i]), 4096)
            aligned_offset = min(aligned_offset, MAX_BYTE_OFFSET)
            aligned_offset = max(aligned_offset, 128*MB) #dont write to lower offsets

            if ops[i] == 0:
                aligned_size = get_next_multiple(int(read_sizes[i]), 512)
            else:
                aligned_size = get_next_multiple(int(write_sizes[i]), 512)

            line = f"{total_time:.5f} 0 {int(aligned_offset)} {int(aligned_size)} {ops[i]}\n"
            fp.write(line)
            
            total_time += timestamps_us[i]
            if total_time >= TIME_US:
                done = True
                break




