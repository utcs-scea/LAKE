#!/usr/bin/env python3
#title           :statistics.py
#author          :Vincentius Martin
#==============================================================================

from operator import itemgetter
from re import I
import sys, math
import numpy as np
import statistics
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Need path to log file to parse")

    sorted_io = []

    inter_arrivals = []
    read_latencies = []
    read_sizes = []
    write_latencies = []

    # 1: timestamp in us
    # 2: latency in us
    # 3: r/w type [0 for r, 1 for w]
    # 4: I/O size in bytes
    # 5: offset in bytes
    # 6: IO submission time (not used)*/

    last_io_time = -1
    last_write_time = -1
    inter_io_time = 0
    inter_write_time = 0

    with open(sys.argv[1]) as f:
        for line in f:
            if line == "\n": break
            tok = map(str.strip, line.split(","))
            tok_list = list(tok)
            sorted_io.append([float(tok_list[0]), int(tok_list[1]),float(tok_list[2]),
                int(tok_list[3]),float(tok_list[4])])

    for io in sorted(sorted_io, key=itemgetter(0)):
        if (io[2] == 1): #read
            read_latencies.append(io[1])
            read_sizes.append(io[3])

        if (io[2] == 0): #write
            write_latencies.append(io[1])

        if last_io_time != -1:
            inter = io[0] - last_io_time
            inter_arrivals.append(inter)
        last_io_time = io[0]

    np_read_latencies = np.array(read_latencies)

    print ("==========Statistics==========")
    print(f"Total reads {len(read_latencies)}")
    print (f"IO inter arrival time average {statistics.mean(inter_arrivals):.2f}us")
    print (f"Min/Max inter arrival time  {min(inter_arrivals)}, {max(inter_arrivals)}")
    print (f"Total writes: {len(write_latencies)}  percent: {len(write_latencies)/len(inter_arrivals)+1}")
    print (f"Total reads: {len(read_latencies)}")
    #print (f"Write iops: {(float(totalwrite) / (last_io_time / 1000)):.2f}")
    #print (f"Read iops: {(float(totalread) / (last_io_time / 1000)):.2f}")
    #print (f"Average write bandwidth: {(writebandwidth / totalwrite):.2f} KB/s")
    print (f"Average write latency: {statistics.mean(write_latencies):.2f} us")
    #print (f"Average read bandwidth: {(readbandwidth / totalread):.2f} KB/s")
    print (f"Median/Stddev read latency: {statistics.median(read_latencies):.2f} us / {statistics.pstdev(read_latencies):.2f} us")
    print (f"Min/Max read latency: {min(read_latencies):.2f} us / {max(read_latencies):.2f} us")
    print (f"Average read latency: {statistics.mean(read_latencies):.2f} us")
    print (f"Average read size: {statistics.mean(read_sizes):.2f} KB")
    print (f"Read latency p85: {np.percentile(np_read_latencies, 85)} us")
    print (f"Read latency p95: {np.percentile(np_read_latencies, 95)} us")
    print (f"Read latency p99: {np.percentile(np_read_latencies, 99)} us")
    print (f"Read latency p99.5: {np.percentile(np_read_latencies, 99.5)} us")
    print (f"Read latency p99.9: {np.percentile(np_read_latencies, 99.9)} us")
    print (f"==============================")

    # count, x = np.histogram(inters, bins=500)
    # print("Inter arrival histogram (ms):")
    # #for i in range(len(x)-1):
    # for i in range(30):
    #     print(f"{x[i]}: {count[i]}")

    # count, bins_count = np.histogram(np_read_latencies, bins=100)  
    # # finding the PDF of the histogram using count values
    # pdf = count / sum(count)
    # # using numpy np.cumsum to calculate the CDF
    # # We can also find using the PDF values by looping and adding
    # cdf = np.cumsum(pdf)
    # plt.plot(bins_count[1:], cdf, label="CDF")


    # sort the data:
    data_sorted = np.sort(np_read_latencies)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(np_read_latencies)) / (len(np_read_latencies) - 1)
    plt.plot(data_sorted, p)

    #plt.legend()
    plt.grid(visible=True)
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF %')
    #plt.ylim(bottom=0)
    plt.title(sys.argv[1])
    plt.xlim(right=np.percentile(np_read_latencies, 99.9), left=min(np_read_latencies))
    
    plt.savefig(sys.argv[1]+"_cdf.pdf")