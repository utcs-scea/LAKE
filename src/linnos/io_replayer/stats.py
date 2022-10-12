#!/usr/bin/env python3
#title           :statistics.py
#author          :Vincentius Martin
#==============================================================================

from operator import itemgetter
from re import I
import sys, math
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Need path to log file to parse")

    sorted_io = []
    inters = []

    readbandwidth = 0
    readlatency = 0
    totalread = 0
    writebandwidth = 0
    writelatency = 0
    totalwrite = 0

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
            readbandwidth += (io[3]/1024) / (io[1]/1000000.0)
            readlatency += io[1]
            totalread += 1
        else: #write
            div = 1 if (io[1]/1000000.0) == 0 else (io[1]/1000000.0)
            writebandwidth += (io[3]/1024) / div
            writelatency += io[1]
            totalwrite += 1
            if last_write_time != -1:
                inter_write_time += io[0] - last_write_time
            last_write_time = io[0]

        if last_io_time != -1:
            inter = io[0] - last_io_time
            inters.append(inter)
            inter_io_time += io[0] - last_io_time
        last_io_time = io[0]

    print ("==========Statistics==========")
    print(f"total read {totalread} totalwrite {totalwrite} ")
    print (f"Last time {str(last_io_time)}")
    print (f"IO inter arrival time average {(inter_io_time / (totalread + totalwrite - 1)):.2f}ms")
    print (f"Write inter arrival time average {(inter_write_time / (totalwrite - 1)):.2f}")
    print (f"Min/Max inter arrival time  {min(inters)}, {max(inters)}")
    print (f"Total writes: {str(totalwrite)}")
    print (f"Total reads: {str(totalread)}")
    print (f"Write iops: {(float(totalwrite) / (last_io_time / 1000)):.2f}")
    print (f"Read iops: {(float(totalread) / (last_io_time / 1000)):.2f}")
    print (f"Average write bandwidth: {(writebandwidth / totalwrite):.2f} KB/s")
    print (f"Average write latency: {(writelatency / totalwrite):.2f} us")
    print (f"Average read bandwidth: {(readbandwidth / totalread):.2f} KB/s")
    print (f"Average read latency: {(readlatency / totalread):.2f} us")
    print (f"==============================")

    count, x = np.histogram(inters, bins=500)

    print("Inter arrival histogram (ms):")
    #for i in range(len(x)-1):
    for i in range(30):
        print(f"{x[i]}: {count[i]}")

