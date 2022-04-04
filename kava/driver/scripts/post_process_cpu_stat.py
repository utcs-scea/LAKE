#!/usr/bin/python3
import sys
import re
import os
from statistics import mean
import numpy as np
from collections import defaultdict

def post_process_cpu_stat(i_file, o_file):
    if o_file == 'stdout':
        fp = sys.stdout
    else:
        fp = open(o_file, 'w')
    previous = {}
    t_axis = 0.0
    summary_time = {}
    time_list = []
    fnname_set = set()
    with open(i_file, 'r') as f:
        for line in f:
            groups = line.strip().split("||")
            num_groups = len(groups) - 1
            t_axis_str = ""
            for g in range(num_groups):
                stats = groups[g].split(",")
                stats_len = len(stats)
                ts = float(stats[0])
                times = []

		# Check process type
                fnname = stats[1].split()[1][1:-1]

                # Sum up time
                for i in range(1, stats_len-1):
                    stat = stats[i]
                    stat_splits = stat.split()
                    if fnname in ["kavad", "ksmd", "insmod", "fs_bench"]:
                        # Kernel space
                        scheduled_time = float(stat_splits[14])
                    else:
                        # User space
                        scheduled_time = float(stat_splits[13])
                    times.append(scheduled_time)

                times = np.array(times)
                sum_time = np.sum(times)
                if g not in previous:
                    if g == 0:
                        fp.write("time,")
                    fnname_set.add(fnname)
                    previous[g] = [ts, fnname, sum_time]
                    fp.write("{},".format(fnname))
                else:
                    prev_ts = previous[g][0]
                    fnname = previous[g][1]
                    prev_sum_time = previous[g][2]
                    diff_t = (ts - prev_ts) * 1000
                    if g == 0:
                        t_axis_str = "{:.2f},".format(t_axis)
                        summary_time[t_axis_str] = defaultdict(list)
                        fp.write(t_axis_str)
                        time_list.append(t_axis_str)
                        t_axis += ts - prev_ts
                    g_percent = (sum_time - prev_sum_time) * 10 / diff_t * 100
                    summary_time[t_axis_str][fnname].append(g_percent)
                    # percent = ",".join("{:.2f}".format(x) for x in g_percent)
                    fp.write("{:.2f},".format(g_percent))
                    # fp.write("{},{},".format(diff_t, (times - prev_times)))
                    previous[g][2] = sum_time
                    previous[g][0] = ts
            fp.write("\n")

    fp.close()
    dirname = os.path.dirname(o_file)
    ofname = os.path.basename(o_file)

    with open(os.path.join(dirname, 'mean_'+ofname), 'w') as f:
        f.write("time,")
        for fn in fnname_set:
            f.write("{},".format(fn))
        f.write("\n")
        for t in time_list:
            f.write("{}".format(t))
            for fn in fnname_set:
                f.write("{},".format(mean(summary_time[t][fn])))
            f.write("\n")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default="stats.txt")
    parser.add_argument('-o', type=str, default="stdout")
    args = parser.parse_args()

    post_process_cpu_stat(args.i, args.o)
