#!/usr/bin/python3

import psutil
import time
import sys
import os
import signal
import subprocess


def findProcessIdByName(processName):
    '''
    Get a list of all the PIDs of a all the running process whose name contains
    the given string processName
    '''

    listOfProcessObjects = []

    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name', 'create_time'])
            # Check if process name starts with the given name string.
            if pinfo['name'].lower().startswith(processName.lower()):
                listOfProcessObjects.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return listOfProcessObjects


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', type=str, default="stats")
    parser.add_argument('-p', type=str, default="")
    parser.add_argument('-d', type=str, default="./")
    parser.add_argument('-n', '--names-list', nargs='+', default=["worker"])
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true')
    parser.set_defaults(gpu=False)
    parser.add_argument('-s', '--sys', dest='sys', action='store_true')
    parser.set_defaults(sys=False)
    args = parser.parse_args()
    pids = []
    for pname in args.names_list:
        pids.extend(findProcessIdByName(pname))
    print(pids)

    fp = open(args.d + args.o + args.p + ".txt", 'w')

    def sig_int(signal, frame):
        if args.gpu:
            subprocess.run("kill -2 {}".format(PROC.pid), shell=True)
        fp.flush()
        fp.close()
        sys.exit(0)

    # GPU
    if args.gpu:
        script_path = os.path.dirname(os.path.realpath(__file__)) + "/fetch_gpu_stat.py"
        PROC = subprocess.Popen(["python3", script_path, args.d + "gpu_stats" + args.p + ".txt"])

    # Throughput
    if args.sys:
        script_path = os.path.dirname(os.path.realpath(__file__)) + "/fetch_sysfs_stat.py"
        PROC = subprocess.Popen(["python3", script_path, args.d + "sysfs_stats" + args.p + ".txt"])

    signal.signal(signal.SIGINT, sig_int)
    while True:
        for pid in pids:
            proc_pid_path = os.path.join('/proc', str(pid['pid']), 'task')
            ts = time.perf_counter()
            fp.write("{},".format(ts))
            for task_dir in os.listdir(proc_pid_path):
                with open(os.path.join(proc_pid_path, task_dir, 'stat'), 'r') as f:
                    t_stat = f.read().strip()
                    fp.write("{},".format(t_stat))
            fp.write("||")
        fp.write('\n')
        fp.flush()
        time.sleep(0.1)
