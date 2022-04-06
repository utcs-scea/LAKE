#!/usr/bin/python3

import six
import time
from time import sleep
import signal
# import arrow
# import psutil
from contextlib import contextmanager


@contextmanager
def nvml_context():
    nvmlInit()
    yield
    nvmlShutdown()


def parse_cmd_roughly(args):
    cmdline = ' '.join(args)
    if 'python -m ipykernel_launcher' in cmdline:
        return 'jupyter'
    python_script = [arg for arg in args if arg.endswith('.py')]
    if len(python_script) > 0:
        return python_script[0]
    else:
        return cmdline if len(cmdline) <= 25 else cmdline[:25] + '...'

def get_tpt():
  with open('/sys/kernel/mm/ksm/throughput','r') as f:
    return f.read().strip()

def get_total_scan_time():
  with open('/sys/kernel/mm/ksm/scan_time','r') as f:
    return f.read().strip()

if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser("sysfs stats")
    parser.add_argument("ofile", type=str)
    args = parser.parse_args()
    fp = None
    if args.ofile == "stdout":
        fp = sys.stdout
    else:
        fp = open(args.ofile, "w")

    def sig_int(signal, frame):
        print("get signal")
        fp.flush()
        fp.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_int)
    t = 0.0
    PERIOD_SECS = 0.1
    fp.write("t,pages_per_sec,total_time_100_scans,\n")
    while True:
        fp.write("{},{},{},".format(time.perf_counter(),
                                    get_tpt(),
                                    get_total_scan_time()))
        fp.write("\n")
        fp.flush()
        t += PERIOD_SECS
        sleep(PERIOD_SECS)
