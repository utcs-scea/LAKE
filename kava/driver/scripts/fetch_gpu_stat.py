#!/usr/bin/python3

import six
import pynvml as nv
from time import sleep
import signal
# import arrow
# import psutil
from contextlib import contextmanager
from pynvml import nvmlInit, nvmlShutdown


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


def device_status(device_index):
    handle = nv.nvmlDeviceGetHandleByIndex(device_index)
    device_name = nv.nvmlDeviceGetName(handle)
    if six.PY3:
        device_name = device_name.decode('UTF-8')
    # nv_procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
    utilization = nv.nvmlDeviceGetUtilizationRates(handle).gpu
    if six.PY3:
        clock_mhz = nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)
    else:
        # old API in nvidia-ml-py
        clock_mhz = nv.nvmlDeviceGetClock(handle, nv.NVML_CLOCK_SM, 0)
    temperature = nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)
    # pids = []
    # users = []
    # dates = []
    # cmd = None
    # for nv_proc in nv_procs:
    #     pid = nv_proc.pid
    #     pids.append(pid)
    #     try:
    #         proc = psutil.Process(pid)
    #         users.append(proc.username())
    #         dates.append(proc.create_time())
    #         if cmd is None:
    #             cmd = parse_cmd_roughly(proc.cmdline())
    #     except psutil.NoSuchProcess:
    #         users.append('?')
    return {
        'type': device_name,
        # 'is_available': len(pids) == 0,
        # 'pids': ','.join([str(pid) for pid in pids]),
        # 'users': ','.join(users),
        # 'running_since': arrow.get(min(dates)).humanize() if len(dates) > 0 else None,
        'utilization': utilization,
        'clock_mhz': clock_mhz,
        'temperature': temperature,
        # 'cmd': cmd,
        }


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser("gpu stats")
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
    fp.write("t,util,clock_mhz,temp,\n")
    with nvml_context():
        device_count = nv.nvmlDeviceGetCount()
        while True:
            fp.write("{:.2f},".format(t))
            for device_index in range(device_count):
                info = device_status(device_index)
                fp.write("{},{},{},".format(info["utilization"],
                                            info["clock_mhz"],
                                            info["temperature"]))
            fp.write("\n")
            fp.flush()
            t += PERIOD_SECS
            sleep(PERIOD_SECS)
