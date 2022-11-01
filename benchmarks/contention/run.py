# Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
# Copyright (C) 2022-2024 Henrique Fingler
# Copyright (C) 2022-2024 Isha Tarte
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import re, os, sys
from subprocess import run, DEVNULL
from time import sleep
import os.path
import subprocess
import signal
import subprocess
from tokenize import Double
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from timeit import default_timer as timer

if os.geteuid() != 0:
    exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

print("We're are goin to make sure things are built..")

kapi = subprocess.check_output("pidof lake_uspace", shell=True)
kapi = kapi.decode("ascii").strip()
if kapi == "":
    print("I didn't find kapi running, did you execute sudo ./load.sh?")
    sys.exit(1)

if not os.path.isfile("uspace_contender/test_uspace"):
    print(f"uspace_contender/test_uspace not found. run make in ./uspace_contender")
    sys.exit(1)

if not os.path.isfile("../../src/linnos/linnos_cont.ko"):
    print(f"Linnos module not found, run make in src/linnos/")
    sys.exit(1)

def run_benchmark():
    print("Launching linnos")
    subprocess.Popen(f"cd ../../src/linnos && sudo ./run_cont_40.sh", shell=True)
    sleep(10)
    print("Launching hasher")
    subprocess.Popen(f"cd uspace_contender && ./test_uspace", shell=True)
    sleep(40) #wait for everyone to finish

def parse_klog():
    lines = []
    klog = subprocess.check_output(f"sudo journalctl -k | tail -n 10000", shell=True)
    klog = klog.decode("ascii")

    for line in reversed(klog.splitlines()):
        if "-----start-----" in line:
            break
        if "lakecont" in line:
            m = re.search("lakecont,(\d+), (\d+)", line) 
            l = f"{m.group(1)},{m.group(2)}\n"
            lines.append(l)

    lines.reverse()

    with open("lake_cont_k.out", "w") as f:
        f.writelines(lines)

run_benchmark()
parse_klog()

uspace_x  = np.loadtxt("uspace_contender/uspace.out",dtype=float, delimiter=',',skiprows=1,usecols=(0,))
uspace_tput = np.loadtxt("uspace_contender/uspace.out",dtype=float, delimiter=',',skiprows=1,usecols=(1,))
kernel_x  = np.loadtxt("lake_cont_k.out",dtype=float, delimiter=',',skiprows=0,usecols=(0,))
kernel_tput  = np.loadtxt("lake_cont_k.out",dtype=float, delimiter=',',skiprows=0,usecols=(1,))

with open("uspace_contender/uspace.out") as f:
    uspace_ts = f.readline()
uspace_ts = np.array([int(x) for x in uspace_ts.split(",")])

np.save("uspace_ts", uspace_ts)
np.save("uspace_x", uspace_x)
np.save("uspace_tput", uspace_tput)
np.save("kernel_x", kernel_x)
np.save("kernel_tput", kernel_tput)

