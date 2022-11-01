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

uspace_x  = np.loadtxt("uspace_contender/uspace.out",dtype=float, delimiter=',',skiprows=0,usecols=(0,))
uspace_tput = np.loadtxt("uspace_contender/uspace.out",dtype=float, delimiter=',',skiprows=0,usecols=(1,))
kernel_x  = np.loadtxt("lake_cont_k.out",dtype=float, delimiter=',',skiprows=0,usecols=(0,))
kernel_tput  = np.loadtxt("lake_cont_k.out",dtype=float, delimiter=',',skiprows=0,usecols=(1,))

np.save("uspace_x", uspace_x)
np.save("uspace_tput", uspace_tput)
np.save("kernel_x", kernel_x)
np.save("kernel_tput", kernel_tput)

uspace_x = uspace_x[10:]
uspace_tput = uspace_tput[10:]

min = np.min(uspace_x)
if min < 0:
    uspace_x = uspace_x + abs(min)

uspace_x += 10


kernel_x = kernel_x[10:]
kernel_tput = kernel_tput[10:]

min = np.min(kernel_x)
if min < 0:
    kernel_x = kernel_x + abs(min)

kernel_x = kernel_x/pow(10, 9)

#uspace
size = uspace_x.shape[0]

uspace_cur = uspace_tput[1:]

time_cur = uspace_x[1:]
time_prev = uspace_x[0:size - 1]
time_diff = time_cur - time_prev

uspace_x_val = time_cur
uspace_y_val = uspace_cur/time_diff
uspace_y_val = uspace_y_val/np.max(uspace_y_val)

kernel_size = 3
kernel = np.ones(kernel_size) / kernel_size
uspace_y_val = np.convolve(uspace_y_val, kernel, mode='same')


#kernel
size = kernel_x.shape[0]

kernel_cur = kernel_tput[1:]

time_cur_kern = kernel_x[1:]
time_prev_kern = kernel_x[0:size - 1]
time_diff_kern = time_cur_kern - time_prev_kern

kernel_x_val = time_cur_kern
kernel_y_val = kernel_cur/time_diff_kern
kernel_y_val = kernel_y_val/np.max(kernel_y_val)

kernel_size = 3
kernel = np.ones(kernel_size) / kernel_size
kernel_y_val = np.convolve(kernel_y_val, kernel, mode='same')


rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
fig, ax = plt.subplots()

ax.plot(uspace_x_val, uspace_y_val, label='uspace', color='red', linestyle='solid')
ax.plot(kernel_x_val, kernel_y_val, label='kernel', color='grey', linestyle='solid')

fig.tight_layout()
plt.gcf().set_size_inches(5, 2.5)
# plt.xlim(0, 40)
# plt.ylim(0, 1.1)
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized Throughput')

plt.savefig('contention.pdf', dpi = 800, bbox_inches='tight', pad_inches=0)