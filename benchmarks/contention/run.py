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

# cmap = matplotlib.cm.get_cmap("tab10")
# fig, ax = plt.subplots()

# #Padding with 0
# pad = x.shape[0] - cpu.shape[0]
# if pad >0:
#     cpu = np.pad(cpu, (0, pad), 'constant')

# pad = x.shape[0] - lake_cpu.shape[0]
# if pad >0:
#     lake_cpu = np.pad(lake_cpu, (0, pad), 'constant')

# pad = x.shape[0] - aes_ni.shape[0]
# if pad >0:
#     aes_ni = np.pad(aes_ni, (0, pad), 'constant')

# pad = x.shape[0] - lake_gpu.shape[0]
# if pad >0:
#     lake_gpu = np.pad(lake_gpu, (0, pad), 'constant')

# pad = x.shape[0] - lake_api.shape[0]
# if pad >0:
#     lake_api = np.pad(lake_api, (0, pad), 'constant')

# #cutiing the array
# index = 0
# for ele in x:
#     if ele > 8:
#         break
#     index += 1

# x = x[index:]
# x = x - x[0] # normalizing
# cpu = cpu[index:]
# aes_ni = aes_ni[index:]
# lake_cpu = lake_cpu[index:]
# lake_gpu = lake_gpu[index:]
# lake_api = lake_api[index:]

# #smoothening the grpah
# kernel_size = 3
# kernel = np.ones(kernel_size) / kernel_size
# cpu = np.convolve(cpu, kernel, mode='same')
# aes_ni = np.convolve(aes_ni, kernel, mode='same')
# lake_cpu = np.convolve(lake_cpu, kernel, mode='same')
# lake_api = np.convolve(lake_api, kernel, mode='same')
# lake_gpu = np.convolve(lake_gpu, kernel, mode='same')

# ax.plot(x, cpu, label='CPU', color=cmap(0) )#, marker="x", markersize=3)
# ax.plot(x, aes_ni, label='AES-NI', color=cmap(1), marker="o", markersize=3)
# ax.plot(x, lake_cpu, label='LAKE CPU', color=cmap(2), marker="v", markersize=3)
# ax.plot(x, lake_api, label='LAKE API', color=cmap(3), marker="s", markersize=3)
# ax.plot(x, lake_gpu, label='LAKE GPU', color='black', linewidth=1, marker='.')
# #     linestyle='dotted', linewidth=3)
    
# fig.tight_layout()
# plt.xlim(left=0, right=x[-5])
# plt.ylim(bottom=0)

# ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(1.25, 1), columnspacing=1)

# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Utilization (%)')

# ax.grid(visible=True, which='major', axis='y', color='#0A0A0A', linestyle='--', alpha=0.2)

# fig.set_size_inches(3.5, 2)
# fig.set_dpi(200)

# plt.savefig('ecryptfs_util.pdf', bbox_inches='tight',pad_inches=0.05)