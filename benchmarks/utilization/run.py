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

#hackvm
#DRIVE="vdc"
#ROOT_DIR="/mnt/vdc/crypto"

#santacruz nvme
DRIVE="nvme1n1"
ROOT_DIR="/mnt/nvme1/crypto"

READAHEAD_MULTIPLIER = 1

if os.geteuid() != 0:
    exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

print(f"Script will run on drive {DRIVE}, mounting at dir {ROOT_DIR}\n")
print(f"Please make sure that {ROOT_DIR} is at drive {DRIVE}.")
print("You can do this by running sudo lsblk and sudo fdisk -l (dont append the partition number. e.g. for /dev/sda3 set DRIVE to sda")
print("To check if the dir is in that drive, run sudo df -h . If you use lvm, run sudo pvdisplay -m\n")
user_ok = input("Is this correct? y/n ")
if user_ok == "y":
    pass
else:
    print("Quiting..")
    sys.exit(0)

this_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
src_dir   = os.path.join(this_path, "..", "..", "src", "ecryptfs")
ecryptfs_dir = os.path.join(src_dir, "ecryptfs")
crypto_dir   = os.path.join(src_dir, "crypto")

#TODO: check for reader

#full check
if os.geteuid() != 0:
    exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

if not os.path.isfile(os.path.join(ecryptfs_dir, "ecryptfs.ko")):
    print(f"ecryptfs.ko not found. run make in {ecryptfs_dir}")
    sys.exit(1)

if not os.path.isfile(os.path.join(ecryptfs_dir, "lake_ecryptfs.ko")):
    print(f"lake_ecryptfs.ko not found. run make in {ecryptfs_dir}")
    sys.exit(1)

if not os.path.isfile("./tools/cpu_gpu"):
    print(f"./tools/cpu_gpu not found. run make in ./tools")
    sys.exit(1)

if not os.path.isfile(os.path.join(crypto_dir, "lake_gcm.ko")):
    print(f"lake_gcm.ko not found. run make in {crypto_dir}")
    sys.exit(1)

if not os.path.isfile(os.path.join(crypto_dir, "gcm_kernels.cubin")):
    print(f"gcm_kernels.cubin not found. run make in {crypto_dir}")
    sys.exit(1)

def load_ecryptfs(modname="ecryptfs.ko"):
    p = os.path.join(ecryptfs_dir, modname)
    r = run(f"sudo insmod {p}", shell=True)
    if r.returncode < 0: 
        print(f"Error {r.returncode} inserting mod {p}")
        sys.exit(1)
    print(f"Inserted {p}")

def load_lake_ecryptfs():
    load_ecryptfs(modname="lake_ecryptfs.ko")

def load_cpu_crypto():
    run("sudo modprobe -r aesni_intel", shell=True)

def load_aesni_crypto():
    run("sudo modprobe aesni_intel", shell=True)

def load_lake_crypto(aesni_fraction=0):
    run("sudo modprobe aesni_intel", shell=True)
    p = os.path.join(crypto_dir, "lake_gcm.ko")
    cubin = os.path.join(crypto_dir, "gcm_kernels.cubin")
    r = run(f"sudo insmod {p} cubin_path={cubin} aesni_fraction={aesni_fraction}", shell=True)
    if r.returncode < 0: 
        print(f"Error {r.returncode} inserting mod {p}")
        sys.exit(1)

def mount_gcm(path, cipher="gcm"):
    os.makedirs(f"{path}_enc", exist_ok=True)
    os.makedirs(f"{path}_plain", exist_ok=True)
    cmd = ("sudo mount -t ecryptfs -o key=passphrase:passphrase_passwd=111,"
        "ecryptfs_cipher_mode={cipher},no_sig_cache," #verbose,"
        "ecryptfs_cipher=aes,ecryptfs_key_bytes=32,ecryptfs_passthrough=n,ecryptfs_enable_filename_crypto=n"
        " {path}_enc {path}_plain")
    cmd = cmd.format(cipher=cipher, path=path, this=this_path)

    r = run(cmd, shell=True) #, input="111\n", universal_newlines=True)
    if r.returncode != 0:
        print(f"Error mounting dir with cmd {cmd}")
        sys.exit(1)

def mount_lakegcm(path):
    mount_gcm(path, "lake_gcm")

def umount(path):
    run(f"sudo umount {path}_plain", shell=True, stdout=DEVNULL)
    sleep(0.5)
    run(f"sudo umount {path}_enc", shell=True)
    sleep(0.5)
    run(f"sudo rm -rf {path}_enc", shell=True, stdout=DEVNULL)
    run(f"sudo rm -rf {path}_plain", shell=True, stdout=DEVNULL)

def set_readhead(bsize):
    run(f"echo {bsize} | sudo tee /sys/block/{DRIVE}/queue/read_ahead_kb", shell=True, stdout=DEVNULL)

def to_bytes(sz):
    sz = sz.strip()
    mag = 1
    if "m" in sz or "M" in sz:
        mag = 1024*1024
    elif "k" in sz or "K" in sz:
        mag = 1024
    raw = sz[:-1]
    return int(raw)*mag

def reset():
    run("sudo rmmod ecryptfs", shell=True)
    run("sudo rmmod lake_ecryptfs", shell=True)
    run("sudo rmmod lake_gcm", shell=True)
    run("sudo modprobe -r aesni_intel", shell=True)
    run(f"echo 4096 | sudo tee /sys/block/{DRIVE}/queue/read_ahead_kb", shell=True, stdout=DEVNULL)

def run_benchmark(fpath):
    bsize = "2m"
    bsize = to_bytes(bsize)
    set_readhead(READAHEAD_MULTIPLIER*bsize)
    sleep(0.5)
    subprocess.check_call(f"sudo ./tools/write_data {fpath}", shell=True)
    sleep(5)
    proc = subprocess.Popen("exec ./tools/cpu_gpu", shell=True)
    sleep(10)
    #("reading")
    start = timer()
    subprocess.check_call(f"sudo ./tools/read_data {fpath}", shell=True)
    end = timer()
    print(f"done, read took {end-start}s")
    sleep(3) # give it some time to settle
    proc.kill()
    sleep(0.5)
    try:
        subprocess.check_call(f"sudo pkill -9 -f cpu_gpu", shell=True)
    except:
        pass
    sleep(1)

tests = {
    "CPU": {
        "cryptomod_fn": load_cpu_crypto,
        "fsmod_fn": load_ecryptfs,
        "mount_fn": mount_gcm,
        "mount_basepath": os.path.join(ROOT_DIR, "cpu")
    },
    "AESNI": {
       "cryptomod_fn": load_aesni_crypto,
       "fsmod_fn": load_ecryptfs,
       "mount_fn": mount_gcm,
       "mount_basepath": os.path.join(ROOT_DIR, "cpu")
    },
    "LAKE": {
        "cryptomod_fn": load_lake_crypto,
        "fsmod_fn": load_lake_ecryptfs,
        "mount_fn": mount_lakegcm,
        "mount_basepath": os.path.join(ROOT_DIR, "lake")
    },
}

x = []
cpu = []
lake_cpu = []
lake_gpu = []
lake_api = []
aes_ni = []


reset()
for name, args in tests.items():
    umount(args["mount_basepath"])
    reset()
    sleep(1)
    #load correct crypto
    args["cryptomod_fn"]()
    #load ecryptfs
    args["fsmod_fn"]()
    sleep(0.5)
    #mount enc and plain dir
    args["mount_fn"](args["mount_basepath"])
    #print("mounted")
    sleep(1)

    fpath = args["mount_basepath"]+"_plain/test.dat"
    run_benchmark(fpath)

    if name == "CPU":
        x = np.loadtxt('tmp.out', dtype=float, delimiter=',',skiprows=0,usecols=(0,))
        np.save("x_data.bin", x)
        x = x/1000
        x = x[np.where(x > 5)] # we wait 5 seconds, cut the first 2
        x = x - x[0] #offset to zero
        #print("x: ", x)
        cpu =  np.loadtxt('tmp.out', dtype=float, delimiter=',',skiprows=0,usecols=(1,))
        np.save("cpu_data.bin", cpu)
        #print(f"cpu: ", cpu)
        cpu = cpu[-x.shape[0]:]
        #print(f"cpu sliced: ", cpu)
    if name == "AESNI":
        aes_ni = np.loadtxt('tmp.out', dtype=float, delimiter=',',skiprows=0,usecols=(1,))
        np.save("aes_ni_data.bin", aes_ni)
        #print("aes: ", aes_ni)
        aes_ni = aes_ni[-x.shape[0]:]
        pad = x.shape[0] - aes_ni.shape[0]
        if pad >0:
            aes_ni = np.pad(aes_ni, (0, pad), 'constant')
        #print("aes sliced: ", aes_ni)
    if name == "LAKE":
        lake_cpu = np.loadtxt('tmp.out', dtype=float, delimiter=',',skiprows=0,usecols=(1,))
        lake_gpu = np.loadtxt('tmp.out', dtype=float, delimiter=',',skiprows=0,usecols=(2,))
        lake_api = np.loadtxt('tmp.out', dtype=float, delimiter=',',skiprows=0,usecols=(3,))
        np.save("lake_cpu_data.bin", lake_cpu)
        np.save("lake_gpu_data.bin", lake_gpu)
        np.save("lake_api_data.bin", lake_api)

        lake_cpu = lake_cpu[-x.shape[0]:]
        pad = x.shape[0] - lake_cpu.shape[0]
        if pad >0:
            lake_cpu = np.pad(lake_cpu, (0, pad), 'constant')

        lake_gpu = lake_gpu[-x.shape[0]:]
        pad = x.shape[0] - lake_gpu.shape[0]
        if pad >0:
            lake_gpu = np.pad(lake_gpu, (0, pad), 'constant')

        lake_api = lake_api[-x.shape[0]:]
        pad = x.shape[0] - lake_api.shape[0]
        if pad >0:
            lake_api = np.pad(lake_api, (0, pad), 'constant')


    #run(f"sudo rm tmp.out", shell=True, stdout=DEVNULL)
    sleep(1)
    run(f"sudo rm {fpath}", shell=True, stdout=DEVNULL)
    sleep(0.5)
    umount(args["mount_basepath"])
    sleep(0.5)
    reset()
    sleep(1)

cmap = matplotlib.cm.get_cmap("tab10")
fig, ax = plt.subplots()

ax.plot(x, cpu, label='CPU', color=cmap(0) )#, marker="x", markersize=3)
ax.plot(x, aes_ni, label='AES-NI', color=cmap(1), marker="o", markersize=3)
ax.plot(x, lake_cpu, label='LAKE CPU', color=cmap(2), marker="v", markersize=3)
ax.plot(x, lake_api, label='LAKE API', color=cmap(3), marker="s", markersize=3)
ax.plot(x, lake_gpu, label='LAKE GPU', color='black', linewidth=1, marker='.')
#     linestyle='dotted', linewidth=3)
    
fig.tight_layout()
plt.xlim(left=0)
plt.ylim(bottom=0)

ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(1.25, 1), columnspacing=1)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Utilization (\%)')

ax.grid(visible=True, which='major', axis='y', color='#0A0A0A', linestyle='--', alpha=0.2)

fig.set_size_inches(3.5, 2)
fig.set_dpi(200)

plt.savefig('ecryptfs_util.pdf', bbox_inches='tight',pad_inches=0.05)