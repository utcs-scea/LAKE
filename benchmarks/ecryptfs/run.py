#!/usr/bin/env python3.8
import re, os, sys
from subprocess import run, DEVNULL
from time import sleep
import os.path

#TODO: check device, we only do sda and vda
DRIVE="sda"
ROOT_DIR="/disk/hfingler/crypto"

if os.geteuid() != 0:
    exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

print(f"Script will run on drive {DRIVE}, mounting at dir {ROOT_DIR}")
print(f"Please make sure that {ROOT_DIR} is at drive {DRIVE}.")
print("You can do this by running sudo fdisk -l (dont append the partition number. e.g. for /dev/sda3 set DRIVE to sda")
print("To check if the dir is in that drive, run sudo df -h . If you use lvm, run sudo pvdisplay -m\n")
user_ok = input("Is this correct? y/n  ")
if user_ok == "n":
    print("Quiting..")
    sys.exit(0)

final = "Final average results:"
read_pat = "Read Sequential\s*(\S*)"
write_pat = "Write Sequential\s*(\S*)"

this_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
src_dir   = os.path.join(this_path, "..", "..", "src", "ecryptfs")
ecryptfs_dir = os.path.join(src_dir, "ecryptfs")
crypto_dir   = os.path.join(src_dir, "crypto")
fileio_dir   = os.path.join(src_dir, "file_io")

#full check
if os.geteuid() != 0:
    exit("You need to have root privileges to run this script.\nPlease try again, this time using 'sudo'. Exiting.")

if not os.path.isfile(os.path.join(ecryptfs_dir, "ecryptfs.ko")):
    print(f"ecryptfs.ko not found. run make in {ecryptfs_dir}")
    sys.exit(1)

if not os.path.isfile(os.path.join(ecryptfs_dir, "lake_ecryptfs.ko")):
    print(f"lake_ecryptfs.ko not found. run make in {ecryptfs_dir}")
    sys.exit(1)

if not os.path.isfile(os.path.join(fileio_dir, "fs_bench")):
    print(f"fs_bench not found. run make in {fileio_dir}")
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

#i didnt want to mess with partials
def load_lake_crypto_75aesni():
    load_lake_crypto(75)

def load_lake_crypto_50aesni():
    load_lake_crypto(50)

def load_lake_crypto_25aesni():
    load_lake_crypto(25)

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
    run(f"sudo umount {path}_enc", shell=True)
    sleep(0.5)
    run(f"sudo umount {path}_plain", shell=True, stdout=DEVNULL)
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

def run_benchmark(p, sz):
    bsize = sz.split()[-1]
    bsize = to_bytes(bsize)
    set_readhead(4*bsize)

    b = os.path.join(fileio_dir, "fs_bench")
    out = run(f"sudo {b} {p} {sz}", shell=True, capture_output=True, text=True)
    #print(f"running:  {b} {p} {sz}")
    #print("out: ", out.stdout)
    return out.stdout

def parse_out(out):
    capture_mode = False
    wt,val=None, None
    for line in out.splitlines():
        if final in line:
            capture_mode = True
            continue

        if capture_mode:
            match = re.search(read_pat, line)
            if match:
                val = match.group(1)
                rd = val

            match = re.search(write_pat, line)
            if match:
                val = match.group(1)
                wt = val
    return rd, wt

tests = {
    # "cpu": {
    #     "cryptomod_fn": load_cpu_crypto,
    #     "fsmod_fn": load_ecryptfs,
    #     "mount_fn": mount_gcm,
    #     "mount_basepath": os.path.join(ROOT_DIR, "cpu")
    # },
    "aesni": {
       "cryptomod_fn": load_aesni_crypto,
       "fsmod_fn": load_ecryptfs,
       "mount_fn": mount_gcm,
       "mount_basepath": os.path.join(ROOT_DIR, "cpu")
    },
    "lake": {
        "cryptomod_fn": load_lake_crypto,
        "fsmod_fn": load_lake_ecryptfs,
        "mount_fn": mount_lakegcm,
        "mount_basepath": os.path.join(ROOT_DIR, "lake")
    },
    # "lake50aesni": {
    #     "cryptomod_fn": load_lake_crypto_50aesni,
    #     "fsmod_fn": load_lake_ecryptfs,
    #     "mount_fn": mount_lakegcm,
    #     "mount_basepath": os.path.join(ROOT_DIR, "lake")
    # },
    "lake75aesni": {
        "cryptomod_fn": load_lake_crypto_75aesni,
        "fsmod_fn": load_lake_ecryptfs,
        "mount_fn": mount_lakegcm,
        "mount_basepath": os.path.join(ROOT_DIR, "lake")
    },
    # "lake25aesni": {
    #     "cryptomod_fn": load_lake_crypto_25aesni,
    #     "fsmod_fn": load_lake_ecryptfs,
    #     "mount_fn": mount_lakegcm,
    #     "mount_basepath": os.path.join(ROOT_DIR, "lake")
    # },
}

sizes = {
    #"16K": "1 1m 16k",
    #"4K": "1 1m 4k",
    #"4M": "1 512m 4m",
    
    "4K": "1 1m 4k",
    "8K": "2 2m 8k",
    "16K": "2 4m 16k",
    "32K": "2 8m 32k",
    "64K": "2 16m 64k",
    "128K": "2 32m 128k",
    "256K": "2 64m 256k",
    "512K": "2 128m 512k",
    "1M": "2 256m 1m",
    "2M": "2 512m 2m",
    "4M": "2 1024m 4m",
}

results = {}

reset()
for name, args in tests.items():
    results[name] = {"rd": [], "wt": []}
    
    for sz_name, sz in sizes.items():
        #load correct crypto
        args["cryptomod_fn"]()
        #load ecryptfs
        args["fsmod_fn"]()
        sleep(0.5)
        #mount enc and plain dir
        args["mount_fn"](args["mount_basepath"])
        print("mounted")
        sleep(1)
        
        out = run_benchmark(args["mount_basepath"]+"_plain", sz)
        rd, wt = parse_out(out)        

        results[name]["rd"].append(rd)
        results[name]["wt"].append(wt)

        sleep(0.5)
        umount(args["mount_basepath"])
        sleep(0.5)
        reset()
        sleep(1)

print("," + ",".join(sizes.keys()))
for name in tests.keys():
    print(f"{name}_rd, {','.join(results[name]['rd'])}")
    print(f"{name}_wt, {','.join(results[name]['wt'])}")