#!/usr/bin/env python3.8
import re, os, sys
from subprocess import run
from time import sleep

final = "Final average results:"
read_pat = "Read Sequential\s*(\S*)"
write_pat = "Write Sequential\s*(\S*)"

this_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))
src_dir   = os.path.join(this_path, "..", "..", "src", "ecryptfs")
ecryptfs_dir = os.path.join(src_dir, "ecryptfs")
crypto_dir   = os.path.join(src_dir, "crypto")
fileio_dir   = os.path.join(src_dir, "file_io")

with open(os.path.join(this_path, "test.env")) as f:
    mnt_dir = f.readline()

def load_ecryptfs(modname="ecryptfs.ko"):
    p = os.path.join(ecryptfs_dir, modname)
    r = run(f"sudo insmod {p}", shell=True)
    if r.returncode < 0: 
        print(f"Error {r.returncode} inserting mod {p}")
        sys.exit(1)

def load_lake_ecryptfs():
    load_ecryptfs("lake_ecryptfs.ko")

def load_cpu_crypto():
    run("sudo modprobe -r aesni_intel", shell=True)

def mount_gcm(path, cipher="gcm"):
    os.makedirs(f"{path}_enc", exist_ok=True)
    os.makedirs(f"{path}_plain", exist_ok=True)
    cmd = ("sudo mount -t ecryptfs -o key=passphrase:passphrase_passwd=111,"
        "ecryptfs_cipher_mode={cipher},no_sig_cache,verbose,"
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
    run(f"sudo umount {path}_plain", shell=True)
    sleep(0.5)
    run(f"sudo rm -rf {path}_enc", shell=True)
    run(f"sudo rm -rf {path}_plain", shell=True)

def reset():
    run("sudo rmmod ecryptfs.ko", shell=True)
    run("sudo rmmod lake_ecryptfs.ko", shell=True)
    run("sudo rmmod lake_gcm", shell=True)
    run("sudo modprobe -r aesni_intel", shell=True)
    run("echo 4096 | sudo tee /sys/block/vda/queue/read_ahead_kb", shell=True)

def run_benchmark(p, sz):
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
    "cpu": {
        "cryptomod_fn": load_cpu_crypto,
        "fsmod_fn": load_ecryptfs,
        "mount_fn": mount_gcm,
        "mount_basepath": os.path.join(mnt_dir, "cpu")
    }
}

sizes = {
    "4K": "1 256k 4k",
    "8K": "1 512k 8k",
    "16K": "1 1m 16k",
    "32K": "1 2m 32k",
    "64K": "1 4m 64k",
    "128K": "1 8m 128k",
    "256K": "1 16m 256k",
    "512K": "1 32m 512k",
    "1M": "1 64m 1m",
    "2M": "1 128m 2m",
    "4M": "1 256m 4m",
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
        print("loaded")
        #mount enc and plain dir
        args["mount_fn"](args["mount_basepath"])
        print("mounted")

        out = run_benchmark(args["mount_basepath"]+"_plain", sz)
        rd, wt = parse_out(out)        

        results[name]["rd"].append(rd)
        results[name]["wt"].append(wt)

        sleep(1)
        umount(args["mount_basepath"])
        sleep(1)


print("," + ",".join(sizes.keys()))
for name in tests.keys():
    print(f"{name}_rd, {','.join(results[name]['rd'])}")
    print(f"{name}_rd, {','.join(results[name]['wt'])}")