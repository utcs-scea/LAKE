#!/usr/bin/env python3
import os, sys, re
import subprocess
from pathlib import Path

base_cmd = "./router runtime=15 cubin_path={cubin} batch={batch} input_throughput=0 block_size=32 numrules=100 {seq}"
script_dir = Path( __file__ ).parent.absolute()
cubin = os.path.join(script_dir, "firewall.cubin")

batches = [1024, 2048, 4096, 8192]
types = {"gpu": ""}

for name, flag in types.items():
    for b in batches:
        cmd = base_cmd.format(cubin=cubin, batch=b, seq=flag)
        
        out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        output = out.stdout.decode() 

        pat = "Average latency: Max: (.*) usec, Min: (.*) usec"
        m =  re.search(pat, output)
        minlat, maxlat = m.group(1), m.group(2)
        print(f"{name}_{b}, {minlat}, {maxlat}")