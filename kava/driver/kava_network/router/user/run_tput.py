#!/usr/bin/env python3
import os, sys, re
import subprocess
from pathlib import Path

base_cmd = "./router runtime=15 cubin_path={cubin} batch={batch} input_throughput=0 block_size=32 numrules=100 {seq}"
script_dir = Path( __file__ ).parent.absolute()
cubin = os.path.join(script_dir, "firewall.cubin")

#batches = [16,32,64,128,256,512,1024,2048,4096,8192]

batches = [32,256]
types = {"gpu": "",
        "cpu": "-sequential"}

for name, flag in types.items():
    for b in batches:
        cmd = base_cmd.format(cubin=cubin, batch=b, seq=flag)
        
        out = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
        output = out.stdout.decode() 
        
        pat = "==> (.*) Gbps"
        m =re.search(pat, output)
        print(f"{name}_{b}, {m.group(1)}")