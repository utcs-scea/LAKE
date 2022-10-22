import re, csv, sys
from subprocess import check_output


KLOG_LINES = 200

klog = check_output(f"sudo journalctl -k | tail -n {KLOG_LINES}", shell=True)
klog = klog.decode("ascii")

gpudata = {}
cpu = {}
gpucomp = {}

for line in reversed(klog.splitlines()):
    print (line)
    if "GPU_batch_" in line:
        #format:  GPU_batch_X,2,4     (2 is comp, 4 is comp+data)
        m = re.search("GPU\_batch\_(\d+),(\d+),(\d+)", line) 
        if not m:
            print("error parsing GPU string: ", line)
            sys.exit(1)
        gpucomp[int(m.group(1))] = int(m.group(2))
        gpudata[int(m.group(1))] = int(m.group(3))

    if "CPU_batch_" in line:
        #format:  CPU_batch_X,2
        m = re.search("CPU\_batch\_(\d+),(\d+)", line) 
        if not m:
            print("error parsing CPU string: ", line)
            sys.exit(1)
        cpu[int(m.group(1))] = int(m.group(2))


#linnos.csv: ,gpu,cpu,gpu_data

with open('linn.csv') as f:
    lines = f.readlines()

outlines = []
header = f"{lines[0].rstrip()},ae_gpu,ae_cpu,ae_gpu_data\n"
outlines.append(header)
for line in lines[1:]:
    bsize = int(line.split(',')[0])
    added_line = f"{line.rstrip()},{gpudata[bsize]},{cpu[bsize]},{gpucomp[bsize]}\n"
    outlines.append(added_line)


with open('tmp.csv', 'w') as f:
    for l in outlines:
        f.write(l)
