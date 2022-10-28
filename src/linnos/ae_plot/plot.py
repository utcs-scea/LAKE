import re, csv, sys
from subprocess import check_output
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

KLOG_LINES = 200

klog = check_output(f"sudo journalctl -k | tail -n {KLOG_LINES}", shell=True)
klog = klog.decode("ascii")

gpudata = {}
cpu = {}
gpucomp = {} # map of nn+ to map of dict {batch size , [gpu, gpudata ]}

for line in reversed(klog.splitlines()):
    #print (line)
    if "_GPU_" in line:
        #linnos+2_GPU_batch_128,447,489
        m = re.search("linnos\+(\d)_GPU\_batch\_(\d+),(\d+),(\d+)", line) 
        if not m:
            continue
            #print("error parsing GPU string: ", line)
            #sys.exit(1)
        if int(m.group(1)) not in gpucomp.keys():
            gpucomp[int(m.group(1))] = {}
            gpudata[int(m.group(1))] = {}

        #maybe the workload was ran twice, so we skip if we already found it
        nn_data = gpudata[int(m.group(1))]
        nn_comp = gpucomp[int(m.group(1))]
        if int(m.group(2)) in nn_data.keys():
            continue

        nn_data[int(m.group(2))] = int(m.group(3))
        nn_comp[int(m.group(2))] = int(m.group(4))


for line in reversed(klog.splitlines()):
    #print (line)
    if "_CPU_" in line:
        m = re.search("linnos\+(\d)_CPU\_batch\_(\d+),(\d+),(\d+)", line) 
        if not m:
            continue
            #print("error parsing GPU string: ", line)
            #sys.exit(1)
        if int(m.group(1)) not in cpu.keys():
            cpu[int(m.group(1))] = {}

        #maybe the workload was ran twice, so we skip if we already found it
        nn_dir = cpu[int(m.group(1))]
        if int(m.group(2)) in nn_dir.keys():
            continue

        nn_dir[int(m.group(2))] = int(m.group(3))


outlines = []
header = f",ae_gpu,ae_cpu,ae_gpu_data,ae_gpu+1,ae_cpu+1,ae_gpu_data+1,ae_gpu+2,ae_cpu+2,ae_gpu_data+2\n"
outlines.append(header)


bsize = 1

while True:
    if bsize > 1024: break
    added_line = f"{bsize},{gpudata[0][bsize]},{cpu[0][bsize]},{gpucomp[0][bsize]}"
    added_line += f",{gpudata[1][bsize]},{cpu[1][bsize]},{gpucomp[1][bsize]}"
    added_line += f",{gpudata[2][bsize]},{cpu[2][bsize]},{gpucomp[2][bsize]}\n"
    outlines.append(added_line)
    bsize *= 2

with open("tmp.dat", "w") as f:
    f.writelines(outlines)


cmap = matplotlib.cm.get_cmap("tab20c")
fig, ax = plt.subplots()

# let's make sure we get the right label
file = "tmp.dat"
with open(file) as f:
    labels = f.readline()

# e.g. 0: cpu, 1: gpu
labels = labels.split(",")
labels = [x.rstrip() for x in labels]

x =     np.loadtxt(file,dtype=str, delimiter=',',skiprows=1,usecols=(0,))
gpu0 =  np.loadtxt(file,dtype=int, delimiter=',',skiprows=1,usecols=(1,))
cpu0 = np.loadtxt(file,dtype=int, delimiter=',',skiprows=1,usecols=(2,))
gpud0 =  np.loadtxt(file,dtype=int, delimiter=',',skiprows=1,usecols=(3,))
gpu1 =  np.loadtxt(file,dtype=int, delimiter=',',skiprows=1,usecols=(4,))
cpu1 = np.loadtxt(file,dtype=int, delimiter=',',skiprows=1,usecols=(5,))
gpud1 =  np.loadtxt(file,dtype=int, delimiter=',',skiprows=1,usecols=(6,))
gpu2 =  np.loadtxt(file,dtype=int, delimiter=',',skiprows=1,usecols=(7,))
cpu2 = np.loadtxt(file,dtype=int, delimiter=',',skiprows=1,usecols=(8,))
gpud2 =  np.loadtxt(file,dtype=int, delimiter=',',skiprows=1,usecols=(9,))


ax.plot(x, cpu0, color=cmap(0),
    linewidth=2, 
    #linestyle=densely_dashdotdotted,
    marker="o", markersize=9, label="CPU")

ax.plot(x, cpu1, color=cmap(1),
    linewidth=2, 
    #linestyle=densely_dashdotdotted,
    marker="o", markersize=9, label="CPU+1")

ax.plot(x, cpu2, color=cmap(2),
    linewidth=2, 
    #linestyle=densely_dashdotdotted,
    marker="o", markersize=9, label="CPU+2")

ax.plot(x, gpud0, linewidth=2, linestyle="-", color=cmap(4),
    marker="x", markersize=9, label="LAKE")

ax.plot(x, gpud1, linewidth=2, linestyle="-", color=cmap(5),
    marker="x", markersize=9,label="LAKE+1")

ax.plot(x, gpud2, linewidth=2, linestyle="-", color=cmap(6),
    marker="x", markersize=9, label="LAKE+2")

plt.xticks(x, rotation=30, ha='right', rotation_mode="anchor")
plt.ticklabel_format(axis='y', style='sci', scilimits=(2,2))

ax.set_xlim(left=0, right=len(x)-1)
ax.set_ylabel('Time (us)')

ax.set_ylim(top = 1000, bottom=0)

ax.set_xlabel('# I/Os having their latency predicted')

ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)

ax.grid(visible=True, which='major', axis='y', color='#0A0A0A', linestyle='--', alpha=0.2)

ax.legend(ncol = 3, 
        loc = 'upper left', 
        frameon=False,
        bbox_to_anchor=(0.1, 1.3), columnspacing=0.8)

fig.tight_layout()
fig.set_size_inches(4, 3)
fig.set_dpi(800)

fig.savefig(f"linnos_xover.pdf", bbox_inches='tight', pad_inches=0.05)
