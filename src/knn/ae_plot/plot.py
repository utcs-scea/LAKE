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
gpucomp = {}

for line in reversed(klog.splitlines()):
    #print (line)
    if "knn_GPU_BATCH" in line:
        print("got knn_gpu: ", line)
        #format:  GPU_batch_X,2,4     (2 is comp, 4 is comp+data)
        m = re.search("knn_GPU\_BATCH\_(\d+), (\d+), (\d+)", line) 
        if not m:
            print("error parsing GPU string: ", line)
            sys.exit(1)
        #maybe the workload was ran twice, so we skip if we already found it
        if int(m.group(1)) in gpucomp.keys():
            continue

        gpucomp[int(m.group(1))] = int(m.group(2))
        gpudata[int(m.group(1))] = int(m.group(3))

    if "knn_CPU_batch_" in line:
        #format:  CPU_batch_X,2
        m = re.search("knn_CPU\_batch\_(\d+),(\d+)", line) 
        if not m:
            print("error parsing CPU string: ", line)
            sys.exit(1)

        #maybe the workload was ran twice, so we skip if we already found it
        if int(m.group(1)) in cpu.keys():
            continue
        cpu[int(m.group(1))] = int(m.group(2))




with open('knn.csv') as f:
    lines = f.readlines()

cpu_data = {}
#getting ae_cpu numbers
with open('ae_cpu.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        cpu_data[int(row[1])] = int(row[2])

outlines = []
header = f"{lines[0].rstrip()},ae_gpu,ae_cpu, ae_gpu_data\n"
outlines.append(header)
for line in lines[1:]:
    bsize = int(line.split(',')[0])
    added_line = f"{line.rstrip()},{gpucomp[bsize]},{cpu_data[bsize]}, {gpudata[bsize]}\n"
    outlines.append(added_line)

with open('tmp.csv', 'w') as f:
    for l in outlines:
        f.write(l)


# Plot graphs
batch_sizes = []
cpu = []
gpu = []
gpu_data = []
ae_cpu = []
ae_gpu = []
ae_gpu_data = []
outlines = outlines[1:]
for l in outlines:
    vals = l.split(",")
    batch_sizes.append(int(vals[0]))
    gpu.append(int(vals[1]))
    cpu.append(int(vals[2]))
    gpu_data.append(int(vals[3]))
    ae_gpu.append(int(vals[4]))
    ae_cpu.append(int(vals[5]))
    #ae_cpu.append(int(vals[2]))
    ae_gpu_data.append(int(vals[5]))

x = np.arange(len(batch_sizes)) + 1
cmap = matplotlib.cm.get_cmap("Paired")

c0 =  cmap(0)
c1 =  cmap(1)
c2 =  cmap(2)
c3 =  cmap(3)
c4 =  cmap(4)
c5 =  cmap(5)

plt.yscale('log')
fig, ax = plt.subplots()
ax.set_yscale('log')


#cpu
ax.plot(x, cpu, label="CPU", color=c0,
    linewidth=2, 
    #linestyle=densely_dashdotdotted,
    marker="o",
    )

ax.plot(x, ae_gpu, label="AE_CPU", color=c1,
    linewidth=2, 
    #linestyle=densely_dashdotdotted,
    marker="*",
    )

#gpu
ax.plot(x, gpu, label="LAKE", 
    linewidth=2, linestyle="-", color=c2, 
    marker="s")

ax.plot(x, ae_gpu, label="AE_LAKE", 
    linewidth=2, linestyle="-", color=c3, 
    marker="v")

#gpu + data
ax.plot(x, gpu_data, label="LAKE (sync.)", linewidth=2, linestyle="-", color=c4,
        marker="x")

ax.plot(x, ae_gpu_data, label="AE_LAKE (sync.)", linewidth=2, linestyle="-", color=c5,
        marker="^")

#plt.xticks(np.asarray(batch_sizes), rotation=30, ha='right', rotation_mode="anchor")

#plt.ticklabel_format(axis='y', style='sci', scilimits=(2,2))
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)

#ax.set_xlim(left=0, right=len(batch_sizes)-1)
ax.set_ylabel('Time (us)')

ax.set_xlabel('Size of Prediction set')

ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.legend(fontsize = 7)

fig.tight_layout()
fig.set_size_inches(4, 2)
fig.set_dpi(800)

fig.savefig(f"knn_ae.pdf", bbox_inches='tight', pad_inches=0.05)
