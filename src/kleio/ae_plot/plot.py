import re, csv, sys
from subprocess import check_output
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

KLOG_LINES = 200

klog = check_output(f"sudo journalctl -k | tail -n {KLOG_LINES}", shell=True)
klog = klog.decode("ascii")

gpu = {}
cpu = {}

for line in reversed(klog.splitlines()):
    if "kleio_" in line:
        m = re.search("kleio\_(\d+),(\d+),(\d+)", line) 
        if not m:
            print("error parsing GPU string: ", line)
            sys.exit(1)

        #maybe the workload was ran twice, so we skip if we already found it
        if int(m.group(1)) in gpu.keys():
            continue

        cpu[int(m.group(1))] = int(m.group(2))
        gpu[int(m.group(1))] = int(m.group(3))


x = []
cpucol = []
gpucol = []


for i in sorted(cpu.keys()):
    x.append(i)
    cpucol.append(cpu[i])
    gpucol.append(gpu[i])


lakecpu = np.loadtxt('kleio.csv',dtype=int, delimiter=',',skiprows=1,usecols=(1,))
lakegpu=  np.loadtxt('kleio.csv',dtype=int, delimiter=',',skiprows=1,usecols=(2,))

cmap = matplotlib.cm.get_cmap("Paired")
c0 =  cmap(0)
c1 =  cmap(1)
c2 =  cmap(2)
c3 =  cmap(3)
c4 =  cmap(4)
c5 =  cmap(5)

fig, ax = plt.subplots()

#cpu
ax.plot(x, lakecpu, label="CPU", color=c0,
    linewidth=2, 
    marker="o",
    )

ax.plot(x, cpucol, label="AE_CPU", color=c1,
    linewidth=2, 
    #linestyle=densely_dashdotdotted,
    marker="*",
    )

#gpu
ax.plot(x, lakegpu, label="LAKE", 
    linewidth=2, linestyle="-", color=c2, 
    marker="s")

ax.plot(x, gpucol, label="AE_LAKE", 
    linewidth=2, linestyle="-", color=c3, 
    marker="v")


#plt.xticks(np.asarray(batch_sizes), rotation=30, ha='right', rotation_mode="anchor")
plt.ticklabel_format(axis='y', style='sci', scilimits=(2,2))
ax.set_xticks(x, rotation=90)
#ax.set_xticklabels(batch_sizes)

#ax.set_xlim(left=0, right=len(batch_sizes)-1)
ax.set_ylabel('Time (us)')
ax.set_xlabel('Input size')

ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
ax.legend(fontsize = 7)

fig.tight_layout()
fig.set_size_inches(4, 2)
fig.set_dpi(800)

fig.savefig(f"kleio_ae.pdf", bbox_inches='tight', pad_inches=0.05)
