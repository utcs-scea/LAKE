import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv
import matplotlib

cmap = matplotlib.cm.get_cmap("tab10")

#rc('text', usetex=True)
#plt.rc('font', family='serif', size=10)
fig, ax = plt.subplots()

def autolabel(line):
    x_array = line.get_xdata()
    y_array = line.get_ydata()
    last_index = len(x_array)-1
    x_pos = x_array[last_index] + 10
    y_pos = y_array[last_index]
    ax.text(x_pos, y_pos, s=line.get_label(), color=line.get_color(), ha='right',va='top' )

x_time = np.arange(0, 10, 0.1).tolist()
aesni_cpu = []
cpu = []
kava_cpu = []
backend_cpu = []
kava_gpu = []

with open('aesni_util/cpu_stats_fs.csv', newline='', encoding='utf-8') as f:
    data = csv.reader(f)
    idx = 0
    for row in data:
        idx += 1
        if idx <= 33:
            continue
        if row:
            aesni_cpu.append(float(row[1]))

with open('cpu_util/cpu_stats.csv', newline='', encoding='utf-8') as f:
    data = csv.reader(f)
    idx = 0
    for row in data:
        idx += 1
        if idx <= 35:
            continue
        if row:
            cpu.append(float(row[1]))

with open('kava_util/cpu_stats.csv', newline='', encoding='utf-8') as f:
    data = csv.reader(f)
    idx = 0
    for row in data:
        idx += 1
        if idx <= 33:
            continue
        if row:
            kava_cpu.append(float(row[1]))
            backend_cpu.append(float(row[2]) + float(row[3]))

with open('kava_util/gpu_stats_fs.txt', newline='', encoding='utf-8') as f:
    data = csv.reader(f)
    idx = 0
    for row in data:
        idx += 1
        if idx <= 33:
            continue
        if row:
            kava_gpu.append(float(row[1]))
kava_gpu = kava_gpu[1:44]

aesni_cpu.extend([0] * (len(x_time) - len(aesni_cpu)))
cpu.extend([0] * (len(x_time) - len(cpu)))
kava_cpu.extend([0] * (len(x_time) - len(kava_cpu)))
backend_cpu.extend([0] * (len(x_time) - len(backend_cpu)))
kava_gpu.extend([0] * (len(x_time) - len(kava_gpu)))

ax.plot(x_time, cpu, label='CPU\neCryptfs', color=cmap(0) )#, marker="x", markersize=3)
ax.plot(x_time, aesni_cpu, label='AES-NI\neCryptfs', color=cmap(1), marker="o", markersize=3)
ax.plot(x_time, kava_cpu, label='LAKE\neCryptfs', color=cmap(2), marker="v", markersize=3)
ax.plot(x_time, backend_cpu, label='LAKE\nAPI server', color=cmap(3), marker="s", markersize=3)

ax.plot(x_time, kava_gpu, label='LAKE\nGPU util.', color='black',
    #linestyle='dotted', linewidth=3)
    linewidth=1, marker='.')
    
fig.tight_layout()
plt.xlim(0, 8.5)
plt.ylim(0, 120)
#ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.2), columnspacing=1)

ax.legend(loc='upper center', ncol=1, bbox_to_anchor=(1.25, 1), columnspacing=1)

ax.set_xlabel('Time (s)')
ax.set_ylabel('Utilization (\%)')

ax.grid(visible=True, which='major', axis='y', color='#0A0A0A', linestyle='--', alpha=0.2)

fig.set_size_inches(3.5, 2)
fig.set_dpi(200)

plt.savefig('plot.pdf', bbox_inches='tight',pad_inches=0.05)