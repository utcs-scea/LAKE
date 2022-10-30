import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv
import matplotlib

cmap = matplotlib.cm.get_cmap("tab10")

#rc('text', usetex=True)
#plt.rc('font', family='serif', size=10)
fig, ax = plt.subplots()

data = []
with open('bandwidth.csv', newline='', encoding='utf-8') as f:
    d = csv.reader(f)
    for row in d:        
        data.append(row)

xlabels = data[0][1:]
x = range(0, len(xlabels))
#for i in x:
#    if i % 2 == 1:
#        xlabels[i] = ""

c = 1.048576 # MiB to MB
cpu_r = [ float(x) * c for x in data[1][1:] ]
cpu_w = [ float(x) * c for x in data[2][1:] ]
aesni_r = [ float(x) * c for x in data[3][1:] ]
aesni_w = [ float(x) * c for x in data[4][1:] ]
kava_r = [ float(x) * c for x in data[5][1:] ]
kava_w = [ float(x) * c for x in data[6][1:] ]
mix_r = [ float(x) * c for x in data[7][1:] ]
mix_w = [ float(x) * c for x in data[8][1:] ]

ax.plot(x, cpu_r, label='CPU Read', color=cmap(0), 
    marker="x", linestyle='solid')
ax.plot(x, cpu_w, label='CPU Write', color=cmap(0),
    marker="x", linestyle='dotted')

ax.plot(x, aesni_r, label='AES-NI Read', color=cmap(1),
    marker="o", linestyle='solid')
ax.plot(x, aesni_w, label='AES-NI Write', color=cmap(1),
    marker="o", linestyle='dotted')

ax.plot(x, kava_r, label='LAKE Read', color=cmap(2),
    marker="s", linestyle='solid')
ax.plot(x, kava_w, label='LAKE Write', color=cmap(2),
    marker="s", linestyle='dotted')


ax.plot(x, mix_r, label='GPU+AES-NI\nRead', color=cmap(3),
    marker="v", linestyle='solid')
ax.plot(x, mix_w, label='GPU+AES-NI\nWrite', color=cmap(3),
    marker="v", linestyle='dotted')

fig.tight_layout()
#plt.gcf().set_size_inches(4, 2.75)
#ax.legend(loc='upper left')
#ax.legend(loc='center left', ncol=1, bbox_to_anchor=(1, 0.5))

ax.legend(loc='center left', ncol=1, bbox_to_anchor=(1, 0.4))

#ax.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.20))
#ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.35))

ax.set_xlabel('Block Size (Bytes)')
ax.set_ylabel('Throughput (MB/s)')
ax.set_xticks(x)
ax.set_xticklabels(xlabels, rotation=45)

ax.grid(visible=True, which='major', axis='y', color='#0A0A0A', linestyle='--', alpha=0.2)

fig.set_size_inches(4, 2)
fig.set_dpi(200)

plt.savefig('tput.pdf', bbox_inches='tight',pad_inches=0.05)