import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import csv
import pandas as pd

rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
fig, ax = plt.subplots()

def autolabel(line):
    x_array = line.get_xdata()
    y_array = line.get_ydata()
    last_index = len(x_array)-1
    x_pos = x_array[last_index] + 10
    y_pos = y_array[last_index]
    ax.text(x_pos, y_pos, s=line.get_label(), color=line.get_color(), ha='right',va='top' )

x_time = np.arange(0, 499, .1).tolist()
kernel = []
kernel_x = []
uspace = []
uspace_x = []


ax.plot(kernel_x, kernel, label='I/O Latency Classifier (K)', color='grey', linestyle='solid')
ax.plot(uspace_x, uspace, label='Hashing (U)', color='red', linestyle='dashed')
#ax.plot(kernel_x[window_size - 1:], moving_avgs, label='Contention (moving avg)', color='red', linestyle='solid')

t0 = (0, 0.8)
t1 = (uspace_offset - .5, 0.05)
t2 = (uspace_offset + .4, 1.0)
t3 = (uspace_offset + 13.8, 1.0)
t4 = (t3[0] + .3, .75)
#t5 = (0, 0.8)
# ax.plot(t0[0],t0[1], "xb")
# plt.annotate('T0', t0, textcoords="offset points", xytext=(10,-17), ha='center')
# ax.plot(t1[0],t1[1], "xb")
# plt.annotate('T1', t1, textcoords="offset points", xytext=(10,-5), ha='center')
# ax.plot(t2[0],t2[1], "xb")
# plt.annotate('T2', t2, textcoords="offset points", xytext=(10,3), ha='center')
# ax.plot(t3[0],t3[1], "xb")
# plt.annotate('T3', t3, textcoords="offset points", xytext=(10,3), ha='center')
# ax.plot(t4[0],t4[1], "xb")
# plt.annotate('T4', t4, textcoords="offset points", xytext=(10,-12), ha='center')
#ax.plot(t5[0],t5[1], "xb")
#plt.annotate('T5', t5, textcoords="offset points", xytext=(10,3), ha='center')


diffs_cont = [ kernel_x[i] - kernel_x[i - 1] for i in range(1, len(kernel_x)) ]
diffs_no_cont = [ uspace_x[i] - uspace_x[i - 1] for i in range(1, len(uspace_x)) ]
print('Kernel sample length (s), avg: ' + str(np.mean(diffs_no_cont)) + ' std:' + str(np.std(diffs_no_cont)))
print('uspace sample length (s), avg: ' + str(np.mean(diffs_cont)) + ' std:' + str(np.std(diffs_cont)))

ax.grid(visible=True, which='major', axis='y', color='#0A0A0A', linestyle='--', alpha=0.2)

fig.tight_layout()
plt.gcf().set_size_inches(5, 2.5)
plt.xlim(0, 40)
plt.ylim(0, 1.1)
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized Throughput')

plt.savefig('adjusted-contention.pdf', dpi = 800, bbox_inches='tight', pad_inches=0)
