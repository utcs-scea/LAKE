import re, os, sys
from subprocess import run, DEVNULL
from time import sleep
import os.path
import subprocess
import signal
import subprocess
from tokenize import Double
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from timeit import default_timer as timer

uspace_x = np.load("uspace_x.npy")
uspace_tput = np.load("uspace_tput.npy")
print(uspace_x)

cnt = 0
for i in range(40):
    if uspace_x[i] < 0:
        cnt += 1
    else:
        break

uspace_x = uspace_x[cnt+1:]
uspace_tput = uspace_tput[cnt+1:]

min = np.min(uspace_x)
if min < 0:
    uspace_x = uspace_x + abs(min)

print(uspace_x)
uspace_x += 10
uspace_x += 1.75 #cuinit, need to find something better


kernel_x = np.load("kernel_x.npy")
kernel_tput = np.load("kernel_tput.npy")

kernel_x = kernel_x[10:]
kernel_tput = kernel_tput[10:]

min = np.min(kernel_x)
if min < 0:
    kernel_x = kernel_x + abs(min)

kernel_x = kernel_x/pow(10, 9)
#kernel_x += 10

print(kernel_x)

#uspace
size = uspace_x.shape[0]

uspace_cur = uspace_tput[1:]

time_cur = uspace_x[1:]
time_prev = uspace_x[0:size - 1]
time_diff = time_cur - time_prev

uspace_x_val = time_cur
uspace_y_val = uspace_cur/time_diff
uspace_y_val = uspace_y_val/np.max(uspace_y_val)

kernel_size = 3
kernel = np.ones(kernel_size) / kernel_size
uspace_y_val = np.convolve(uspace_y_val, kernel, mode='same')


#kernel
size = kernel_x.shape[0]

kernel_cur = kernel_tput[1:]

time_cur_kern = kernel_x[1:]
time_prev_kern = kernel_x[0:size - 1]
time_diff_kern = time_cur_kern - time_prev_kern

kernel_x_val = time_cur_kern
kernel_y_val = kernel_cur/time_diff_kern
kernel_y_val = kernel_y_val/np.max(kernel_y_val)

kernel_size = 3
kernel = np.ones(kernel_size) / kernel_size
kernel_y_val = np.convolve(kernel_y_val, kernel, mode='same')


rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
fig, ax = plt.subplots()

ax.plot(uspace_x_val, uspace_y_val, label='uspace', color='red', linestyle='solid')
ax.plot(kernel_x_val, kernel_y_val, label='kernel', color='grey', linestyle='solid')

fig.tight_layout()
plt.gcf().set_size_inches(5, 2.5)
#plt.xlim(left=0, right=kernel_x_val[-1])
plt.xlim(left=0, right=30)
# plt.ylim(0, 1.1)
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized Throughput')

plt.savefig('contention.pdf', dpi = 800, bbox_inches='tight', pad_inches=0)







