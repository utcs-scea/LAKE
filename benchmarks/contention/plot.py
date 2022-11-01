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

uspace_ts = np.load("uspace_ts.npy")
uspace_x = np.load("uspace_x.npy")
uspace_tput = np.load("uspace_tput.npy")
kernel_x = np.load("kernel_x.npy")
kernel_tput = np.load("kernel_tput.npy")

us_start = uspace_ts[0]
us_cstart = uspace_ts[1]
us_end = uspace_ts[2]

#ns to us
t_zero = kernel_x[0]

#offset annotations
us_start -= t_zero
us_cstart -= t_zero
us_end -= t_zero
us_start = us_start/1000000000
us_cstart = us_cstart/1000000000
us_end = us_end/1000000000

print(f"timestamps {us_start} {us_cstart} {us_end}")

uspace_real_tput = []
for i in range(1, uspace_tput.shape[0]):
    tp = uspace_tput[i] / (uspace_x[i] - uspace_x[i-1])
    uspace_real_tput.append(tp)

#normalize
uspace_tput = np.array(uspace_real_tput)
uspace_tput = uspace_tput/np.max(uspace_tput)

uspace_tput = uspace_tput - min(uspace_tput) 
uspace_tput = uspace_tput / (max(uspace_tput) - min(uspace_tput)) 

#shift uspace x time relative to t0
uspace_x = uspace_x-t_zero
print(f"diff from us x to kx: (sec)", (uspace_x[0]-t_zero)/1000000)
#convert from us to s
uspace_x = uspace_x/1000000000

print("start of uspace x ", uspace_x[0])

kernel_real_tput = []
for i in range(1, kernel_tput.shape[0]):
    tp = kernel_tput[i] / (kernel_x[i] - kernel_x[i-1])
    kernel_real_tput.append(tp)

#normalize
kernel_tput = np.array(kernel_real_tput)
#kernel_tput = kernel_tput/np.max(kernel_tput)
kernel_tput = kernel_tput - min(kernel_tput) 
kernel_tput = kernel_tput / (max(kernel_tput) - min(kernel_tput) ) 

#shift to start at 0
kernel_x = kernel_x - kernel_x[0]
#conver to s
kernel_x = kernel_x / 1000000000
 
#cut so we dont see the ugly start
uspace_x = uspace_x[3:]
uspace_tput = uspace_tput[3:]

kernel_x = kernel_x[3:]
kernel_tput = kernel_tput[3:]

#smooth
kernel_size = 3
kernel = np.ones(kernel_size) / kernel_size
uspace_tput = np.convolve(uspace_tput, kernel, mode='same')

rc('text', usetex=True)
plt.rc('font', family='serif', size=12)
fig, ax = plt.subplots()

ax.plot(uspace_x[1:], uspace_tput, label='Hashing (u)', color='red', linestyle='solid')
ax.plot(kernel_x[1:], kernel_tput, label='I/O Latency Predictor (k)', color='grey', linestyle='solid')

#annotate
for i in range(kernel_x.shape[0]):
    if kernel_x[i] > us_start:
        us_start_x = kernel_x[i-1]
        print(f"{us_start}  closes it {us_start_x}")
        break

#us_start_y = uspace_tput[i]
us_start_y = 0.3
print(f"anotating {us_start_x} {us_start_y}")
ax.plot(us_start_x,us_start_y, "xb")
plt.annotate('T0', (us_start_x, us_start_y), textcoords="offset points", 
    xytext=(-15,-5), ha='center')

#annotate
for i in range(kernel_x.shape[0]):
    if kernel_x[i] > us_cstart:
        us_cstart_x = kernel_x[i-1]
        print(f"{us_cstart}  closes it {us_cstart_x}")
        break
us_cstart_y = uspace_tput[i]
print(f"anotating {us_cstart_x} {us_cstart_y}")
ax.plot(us_cstart_x,us_cstart_y, "xb")
plt.annotate('T1', (us_cstart_x, us_cstart_y), textcoords="offset points", 
    xytext=(-15,-5), ha='center')

#annotate
for i in range(kernel_x.shape[0]):
    if kernel_x[i] > us_end:
        us_end_x = kernel_x[i-1]
        print(f"{us_end}  closes it {us_end_x}")
        break
us_end_y = uspace_tput[i]
print(f"anotating {us_end_x} {us_end_y}")
ax.plot(us_end_x,us_end_y, "xb")
plt.annotate('T2', (us_end_x, us_end_y), textcoords="offset points", 
    xytext=(-15,-15), ha='center')


# us_start_x
# us_cstart_x 
# us_end_x 

# uspace_x
# uspace_tput


fig.tight_layout()
plt.gcf().set_size_inches(5, 2.5)
#plt.xlim(left=0, right=kernel_x[-1])
plt.xlim(left=0, right=30)
# plt.ylim(0, 1.1)
ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
ax.set_xlabel('Time (s)')
ax.set_ylabel('Normalized Throughput')

plt.savefig('contention.pdf', dpi = 800, bbox_inches='tight', pad_inches=0)





# # cnt = 0
# # for i in range(40):
# #     if uspace_x[i] < 0:
# #         cnt += 1
# #     else:
# #         break

# uspace_x = uspace_x[cnt+1:]
# uspace_tput = uspace_tput[cnt+1:]

# min = np.min(uspace_x)
# if min < 0:
#     uspace_x = uspace_x + abs(min)

# print(uspace_x)
# uspace_x += 10
# uspace_x += 1.75 #cuinit, need to find something better


# kernel_x = kernel_x[10:]
# kernel_tput = kernel_tput[10:]

# min = np.min(kernel_x)
# if min < 0:
#     kernel_x = kernel_x + abs(min)

# kernel_x = kernel_x/pow(10, 9)
# #kernel_x += 10

# print(kernel_x)

# #uspace
# size = uspace_x.shape[0]

# uspace_cur = uspace_tput[1:]

# time_cur = uspace_x[1:]
# time_prev = uspace_x[0:size - 1]
# time_diff = time_cur - time_prev

# uspace_x_val = time_cur
# uspace_y_val = uspace_cur/time_diff
# uspace_y_val = uspace_y_val/np.max(uspace_y_val)

# kernel_size = 3
# kernel = np.ones(kernel_size) / kernel_size
# uspace_y_val = np.convolve(uspace_y_val, kernel, mode='same')


# #kernel
# size = kernel_x.shape[0]

# kernel_cur = kernel_tput[1:]

# time_cur_kern = kernel_x[1:]
# time_prev_kern = kernel_x[0:size - 1]
# time_diff_kern = time_cur_kern - time_prev_kern

# kernel_x_val = time_cur_kern
# kernel_y_val = kernel_cur/time_diff_kern
# kernel_y_val = kernel_y_val/np.max(kernel_y_val)

# kernel_size = 3
# kernel = np.ones(kernel_size) / kernel_size
# kernel_y_val = np.convolve(kernel_y_val, kernel, mode='same')


# rc('text', usetex=True)
# plt.rc('font', family='serif', size=12)
# fig, ax = plt.subplots()

# ax.plot(uspace_x_val, uspace_y_val, label='uspace', color='red', linestyle='solid')
# ax.plot(kernel_x_val, kernel_y_val, label='kernel', color='grey', linestyle='solid')

# fig.tight_layout()
# plt.gcf().set_size_inches(5, 2.5)
# #plt.xlim(left=0, right=kernel_x_val[-1])
# plt.xlim(left=0, right=30)
# # plt.ylim(0, 1.1)
# ax.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Normalized Throughput')

# plt.savefig('contention.pdf', dpi = 800, bbox_inches='tight', pad_inches=0)







