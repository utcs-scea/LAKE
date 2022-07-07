import sys, os

t1 = 6.3
t2 = 20.8

first_ts = None
gpus = {}
with open("gpu_tput.txt") as f:
    for line in f:
        ts,v = line.split(",")
        ts = float(ts)
        v = float(v)
        if first_ts is None:
            first_ts = ts
        gpus[round(ts-first_ts, 4)] = round(v, 4)

tmax = max(gpus.values())

first_ts = None
cpus = {}
with open("cpu_tput.txt") as f:
    for line in f:
        ts,v = line.split(",")
        ts = float(ts)
        v = float(v)
        if first_ts is None:
            first_ts = ts
        cpus[round(ts-first_ts, 4)] = round(v, 4) 


final = []
for k,v in gpus.items():
    if k > t1: 
        break
    final.append(f"{k}, {v/tmax}")
    
for k,v in cpus.items():
    if k > t1: 
        final.append(f"{k}, {v/tmax}")
    if k > t2:
        break

for k,v in gpus.items():
    if k > t2: 
        final.append(f"{k}, {v/tmax}")

final = final[1:]
with open("cutted.txt", "w") as f:
    for l in final:
        f.write(l+"\n")
