import math
import numpy as np
import matplotlib.pyplot as plt

data=""",az,cosmos,bingi,mix2x,mix3x"
baseline,138.55,158.35,138.55,281.83,558.81
linnos,147.7,193.7,147.7,255.7,328.95
lake,160.63,254.29,151.82,248.38,331.86
linnos1,320.64,307.66,320.64,573.79,437.49
lake1,307.34,357.08,307.34,444.25,411.6
linnos2,417.76,507.58,417.76,673.48,523.82
lake2,325.81,547.11,325.81,473.71,479.84"""

kvdata = {
    "Baseline": [],
    "NN cpu": [],
    "NN LAKE": [],
    "NN+1 cpu": [],
    "NN+1 LAKE": [],
    "NN+2 cpu": [],
    "NN+2 LAKE": [],
}

xlab= ["Azure*", "Cosmos*", "Bing-I*", "Mixed", "Mixed (3x IOPS)"]

keys = list(kvdata.keys())
lines = data.splitlines()

idx = 0
for line in lines[1:]:
    k = keys[idx] 
    vs = line.split(",")
    kvdata[k] = [float(x) for x in vs[1:]]
    idx += 1

#print(kvdata)

w = 1.2
num_yrs = 5
num_plyrs = 7

first_tick = int(math.ceil((num_plyrs*w/2))) 
gap = num_plyrs*w + 1
x = np.array([first_tick + i*gap for i in range(num_yrs)])

fig, ax = plt.subplots()
#colors = plt.cm.get_cmap('inferno',num_plyrs)
#colors = plt.cm.get_cmap('tab10',num_plyrs)
cc = plt.cm.get_cmap('tab20c')

colors = [cc(17), cc(0), cc(4), cc(1), cc(5), cc(2), cc(6)]



patterns = ["..", "//", None, "//", None, "//", None]

#, "\\" , "|" , "-" , "+" , "x", "o", "O", ".", "*" ]

fig,ax = plt.subplots(1,1, figsize=(10,10))
b = []
i = len(kvdata.keys())-1
for ci, (k,v) in enumerate(kvdata.items()):
    xx = x - (i - num_plyrs/2 + 0.5)*w
    print(xx)
    #xx = xx[::-1]
    #print(xx)
    b.append(ax.bar(xx, 
             v,
             width=w, 
             #color=colors(ci), 
             color=colors[ci], 
             align='center', 
             hatch=patterns[ci],
             #edgecolor = 'black', 
             #linewidth = 1.0, 
             #alpha=0.9))
        ))
    i -= 1


           
#ax.set_ylabel('Goals')
ax.set_ylabel('Average Read Latency (us)')
#ax.set_title('Goals scored by players')
plt.ylim(top=700)

ax.legend([b_ for b_ in b], 
           kvdata.keys(), 
           ncol = 3, 
           loc = 'upper left', 
           frameon=False,
           bbox_to_anchor=(0.01, 1.01))


ax.set_xticks(x)
ax.set_xticklabels(xlab)

# for i in range(num_plyrs):
#     ax.bar_label(b[i], 
#                  padding = 3, 
#                  label_type='center', 
#                  rotation = 'vertical')

ax.grid(visible=True, which='major', axis='y', color='#0A0A0A', linestyle='--', alpha=0.2)                 


fig.tight_layout()
fig.set_size_inches(8, 2)
fig.set_dpi(800)

plt.savefig(f"linnos.pdf", bbox_inches='tight', pad_inches=0.05)