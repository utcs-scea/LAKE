import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib
from cycler import cycler
import sys, csv

label_map = {
    "cpu": "CPU",
    "gpu": "LAKE",
    "gpu_data": "LAKE (sync.)",
}

def get_label(l):
    return l if l not in label_map.keys() else label_map[l]

#prefix is set here
input_files = glob.glob('santacruz/*.csv')
out_prefix = "xover_"

#cmap = matplotlib.cm.get_cmap("Set1")
cmap = matplotlib.cm.get_cmap("tab10")

c0 =  cmap(0)
c1 =  cmap(1)
c2 =  cmap(2)

for file in input_files:
    print(f"processing {file}")
    fig, ax = plt.subplots()

    # let's make sure we get the right label
    with open(file) as f:
        labels = f.readline()

    # e.g. 0: cpu, 1: gpu
    labels = labels.split(",")
    labels = [x.rstrip() for x in labels]

    x =          np.loadtxt(file,dtype=str, delimiter=',',skiprows=1,usecols=(0,))
    first_col =  np.loadtxt(file,dtype=float, delimiter=',',skiprows=1,usecols=(1,))
    second_col = np.loadtxt(file,dtype=float, delimiter=',',skiprows=1,usecols=(2,))
    
    #cpu
    ax.plot(x, second_col, label=get_label(labels[2]), color=c0,
        linewidth=2, 
        #linestyle=densely_dashdotdotted,
        marker="o",
        )
    #gpu
    ll = get_label(labels[1]) if "kleio" not in file else label_map["gpu_data"]
    ax.plot(x, first_col, label=ll, 
        linewidth=2, linestyle="-", color=c1, 
        marker="s")

    #gpu + data
    if len(labels) == 4:
        third_col = np.loadtxt(file,dtype=float, delimiter=',',skiprows=1,usecols=(3,))
        ax.plot(x, third_col, label=get_label(labels[3]), linewidth=2, linestyle="-", color=c2,
            marker="x")



    plt.xticks(x, rotation=30, ha='right', rotation_mode="anchor")
    plt.ticklabel_format(axis='y', style='sci', scilimits=(2,2))

    ax.set_xlim(left=0, right=len(x)-1)
    ax.set_xlabel('# of inputs')
    ax.set_ylabel('Time (us)')

    ax.set_xlabel('# I/Os having their latency predicted')


    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.legend()

    fig.tight_layout()
    fig.set_size_inches(4, 2)
    fig.set_dpi(800)


    fig.savefig(f"linnos_ae.pdf", bbox_inches='tight', pad_inches=0.05)