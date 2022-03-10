import sys
import random

if len(sys.argv) != 2:
    print("Need output file as argument")
    sys.exit(1)

fname = sys.argv[1]
N_INPUTS = 2048

print(f"Generating {N_INPUTS} inputs into {fname}")

with open(fname, "w") as f:
    f.write("src_non_pref_nr,delta_hot,cpu_idle,cpu_not_idle,cpu_newly_idle,same_node,prefer_src,prefer_dst,src_len,src_load,dst_load,dst_len,delta_faults,extra_fails,buddy_hot,can_migrate\n")
    for _ in range(N_INPUTS):
        
        s1 = [random.randint(0,1) for _ in range(8) ]
        s2 = [random.randint(0, 3000), ]
        s3 = [random.randint(0,1) for _ in range(2) ]
        s4 = [random.randint(0, 100), ]
        s5 = [random.uniform(-1.0, 0), ]
        s6 = [random.randint(0,1) for _ in range(3) ]

        final = s1 + s2 + s3 + s4 + s5 + s6

        final = [str(x) for x in final]

        f.write(",".join(final)+"\n")

