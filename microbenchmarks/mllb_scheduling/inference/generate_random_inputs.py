import sys
import random

if len(sys.argv) != 2:
    print("Need output fie as argument")
    sys.exit(1)

fname = sys.argv[1]
N_INPUTS = 1024
N_FEATS = 15

print(f"Generating {N_INPUTS} inputs into {fname}")

with open(fname, "w") as f:
    f.write(f"{N_INPUTS}\n")
    for _ in range(N_INPUTS):
        inps = [ str(round(random.random(),4)) for _ in range(N_FEATS)]
        #append label, and pred
        inps.append(str(0))
        inps.append(str(0))
        f.write(",".join(inps)+"\n")

