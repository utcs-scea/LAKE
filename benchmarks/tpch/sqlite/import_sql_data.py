#!/usr/bin/env python3

import sys, os
import glob


if len(sys.argv) != 2:
    print("Need path to data dir")
    sys.exit(1)

file_path = os.path.realpath(__file__)


tbls_dir = os.path.join( os.path.dirname(file_path), "tbls")
os.makedirs(tbls_dir, exist_ok=True)

for name in glob.glob( os.path.join(sys.argv[1], "*")):
    if not name.endswith(".tbl"):
        continue

    fname = name.split("/")[-1]

    fout = open(os.path.join(tbls_dir, fname), "w")

    with open(name, "r") as fin:
        for line in fin:
            if line[:-1] == "|":
                fout.write(line[:-1]+"\n" )
            elif line[:-2] == "|":
                fout.write(line[:-2]+"\n" )
            else:
                fout.write(line)
