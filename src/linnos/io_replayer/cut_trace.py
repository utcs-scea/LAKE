#!/usr/bin/env python3
#title           :busy_load.py
#description     :Get the busiest or the most loaded disk from a trace
#author          :Vincentius Martin
#date            :20150203
#version         :0.1
#usage           :
#notes           :
#python_version  :2.7.5+  
#precondition    :ordered
#==============================================================================


CUT_FROM_MS = 0
CUT_TO_MS =  300000 #5min

def cut(tracefile, devno = -1):
    out = open(tracefile + "-cut.trace", 'w')
    
    lowerb = CUT_FROM_MS
    upperb = CUT_TO_MS

    with open(tracefile) as f:
        for line in f:
            tok = list(map(str.lstrip, line.split(" ")))
            
            if devno != -1 and int(tok[1]) != devno:
                continue

            if lowerb <= float(tok[0]) < upperb:
                out.write(line)
            else:
                break
    
    out.close()


import sys

if len(sys.argv) == 1:
    print("Need at least one path to trace file to cut")
    sys.exit(1)

for t in sys.argv[1:]:
    print(f"Cutting: {t}")
    cut(t)