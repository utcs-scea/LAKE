#!/usr/bin/env python3
import os, sys

if len(sys.argv) != 2:
    print("Need path to input trace")
    sys.exit(1)


reads = []
with open(sys.argv[1], "r") as f:
    for line in f:
        offset = line.split(",")[-1].rstrip()
        reads.append(int(offset))
print(f"Parsed {len(reads)} inputs")

MIN_DIST = 6
sum_dist = 0
max_dist = 0
i = 0
count = 0
last_printed = None
while True:
    if i == len(reads)-1: break
    
    if abs(reads[i] - reads[i+1]) > MIN_DIST:
        if last_printed is None or last_printed != i:
            print(reads[i])
        print(reads[i+1])
        last_printed = i+1
        count += 1
        
        if abs(reads[i] - reads[i+1]) > max_dist:
            max_dist = abs(reads[i] - reads[i+1])

        sum_dist += abs(reads[i] - reads[i+1])

    i += 1


print(f"There are {count} pairs of far IO reads")
print(f"Max dist: {max_dist}")
print(f"Avg distance (distant ios): {sum_dist/count}")
print(f"Avg distance (all ios): {sum_dist/len(reads)}")



