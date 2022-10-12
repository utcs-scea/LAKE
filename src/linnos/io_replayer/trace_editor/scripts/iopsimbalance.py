#!/usr/bin/env python
#title           :tracebucketing.py
#description     :Divide a trace into some size-buckets
#author          :Vincentius Martin
#date            :20150210
#version         :0.1
#usage           :python tracebucketing.py 
#notes           :
#python_version  :2.7.5+
#==============================================================================

import numpy

def median(lst):
    return numpy.median(numpy.array(lst))

def checkIOImbalance(inputdisk, granularity):
  #for i in range(len(inputdisk)):
  #  tracefile = open("in/" + inputdisk[i])
  #  inputdisk[i] = [line.strip().split(" ") for line in tracefile.readlines()]
  
  delta = granularity * 60000 #minutes to ms
  bucket = {}

  # now fill the bucket
  for i in range(0, len(inputdisk)):
    for request in inputdisk[i]:
      if (int(float(request[0]) * 1000) / (delta * 1000)) not in bucket:
        bucket[int(float(request[0]) * 1000) / (delta * 1000)] = [0] * len(inputdisk)
      bucket[int(float(request[0]) * 1000) / (delta * 1000)][i] += 1

  for key in sorted(bucket):
    print(str(int(key * granularity)) + "-" + str(int(key * granularity + granularity)) + ": " + str(bucket[key])  + " - imbalance:" + str(float(max(bucket[key]) / median(bucket[key]))))

