#!/usr/bin/env python
#title           :tracescombiner.py
#description     :Parse hourly I/O size for MS traces format
#author          :Vincentius Martin
#date            :20150203
#version         :0.1
#usage           :python hourlyiosize.py file (--filter read/write)
#notes           :
#python_version  :2.6.6  
#==============================================================================

import argparse
from os import listdir

def combine(tracesdir):

  out = open("out/" + tracesdir + "-combine.trace",'w')

  # get all files
  listoffiles = []
  for ftrace in listdir("in/" + tracesdir):
    listoffiles.append(str(ftrace))

  listoffiles.sort()

  # combine

  timeoffset = 0

  for tracefile in listoffiles:
    print (tracefile)
    with open("in/" + tracesdir + "/" + tracefile) as f:
      timetmp = 0
      for line in (f):
        tok = map(str.lstrip, line.split(" "))
      
        t = {
          "time": float(tok[0]) + timeoffset,
          "devno": int(tok[1]),
          "blkno": int(tok[2]),
          "bcount": int(tok[3]),
          "flags": int(tok[4]),
        };
      
        timetmp = float(tok[0])
        out.write("%s %d %d %d %d\n" % ("{0:.3f}".format(t['time']), t['devno'], t['blkno'], t['bcount'], t['flags']))

    timeoffset += timetmp + 0.001
    
  out.close()


      
