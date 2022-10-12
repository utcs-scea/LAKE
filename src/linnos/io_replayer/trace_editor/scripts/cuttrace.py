#!/usr/bin/env python
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

def cut(tracefile, lowerb, upperb, devno = -1):
  out = open("out/" + tracefile + "-cut.trace", 'w')
  
  lowerb = 60000 * lowerb
  upperb = 60000 * upperb

  with open("in/" + tracefile) as f:
    for line in f:
      tok = map(str.lstrip, line.split(" "))
      
      if devno != -1 and int(tok[1]) != devno:
        continue

      if lowerb <= float(tok[0]) < upperb:
        out.write(line)
        
  out.close()
