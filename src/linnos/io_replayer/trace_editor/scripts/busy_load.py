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

import operator

def checkCongestedTime(tracefile, process_type, devno, minutes, top = 1):
  #process type
  #1 = busiest
  #2 = most loaded
  #3 = largest average
  #4 = most random write
  timerange = int(minutes * 60000000) #ns

  result = {}
  traceIn = open("in/" + tracefile)

  #using -1 distinguish first time and other
  lastBlockNo = -1
  lastBlockCount = 0

  for line in traceIn:
    tok = line.split(" ")
    if int(tok[1]) == devno:
      if process_type == "3":
        if (int(float(tok[0]) * 1000)/timerange) not in result:
          result[int(float(tok[0]) * 1000)/timerange] = [0.0,0]
          
          result[int(float(tok[0]) * 1000)/timerange][0] += (float(tok[3]) / 1024) # size count
          result[int(float(tok[0]) * 1000)/timerange][1] += 1 # IOs count
      else:
        if (int(float(tok[0]) * 1000)/timerange) not in result:
          result[int(float(tok[0]) * 1000)/timerange] = 0.0

        if process_type == "1": #busiest
          result[int(float(tok[0]) * 1000)/timerange] += 1
        elif process_type == "2": # process_type == "2": #most loaded
          result[int(float(tok[0]) * 1000)/timerange] += (float(tok[3]) / 1024)
        elif process_type == "4": #for most random write
          if int(tok[4]) == 0:
            if lastBlockNo != -1:
              if (lastBlockNo + lastBlockCount) != int(tok[2]):
                result[int(float(tok[0]) * 1000)/timerange] += 1
            lastBlockNo = int(tok[2])
            lastBlockCount = int(tok[3])
  
  # average count        
  if process_type == "3":
    for key in result:
      result[key] = result[key][0] / result[key][1]
  # else: do nothing        

  i = 0
  unit = ""
  if process_type == "1" or process_type == "4":
    unit = "requests"
  else:
    unit = "KB"
  for elm in sorted(result.items(), key=operator.itemgetter(1), reverse=True):
    print("time(minutes): " + str(elm[0] * minutes) + "-" + str(elm[0] * minutes + minutes) + ": " + str(elm[1]).rstrip('0').rstrip('.') + " " + unit)
    i += 1
    if i >= top:
      break

#lowerb = max(result.iteritems(), key=operator.itemgetter(1))[0] * 3600000000
#upperb = (max(result.iteritems(), key=operator.itemgetter(1))[0] + args.hours) * 3600000000

#iolist = []

#for elm in inlist:
  #tok = map(str.lstrip, elm.split(" "))
  #if lowerb <= int(tok[0]) < upperb:
    #iolist.append(int(tok[3]))

# CDF part
#cfreq = 0

#for elm in sorted(iolist):
#  cfreq = (1.0/len(iolist)) + cfreq
#  print(str(elm) + " " + str(cfreq))






