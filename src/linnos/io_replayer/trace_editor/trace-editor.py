#!/usr/bin/env python2
#title           :trace-editor.py
#description     :process traces
#author          :Vincentius Martin
#date            :-
#version         :0.1
#usage           :see readme
#notes           :
#python_version  :2.7.5+
#==============================================================================

# import default
import sys
import argparse
from os import listdir
from subprocess import call

sys.path.insert(0, './scripts/')

import trace_modifier
import preprocess_trace
import traces_combiner
import busy_load
import filter_raid
import iopsimbalance
import toplargeio
import cuttrace
import characteristic
import traces_merger
import trace_sanitizer
# end of import part

# define global variables
requestlist = []

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-file", help="trace file to process",type=str)
  #parser.add_argument("-files", nargs='+', help="trace files to process",type=str)
  parser.add_argument("-dir", help="directory file to process",type=str)
  
  parser.add_argument("-produceTrace", help="produce preprocessed trace", action='store_true')
  parser.add_argument("-preprocessMSTrace", help="preprocess the MS trace into disksim ascii format", action='store_true')
  parser.add_argument("-preprocessBlkReplayTrace", help="preprocess the blkreplay trace into disksim ascii format", action='store_true')
  parser.add_argument("-preprocessUnixBlkTrace", help="preprocess the blkreplay trace into disksim ascii format", action='store_true')
  parser.add_argument("-breaktoraid", help="create a RAID-0 subtrace", action='store_true')
  parser.add_argument("-ioimbalance", help="check RAID IO Imbalance", action='store_true')
  parser.add_argument("-combine", help="combine preprocessed traces inside a directory based on file name", action='store_true')
  parser.add_argument("-merge", help="merge preprocessed traces inside a directory based on time", action='store_true')
  parser.add_argument("-toplargeio", help="get n top large io", action='store_true')
  parser.add_argument("-cuttrace", help="cut a trace", action='store_true')
  parser.add_argument("-getLargestIO", help="get largest IO", action='store_true')
  
  parser.add_argument("-offset", help="offset", choices=['0','32','64','128','256','512','1024'], default = '0')
  parser.add_argument("-filter", help="filter specific type", choices=['all','write','read'], default='all')
  parser.add_argument("-devno", help="disk/device number", type=int, default=0)
  parser.add_argument("-duration", help="how many minutes", type=float, default=1.0)
  parser.add_argument("-mostLoaded", help="most loaded", action='store_true')
  parser.add_argument("-mostRandomWrite", help="most random write", action='store_true')
  parser.add_argument("-busiest", help="busiest", action='store_true')
  parser.add_argument("-largestAverage", help="largest average", action='store_true')
  parser.add_argument("-characteristic", help="characteristic of a trace", action='store_true')
  parser.add_argument("-top", help="top n", type=int, default=1)
  parser.add_argument("-resize", help="resize a trace", type=float, default=1.0)
  parser.add_argument("-rerate", help="rerate a trace", type=float, default=1.0)
  parser.add_argument("-insert", help="insert a 'size' KB 'iotype' request every 'interval' ms", action='store_true')
  parser.add_argument("-ndisk", help="n disk for RAID", type=int, default=2)
  parser.add_argument("-stripe", help="RAID stripe unit size in byte", type=int, default=4096)
  parser.add_argument("-granularity", help="granularity to check RAID IO imbalance in minutes", type=int, default=1)
  parser.add_argument("-timerange", help="time range to cut the trace", type=float, nargs = 2)
  parser.add_argument("-sanitize", help="sanitize (incorporate contiguous + remove repeated reads)", action='store_true')
  parser.add_argument("-maxsize", help="maximum request size", type=int, default=1099511627776)
  parser.add_argument("-size", help="size in KB", type=int, default=4)
  parser.add_argument("-iotype", help="1 for read and 0 for write", type=int, default=0)
  parser.add_argument("-interval", help="time interval in ms", type=int, default=1000)
  args = parser.parse_args()

  # parse to request list
  if (args.preprocessMSTrace): #preprocess
    if (not args.file and args.dir):
      for ftrace in listdir("in/" + args.dir):
        preprocess_trace.preprocessMSTrace(args.dir + "/" + ftrace, args.filter)
    else:
      preprocess_trace.preprocessMSTrace(args.file, args.filter)
  elif (args.preprocessBlkReplayTrace): #preprocess
    if (not args.file and args.dir): 
      for ftrace in listdir("in/" + args.dir):
        preprocess_trace.preprocessBlkReplayTrace(args.dir + "/" + ftrace, args.filter)
    else:
      preprocess_trace.preprocessBlkReplayTrace(args.file, args.filter)
  elif (args.preprocessUnixBlkTrace): #preprocess
    if (not args.file and args.dir):
      for ftrace in listdir("in/" + args.dir):
        preprocess_trace.preprocessUnixBlkTrace(args.dir + "/" + ftrace, args.filter)
    else:
      preprocess_trace.preprocessUnixBlkTrace(args.file, args.filter)
  elif (args.getLargestIO):
    toplargeio.getLargestIO(args.file)
  elif (args.breaktoraid):
    filter_raid.createAllRaidFiles(args.file, args.ndisk, args.stripe)
  elif (args.ioimbalance):
    iopsimbalance.checkIOImbalance(filter_raid.createAllRaidList(args.file,args.ndisk,args.stripe), args.granularity)
  elif (args.combine):
    traces_combiner.combine(args.dir)
  elif (args.merge):
    traces_merger.merge(args.dir)
  elif args.mostLoaded or args.busiest or args.largestAverage or args.mostRandomWrite: #need combine
    if args.busiest:
      busy_load.checkCongestedTime(args.file, "1", args.devno, args.duration, args.top)
    elif args.mostLoaded:
      busy_load.checkCongestedTime(args.file, "2", args.devno, args.duration, args.top)
    elif args.largestAverage:
      busy_load.checkCongestedTime(args.file, "3", args.devno, args.duration, args.top)
    elif args.mostRandomWrite:
      busy_load.checkCongestedTime(args.file, "4", args.devno, args.duration, args.top)
  elif (args.characteristic):
    if (not args.file and args.dir):
      call(["mkdir", "out/" + args.dir])
      for ftrace in listdir("in/" + args.dir):
        characteristic.getTraceInfo(args.dir + "/" + ftrace)
    else:
      characteristic.getTraceInfo(args.file)
  elif (args.toplargeio):
    toplargeio.getTopLargeIO(args.file, args.offset, args.devno, args.duration, args.filter, args.top)
  elif (args.cuttrace):
    cuttrace.cut(args.file, args.timerange[0], args.timerange[1], args.devno)
  elif (args.sanitize):
    if not args.dir:
        trace_sanitizer.sanitize(args.file,args.maxsize)
    else:
        for ftrace in listdir("in/" + args.dir):
            trace_sanitizer.sanitize(ftrace,args.maxsize)
  elif (args.resize or args.rerate): #modify a trace
    with open("in/" + args.file) as f:
      for line in f:
        requestlist.append(line.rstrip().split(" "))
    if args.resize != 1.0 or args.rerate!= 1.0 or args.insert:
      if (args.resize != 1.0):
        requestlist = trace_modifier.resize(requestlist,args.resize)     
      if (args.rerate != 1.0):
        requestlist = trace_modifier.modifyRate(requestlist,args.rerate) 
      if args.insert:
        requestlist = trace_modifier.insertIO(requestlist,args.size,args.interval,args.iotype) 
      trace_modifier.printRequestList(requestlist, args.file)

  
