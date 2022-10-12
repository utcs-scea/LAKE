#!/usr/bin/env python2
#title           :trace-editor.py
#description     :process a trace disk
#author          :Vincentius Martin
#date            :-
#version         :0.1
#usage           :python trace-editor.py
#notes           :
#python_version  :2.7.5+
#==============================================================================

from random import randint

# input: request list (list), modify the size x times (float)
def resize(reqlist, times):
  for request in reqlist:
    request[3] = ('%f' % (times * float(request[3]))).rstrip('0').rstrip('.')
  return reqlist

# input: request list (list), modify the size x rate times (float)
def modifyRate(reqlist, rate):
  i = 0
  while i < len(reqlist):
    #if float(reqlist[i][0]) * rate > 300000:
    #  del reqlist[i:len(reqlist)]
    #  break
    reqlist[i][0] = '%.3f' % (rate * float(reqlist[i][0]))
    i += 1
  return reqlist

#interval: in ms; size in KB
def insertIO(reqlist,size,interval,iotype):
    insert_time = interval
    maxoffset = int(max(reqlist, key=lambda x: int(x[2]))[2])
    i = 0
    while i < len(reqlist):
        if float(reqlist[i][0]) > insert_time: #7190528,7370752
            reqlist.insert(i,['%.3f' % insert_time,str(0),str(randint(0,maxoffset)),str(size * 2),str(iotype)])
            insert_time += interval
            i += 1
        i += 1
    return reqlist
  
def printRequestList(requestlist, filename):
  out = open("out/" + filename + "-modified.trace" , 'w')
  for elm in requestlist:
    out.write(str(elm[0]) + " " + str(elm[1]) + " " + str(elm[2]) + " " + str(elm[3]) + " " + str(elm[4])+"\n")
  out.close()

