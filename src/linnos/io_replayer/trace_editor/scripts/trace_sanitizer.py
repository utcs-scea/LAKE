#!/usr/bin/env python

from bitarray import bitarray

largestblkno = 0

def sanitize(tracefile,maxsize):
  print ("sanitizing: " + str(tracefile))
  out = open("out/" + tracefile.split('/')[-1] + "-sanitize.trace", 'w')

  with open ("in/" + tracefile, "r") as myfile:
    tracein=myfile.read()
    out.write(fitIOToDisk(removeRepeatedReads(fuseContiguousIO(tracein)),maxsize)) #2147483648
  
  out.close()

def fuseContiguousIO(tracein):
  global largestblkno

  #initialize
  out = ""
  lr = ["-1"] * 5 #last request
  
  for line in tracein.splitlines():
    tok = map(str.lstrip, line.split(" ")) #time,devno,blkno,blkcount,flag-0 write/1 read
    
    #for largest blkno
    largestblkno = max(largestblkno,int(tok[2]) + int(tok[3]))
    #-----------------
    
    if (lr[4] == tok[4]) and (int(lr[2]) + int(lr[3]) == int(tok[2])):
      lr[3] = str(int(lr[3]) + int(tok[3]))
    else:
      if lr[0] != "-1":
        out += ("%s %s %s %s %s\n" % (lr[0], lr[1], lr[2], lr[3], lr[4]))
      lr = tok
  out += ("%s %s %s %s %s\n" % (lr[0], lr[1], lr[2], lr[3], lr[4]))
  
  print ("finish - fuse contiguous IOs")
  return out
  
def removeRepeatedReads(tracein):
  #initialize
  out = ""
  has_been_read = bitarray(largestblkno)
  has_been_read.setall(False)
  
  for line in tracein.splitlines():
    tok = map(str.lstrip, line.split(" ")) #time,devno,blkno,blkcount,flag-0 write/1 read
    
    if (tok[4] == "0") or (0 in has_been_read[int(tok[2]):int(tok[2]) + int(tok[3])]):
      out += ("%s %s %s %s %s\n" % (tok[0], tok[1], tok[2], tok[3], tok[4]))
      if (tok[4] == "1"):
        has_been_read[int(tok[2]):int(tok[2]) + int(tok[3])] = int(tok[3]) * bitarray([True])
  
  print ("finish - remove repeated reads") 
  return out
  
def fitIOToDisk(tracein,disk_size): #by default 1TB
    #initialize
    out = ""
    
    for line in tracein.splitlines():
        tok = map(str.lstrip, line.split(" ")) #time,devno,blkno,blkcount,flag-0 write/1 read
        out += ("%s %s %s %s %s\n" % (tok[0], tok[1], str(int(tok[2]) % (disk_size // 512)), tok[3], tok[4]))
    
    print ("finish - fit all IOs to disk") 
    return out
    
  
