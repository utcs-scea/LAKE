import re, os, sys
from subprocess import run, DEVNULL
from time import sleep
import os.path
import subprocess
import signal
import subprocess




proc = subprocess.Popen("exec ./tools/cpu_gpu", shell=True)  #, stdout=subprocess.PIPE)
sleep(10)
proc.kill()
with open("tmp.out") as f:
    print(f.readlines())


#for line in proc.stdout:
#    #the real code does filtering here
#    print ("test:", line.rstrip())



#   proc = subprocess.Popen("./tools/cpu_gpu > tmp1.out", stdout=subprocess.PIPE, 
#                        shell=True, preexec_fn=os.setsid) 
#     sleep(6) #give it time to start

#     #TODO: run the app that reads 2GB file
#     process = subprocess.Popen("./tools/ReadWriteData", shell=False)

#     sleep(3) # give it some time to settle
#     os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Send the signal to all the process groups





# proc = subprocess.Popen(["./tools/cpu_gpu"], stdout=log, preexec_fn=os.setsid) 

# sleep(6) #give it time to start

# os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Send the signal to all the process groups

