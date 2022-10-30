import re, os, sys
from subprocess import run, DEVNULL
from time import sleep
import os.path
import subprocess
import signal
import subprocess

log = open("log1.txt", "w")


proc = subprocess.Popen(["./tools/cpu_gpu"],stdout=log)


# proc = subprocess.Popen(["./tools/cpu_gpu"], stdout=log, preexec_fn=os.setsid) 

# sleep(6) #give it time to start

# os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Send the signal to all the process groups

