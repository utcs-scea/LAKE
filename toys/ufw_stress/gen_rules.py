#!/usr/bin/env python3
import sys

base = "sudo ufw deny from 192.168.{a}.{b}"
base_del = "sudo ufw delete deny from 192.168.{a}.{b}"

if len(sys.argv) != 2:
    print("need one arg: add or del")

print("#!/bin/bash")

for i in range(10,254):
    for j in range(1,254):
        if sys.argv[1] == "add":
            print(base.format(a=i, b=j))
        elif sys.argv[1] == "del":
            print(base_del.format(a=i, b=j))