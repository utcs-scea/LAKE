#!/bin/bash

./router runtime=15 cubin_path=$(readlink -f firewall.cubin) batch=8192 input_throughput=0 block_size=32 numrules=100 -sequential
