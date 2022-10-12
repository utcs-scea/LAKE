#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'need argument: baseline|failover'
    exit 1
fi

sudo ./replayer $1 /dev/nvme0n1-/dev/nvme1n1-/dev/nvme2n1 ./testTraces/testdrive0.trace-cut.trace ./testTraces/testdrive1.trace-cut.trace ./testTraces/testdrive2.trace-cut.trace outputs/real_5min