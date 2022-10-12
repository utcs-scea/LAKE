#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo 'need argument: baseline|failover'
    exit 1
fi

sudo ./replayer $1 /dev/nvme0n1-/dev/nvme1n1-/dev/nvme2n1 ./testTraces/testdrive0-4x.trace ./testTraces/testdrive1-4x.trace ./testTraces/testdrive2-4x.trace outputs/real_5min4x