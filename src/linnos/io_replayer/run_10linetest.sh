#!/bin/bash


if [[ $# -eq 0 ]] ; then
    echo 'need argument: baseline|failover'
    exit 1
fi

sudo ./replayer $1 /dev/nvme0n1p1-/dev/nvme1n1-/dev/nvme2n1 ./testTraces/100lines.trace ./testTraces/100lines.trace ./testTraces/100lines.trace outputs/test10output
