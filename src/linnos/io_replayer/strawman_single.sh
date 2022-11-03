#!/bin/bash


if [ $# -eq 0 ]
  then
    echo "Need type, baseline or failover and trace file"
fi

#sudo ./replayer $1 xxx 1 /dev/vdb-/dev/vdc $2
sudo ./replayer $1 xxx 1 /dev/nvme0n1-/dev/nvme1n1 $2

