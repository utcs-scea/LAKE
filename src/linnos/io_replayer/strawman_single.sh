#!/bin/bash


if [ $# -eq 0 ]
  then
    echo "Need type, baseline or failover and trace file"
fi


sudo ./replayer $1 xxx 1 /dev/nvme2n1-/dev/nvme1n1 $2

#./strawman.sh strawman ../trace_tools/15s_256k_50us.trace
#./strawman.sh strawman ../trace_tools/15s_1m_100us.trace

#./strawman.sh baseline ../trace_tools/15s_1m_100us.trace
#./strawman.sh baseline ../trace_tools/15s_256k_50us.trace
