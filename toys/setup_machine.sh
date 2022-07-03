#!/usr/bin/env bash

echo "Disabling hyperthreading"
for cpunum in $(cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list | cut -s -d, -f2- | tr ',' '\n' | sort -un)
do
        echo 0 > /sys/devices/system/cpu/cpu$cpunum/online
done

# smt
echo "forceoff" | sudo tee /sys/devices/system/cpu/smt/control
(grep -q "^forceoff$\|^notsupported$" /sys/devices/system/cpu/smt/control && echo "Hyperthreading: DISABLED (OK)") || echo "Hyperthreading: ENABLED (Recommendation: DISABLED)"


echo "Disabling THP"
echo never | sudo tee -a /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee -a /sys/kernel/mm/transparent_hugepage/defrag


echo "Setting cpu to performance"
numcpu=$(nproc)
numcpu=$(expr $numcpu - 1)

for cpu in $( seq 0 $numcpu )
do
  #cpufreq-set -c $cpu -g performance
  cpufreq-set -c $cpu -g powersave
done

echo "Disabling cstate"
for cpu in $( seq 0 $numcpu )
do
  echo 1 > /sys/devices/system/cpu/cpu${cpu}/cpuidle/state0/disable
  echo 1 > /sys/devices/system/cpu/cpu${cpu}/cpuidle/state2/disable
  echo 1 > /sys/devices/system/cpu/cpu${cpu}/cpuidle/state3/disable
done