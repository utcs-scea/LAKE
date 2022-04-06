#!/bin/bash

. `dirname $0`/environment

if [ $# -eq 0 ]; then
  batch_size=1
else
  batch_size=$1
fi

echo 'Setting batch size'
echo $batch_size | sudo tee /sys/kernel/mm/ksm/batch_size

# Only start kava if batch size is > 0
if [ $batch_size -gt 0 ]; then
  echo 'Switching to kava'
  sudo insmod $KAVA_KSM_START_ROOT/kava_ksm_start.ko
fi

echo 'Starting ksm'
$SCRIPTPATH/run_ksm.sh $batch_size
