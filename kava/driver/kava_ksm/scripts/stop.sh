#!/bin/bash

. `dirname $0`/environment

echo 'Stopping ksm'
echo 0 | sudo tee /sys/kernel/mm/ksm/run

echo 'Removing kava_ksm_start to kava'
sudo rmmod $KAVA_KSM_START_ROOT/kava_ksm_start.ko
