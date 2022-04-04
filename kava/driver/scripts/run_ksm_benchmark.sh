#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
KAVA_BEN_ROOT=$SCRIPTPATH/..
KAVA_ROOT=/data/ariel/kava
RESULT_DIR=/data/ariel/results/

if [[ -z $1 ]]; then
    echo 'Please specify batch size'
    exit
fi
batch_size=$1

echo 'Stop GDM'
sudo service gdm3 stop

echo 'Starting KAvA'
cd ${KAVA_ROOT}/scripts
./load_all.sh &

sleep 20

echo 'Switching to KAvA'
cd ${KAVA_BEN_ROOT}/ksm/scripts
./start.sh $batch_size

echo 'Starting samepage generator'
cd ${KAVA_BEN_ROOT}/ksm/samepage_generator
./generator -n 500000 &

echo 'Starting data collection'
cd ${SCRIPTPATH}
python3 fetch_cpu_stat.py -n ksmd worker kavad -gpu -s -d ${RESULT_DIR}/ -p _batch_size_$batch_size &

sleep 180

echo 'Rebooting'
sudo reboot
