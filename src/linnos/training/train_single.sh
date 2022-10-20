#!/bin/bash
set -e 
set -o pipefail

#  ./train_single.sh --traintrace ../trace_tools/15s_1m_100us.trace --percentile 85
#
#

if [ $# -eq 0 ] || [ $# -eq 1 ] || [ $# -eq 2 ]
  then
    echo "Usage .\\\train.sh --traintrace <name of training trace> --percentile <inflection point> --ndevice <1 for now>"
    exit
fi

SHORT=t:,l:,d:,
LONG=traintrace:,percentile:,ndevice:,
OPTS=$(getopt --options $SHORT --longoptions $LONG -- "$@") 

eval set -- "$OPTS"

while :
do
  case "$1" in
    -t | --traintrace )
      traintrace="$2"
      shift 2
      ;;
    -l | --percentile )
      percentile="$2"
      shift 2
      ;;
    -d | --ndevice )
      ndevice="$2"
      shift 2
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      ;;
  esac
done

echo $traintrace, $percentile, $ndevice

#TODO: add a suffix so we can train concurrently
#TODO: check that ../io_replayer/replayer exists, if not quit
TNAME=$(basename $traintrace)
echo $TNAME

#echo "../io_replayer/replayer baseline mlData/$TNAME 1 /dev/nvme0n1 $traintrace"
#sudo ../io_replayer/replayer baseline mlData/$TNAME 1 /dev/nvme0n1 $traintrace

#exit 0

#this currently outputs mlData/{TNAME}baseline
REPLAY_OUT="mlData/${TNAME}baseline"

python3 -m venv linnOSvenv
source linnOSvenv/bin/activate
pip3 install numpy
pip3 install --upgrade pip
pip3 install tensorflow
pip3 install keras
pip3 install pandas
pip3 install scikit-learn

i=0
#device number is only necessar if we run a trace on multiple devices concurrently, otherwise its just 0
#args here are: direct, 3, 4, <input (from replayer)> <temp file> <output file> <device number to use>
#python3 traceParser.py direct 3 4 ${REPLAY_OUT} mlData/temp1 mlData/"mldrive${i}.csv" $i

echo "python3 traceParser.py direct 3 4 ${REPLAY_OUT} mlData/temp1 mlData/"mldrive${i}.csv" $i"

exit 0

echo "Latency Threshold ${percentile} percentile"
python3 pred1.py mlData/"mldrive${i}.csv" $percentile #> mlData/"mldrive${i}results".txt
#pred1 outputs  mlData/mldrive${i}.csv.weights_<something>.csv
# and           mlData/mldrive${i}.csv.bias_<something>.csv

WEIGHTS_DIR="weights_${TNAME}/weights"
HEADER_DIR="weights_${TNAME}/header"
mkdir -p $WEIGHTS_DIR
mkdir -p $HEADER_DIR
mv mlData/mldrive${i}.csv.* $WEIGHTS_DIR

#TODO: we gotta set the device name somehow
#args:               workload, device_name, input_folder, output_folder
python3 mlHeaderGen.py $TNAME vdb $WEIGHTS_DIR $HEADER_DIR
