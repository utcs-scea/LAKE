#!/bin/bash
set -e 
set -o pipefail

if [ $# -eq 0 ] || [ $# -eq 1 ] || [ $# -eq 2 ]
  then
    echo "Usage .\\\train.sh --traintrace <name of training trace> --percentile <inflection point> --device <device name>"
    exit
fi

SHORT=t:,l:,d:,
LONG=traintrace:,percentile:,device:,
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
    -d | --device )
      device="$2"
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

echo $traintrace, $percentile, $device

#TODO: add a suffix so we can train concurrently
#TODO: check that ../io_replayer/replayer exists, if not quit
TNAME=$(basename $traintrace)
echo $TNAME
sudo ../io_replayer/replayer baseline mlData/$TNAME /dev/$device $traintrace
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
#args here are: direct, 3, 4, <input (from replayer)> <temp file> <output file> <device number to use>
#device number is only necessar if we run a trace on multiple devices concurrently, otherwise its just 0
python3 traceParser.py direct 3 4 ${REPLAY_OUT} mlData/temp1 \
    mlData/"mldrive${i}.csv" 0

echo "Latency Threshold ${percentile}"
python3 pred1.py mlData/"mldrive${i}.csv" $percentile > mlData/"mldrive${i}results".txt
#pred1 outputs  mlData/mldrive${i}.csv.weights_<something>.csv
# and           mlData/mldrive${i}.csv.bias_<something>.csv

WEIGHTS_DIR="weights_${TNAME}/weights"
HEADER_DIR="weights_${TNAME}/header"
mkdir -p $WEIGHTS_DIR
mkdir -p $HEADER_DIR
mv mlData/mldrive${i}.csv.* $WEIGHTS_DIR

#TODO: we gotta set the device name somehow
python3 mlHeaderGen.py $TNAME $device $WEIGHTS_DIR $HEADER_DIR
