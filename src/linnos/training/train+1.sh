#!/bin/bash

TraceTag='trace'

if [ $# -ne 4 ]
  then
    echo "Usage train.sh <trace_1> <trace_2> <trace_3> <inflection_point>"
    # eg : ./train.sh testTraces/hacktest.trace testTraces/hacktest.trace testTraces/hacktest.trace 85
    exit
fi

echo $1, $2, $3, $4

sudo ../io_replayer/replayer baseline mlData/TrainTraceOutput 3 /dev/nvme0n1-/dev/nvme1n1-/dev/nvme2n1 $1 $2 $3

python3 -m venv linnOSvenv
source linnOSvenv/bin/activate
pip3 install numpy
pip3 install --upgrade pip
pip3 install tensorflow
pip3 install keras
pip3 install pandas
pip3 install scikit-learn

for i in 0 1 2 
do
   python3 traceParser.py direct 3 4 \
   mlData/TrainTraceOutput_baseline.data mlData/temp1 \
   mlData/"mldrive${i}.csv" "$i"
done

for i in 0 1 2 
do
   python3 pred1_+1.py \
   mlData/"mldrive${i}.csv" $4 > mlData/"mldrive${i}results".txt
done

cd mlData
mkdir -p drive0weights
mkdir -p drive1weights
mkdir -p drive2weights
cp mldrive0.csv.* drive0weights
cp mldrive1.csv.* drive1weights
cp mldrive2.csv.* drive2weights

cd ..
mkdir -p "weights_header+1"
python3 mlHeaderGen+1.py Trace nvme0n1 mlData/drive0weights "weights_header+1"
python3 mlHeaderGen+1.py Trace nvme1n1 mlData/drive1weights "weights_header+1"
python3 mlHeaderGen+1.py Trace nvme2n1 mlData/drive2weights "weights_header+1"