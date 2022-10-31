#!/bin/bash

TraceTag='trace'

if [ $# -ne 4 ]
  then
    echo "Usage train.sh <trace_1> <trace_2> <trace_3> <inflection_point> <name of trace>"
    # eg : ./train.sh testTraces/hacktest.trace testTraces/hacktest.trace testTraces/hacktest.trace 85
    exit
fi

echo $1, $2, $3, $4, $5

mkdir -p mlData
sudo ../io_replayer/replayer baseline mlData/TrainTraceOutput 3 /dev/nvme0n1-/dev/nvme1n1-/dev/nvme2n1 $1 $2 $3

source ../../../lakevenv/bin/activate

for i in 0 1 2 
do
   python3 traceParser.py direct 3 4 \
   mlData/TrainTraceOutput_baseline.data mlData/temp1 \
   mlData/"mldrive${i}.csv" "$i"
done

for i in 0 1 2 
do
   python3 pred1_+2.py \
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
mkdir -p weights_header+2
python3 mlHeaderGen+2.py Trace nvme0n1 mlData/drive0weights weights_header+2
python3 mlHeaderGen+2.py Trace nvme1n1 mlData/drive1weights weights_header+2
python3 mlHeaderGen+2.py Trace nvme2n1 mlData/drive2weights weights_header+2

mkdir -p ../kernel_hook/weights_header/$5+2
mv weights_header+2/*  ../kernel_hook/weights_header/$5+2/