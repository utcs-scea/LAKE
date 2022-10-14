TraceTag='trace'

if [ $# -eq 0 ] || [ $# -eq 1 ]
  then
    echo "Usage .\\\train.sh --traintrace <name of training trace> --latencythreshold <inflection point>"
    exit
fi

SHORT=t:,l:,
LONG=traintrace:,latencythreshold:,
OPTS=$(getopt --options $SHORT --longoptions $LONG -- "$@") 

eval set -- "$OPTS"

while :
do
  case "$1" in
    -t | --traintrace )
      traintrace="$2"
      shift 2
      ;;
    -l | --latencythreshold )
      latencythreshold="$2"
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

echo $traintrace, $latencythreshold

sudo ./replayer_fail /dev/nvme0n1-/dev/nvme0n1-/dev/nvme2n1 \
 "testTraces/${traintrace}0."$TraceTag \
 "testTraces/${traintrace}1."$TraceTag \
 "testTraces/${traintrace}2."$TraceTag mlData/TrainTraceOutput

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
   mlData/TrainTraceOutput mlData/temp1 \
   mlData/"mldrive${i}.csv" "$i"
done

for i in 0 1 2 
do
   python3 pred1.py \
   mlData/"mldrive${i}.csv" $latencythreshold > mlData/"mldrive${i}results".txt
done

cd mlData
mkdir -p drive0weights
mkdir -p drive1weights
mkdir -p drive2weights
cp mldrive0.csv.* drive0weights
cp mldrive1.csv.* drive1weights
cp mldrive2.csv.* drive2weights

cd ..
python3 mlHeaderGen.py Trace nvme0n1 mlData/drive0weights weights_header
python3 mlHeaderGen.py Trace nvme1n1 mlData/drive1weights weights_header
python3 mlHeaderGen.py Trace nvme2n1 mlData/drive2weights weights_header