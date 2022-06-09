import sys, csv, math, gc
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# switch between "" and "0" to use cpu or gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from multiprocessing import Process
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sim.perf_model import *
from kleio.page_selector import *
from kleio.lstm import *
from sim.profile import *
from timeit import default_timer as timer

kleio_lstm = None

st_times = {}
for i in range(1, 86, 5):
    st_times[i] = []

def kleio_load_model(path):
  print("py model at ", path)
  global kleio_lstm
  kleio_lstm = LSTM_model(path)
  return 0

def kleio_inference(inputs, n, batch_size):
  do_timer = True
  if (batch_size-1)%5 != 0:
    do_timer = False
    batch_size = batch_size-1
    gc.collect()

  start = timer()
  #inputs = [60, 500, 560, 60, 320, 620, 440, 180, 60, 620, 560, 240, 60, 360, 620, 380, 180, 120, 620, 620, 100, 60, 420, 620, 340, 140] 
  kinput = LSTM_input(inputs)
  history_length = 6 # periods
  kinput.timeseries_to_history_seq(history_length)
  kinput.split_data(1)
  num_classes = max(set(inputs)) + 1
  print(f"num_classes {num_classes}")
  kinput.to_categor(num_classes)

  kleio_lstm.infer(kinput, batch_size)

  end = timer()
  if do_timer:
    if batch_size not in st_times.keys():
      print(f"{batch_size} not in {st_times.keys()}")
    else:
      st_times[batch_size].append((end-start)*1000)
  return 0

def print_stats():
  global st_times
  from statistics import mean
  for k, v in st_times.items():
    m = mean(v) if len(v) > 0 else 0
    print(f"{k}, {m}")

  st_times = {}
  for i in range(1, 86, 5):
    st_times[i] = []

if __name__ == "__main__":
  kleio_load_model("/home/hfingler/hf-HACK/kava/worker/lstm_tf/lstm_tf_wrapper/coeus-sim-master/lstm_page_539")
  t = [60, 500, 560, 60, 320, 620, 440, 180, 60, 620, 560, 240, 60, 360, 620, 380, 180, 120, 620, 620, 100, 60, 420, 620, 340, 140] 
  for i in range(1, 86, 5):
    kleio_inference(t, 26, i)
  print_stats()

