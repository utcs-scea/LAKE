import sys, csv, math
from multiprocessing import Process
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sim.perf_model import *
from kleio.page_selector import *
from kleio.lstm import *
from sim.profile import *

kleio_lstm = None


def kleio_load_model(path):
  global kleio_lstm
  kleio_lstm = LSTM_model(path)
  return 0

def kleio_inference(inputs, n):
  kinput = LSTM_input(inputs)
  history_length = 6 # periods
  kinput.timeseries_to_history_seq(history_length)
  kinput.split_data(1)
  num_classes = max(set(inputs)) + 1
  print(f"num_classes {num_classes}")
  kinput.to_categor(num_classes)

  kleio_lstm.infer(kinput)

  return 0

if __name__ == "__main__":
  kleio_load_model("/home/hfingler/hf-HACK/kava/worker/lstm_tf/lstm_tf_wrapper/coeus-sim-master/lstm_page_539")
  t = [60, 500, 560, 60, 320, 620, 440, 180, 60, 620, 560, 240, 60, 360, 620, 380, 180, 120, 620, 620, 100, 60, 420, 620, 340, 140] 
  kleio_inference(t)

