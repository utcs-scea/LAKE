import sys, csv, math, gc
import os
# switch between "" and "0" to use cpu or gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, False)

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

def kleio_load_model(path):
    print("Kleio: loading model from ", path)
    global kleio_lstm
    kleio_lstm = LSTM_model(path)
    print("Kleio: model loaded!")
    return 0

def kleio_force_gc():
    gc.collect()

def kleio_inference(inputs, n):
    #print("input: ", inputs)
    #print("type: ", type(inputs))
    #print("len ", len(inputs))
    start = timer()
    #python bug won't let us use inputs.. even though they are the same
    inputs = [60, 500, 560, 60, 320, 620, 440, 180, 60, 620, 560, 240, 60, 360, 620, 380, 180, 120, 620, 620, 100, 60, 420, 620, 340, 140] 
    #print("input: ", inputs)
    #print("type: ", type(inputs))
    #print("len ", len(inputs))
    inputs = np.array(inputs)
    inputs = np.resize(inputs, (n,) )

    kinput = LSTM_input(inputs)
    history_length = 6 # periods
    kinput.timeseries_to_history_seq(history_length)
    kinput.split_data(1)
    num_classes = max(set(inputs)) + 1
    kinput.to_categor(num_classes)
    kleio_lstm.infer(kinput, n)
    end = timer()
    return (end-start)*1000

if __name__ == "__main__":
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    # for gpu in gpus:
    #     print(f"setting grown on gpu")
    #     tf.config.experimental.set_memory_growth(gpu, False)
    kleio_load_model("../../../kleio/lstm_page_539")

    t = [60, 500, 560, 60, 320, 620, 440, 180, 60, 620, 560, 240, 60, 360, 620, 380, 180, 120, 620, 620, 100, 60, 420, 620, 340, 140] 
    for i in range(1, 130, 8):
        time = kleio_inference(t, i)
        print(f"{i} : {time}ms")

    for i in range(1, 130, 8):
        time = kleio_inference(t, i)
        print(f"{i} : {time}ms")

