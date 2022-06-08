#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gc
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import preprocess
from timeit import default_timer as timer


sequence_length = 20
epochs = 10
batch_size = 32
feature_dimension = 341

st_times = {}
for i in range(20, 380, 40):
    st_times[i] = []

def convertToOneHot(vector, num_classes=341):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.
    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v
        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    # assert isinstance(vector, np.ndarray)
    if not isinstance(vector, np.ndarray):
        print("convertToOneHot failed because vector is no np.ndarray")
        return np.empty( shape=(0) )
    # assert len(vector) > 0
    if len(vector) <= 0:
        print("convertToOneHot failed because vector length is less than 1")
        return np.empty( shape=(0) )

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        # assert num_classes > 0
        if (num_classes <= 0):
            print("convertToOneHot failed because num_class is less than 1")
            return np.empty( shape=(0) )
        # assert num_classes >= np.max(vector)
        if (num_classes < np.max(vector)):
            print("convertToOneHot failed because num_class is less than the max syscall id")
            return np.empty( shape=(0) )

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1  # The last element 0, is reserverd as normal target???
    return result.astype(int)


def summarize_results(prediction):
    rounded_prediction = prediction.round()
    sum = np.sum(rounded_prediction)
    if (sum > len(prediction) / 2):
        return 1
    else:
        return 0

def sequence_n_gram_parsing(alist,n_gram=20,num_class=341,sliding_window=1):
    if len(alist) < n_gram:
        print("number of syscalls are less then 20, can't perform inference")
        return np.empty( shape=(0) )

    ans = []
    for i in range(0,len(alist)-n_gram+1,sliding_window):
        tmp = alist[i:i+n_gram]
        tmp = np.asarray(tmp)
        oneHot = convertToOneHot(tmp, num_class)
        if len(oneHot) == 0:
            print("convertToOneHot failed, existing...")
            return np.empty( shape=(0) )
        ans.append(oneHot)

    #transform into nmup arrray
    ans = np.array(ans) # ans is an array of 20 one-hot encoding for a given syscall file
    return (ans)


def load_model(filepath="/home"):
    global model
    try:
        print("Loading Keras module...")
        model = keras.models.load_model(filepath)
        model.summary()
    except ImportError:
        print("load_model: Loading from an hdf5 file and h5py is not available")
        return -1
    except IOError:
        print("load_model: Invalid file path")
        return -1

    # print("Model successfully loaded.")
    return 0
 
def print_stats():
    from statistics import mean
    for k, v in st_times.items():
        m = mean(v) if len(v) > 0 else 0
        print(f"{k}, {m}")

def standard_inference(syscalls, num_syscall, sliding_window=1):
    #print(f"num_syscall {num_syscall}")
    #print(f"syscalls {syscalls}")

    start = timer()

    if (len(syscalls) != num_syscall):
        print("standard_inference failed because number of syscalls sent is different to num_syscall\n")
        return -1
    
    n_gram_data = sequence_n_gram_parsing(syscalls, sliding_window=sliding_window)
    if (len(n_gram_data) <= 0):
        print("sequence_n_gram_parsing failed n standard_inference...")
        return -1

    print(f"shape: {n_gram_data.shape}")

    try:
        prediction = model.predict(n_gram_data)
        # print(prediction.shape)
    except RuntimeError:
        print("model prediction runtime error...")
        return -1
    except ValueError:
        print("model prediction value Error...")
        return -1

    result = summarize_results(prediction)

    end = timer()
    # we store time in  ms
    st_times[num_syscall].append((end-start)*1000)

    return result

def close_ctx():
    gc.collect()
