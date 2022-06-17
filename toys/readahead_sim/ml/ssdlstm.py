#!/usr/bin/env python3
import os, sys
import argparse
from math import floor
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam, SGD
from base import *
from collections import Counter
from timeit import default_timer as timer
from collections import OrderedDict
 
INFERENCE_CACHE_SIZE = 10000
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
 
    def get(self, key):
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]
 
    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)

class LSTM_SSD (BaseModel):
    # Accepted args:
    #  reads - Required. List of reads or path to filename of trace
    #  simulator - optional. If set, we instantly load model
    #  models_path - required if simulator is set. Path to dir with all models
    #  readahead_size - required if simulator is set. how much to read ahead
    def __init__(self, **kwargs):
        if "reads" not in kwargs.keys():
            print("Need to pass reads to LSTM_SSD constructor")
            sys.exit(1)

        self.base_model_name = "lstmsdd_"
        self.dist_categ_onehot_map = {}
        self.class_to_dist = {}
        self.inf_read_window = []
        self.inference_cache = LRUCache(INFERENCE_CACHE_SIZE)

        if "trace" in kwargs.keys():
            self.trace_name = kwargs["trace"]

        # accept both list of reads or filename
        if isinstance(kwargs["reads"], str):
            self.reads = self.parse_input(kwargs["reads"])
        else:
            self.reads = kwargs["reads"]
        self.prepare_inputs()

        # if we see simulator as args, get ready
        if "simulator" in kwargs.keys():
            # models_path must be set, use default model name
            mpath = os.path.join(kwargs["models_path"], self.base_model_name+self.trace_name)
            self.load_model(mpath)

            self.readahead_size = kwargs["readahead_size"]

    def set_distance_map(self, hist):
        #transform top deltas
        top = hist.most_common(SSD_N_CLASSES)
        #print(f"Delta hist: {top}")

        i = 0
        for delta, _ in top:
            self.class_to_dist[i] = delta
            #print(f"class {i} is delta {delta}")
            i += 1

        i = 0
        for d, count in top:
            self.dist_categ_onehot_map[d] = to_categorical(i, num_classes=SSD_N_CLASSES+1, dtype=int)
            #print(f"dist {d} (count {count}) is categorical: {self.dist_class_map[d]}")
            i += 1
        self.default_dist = to_categorical(SSD_N_CLASSES, num_classes=SSD_N_CLASSES+1, dtype=int)

    def get_onehot(self, d):
        return self.dist_categ_onehot_map.get(d, self.default_dist)

    def onehot_to_dist(self, onehot):
        x = np.argmax(onehot)
        return self.class_to_dist.get(x, None)

    def prepare_inputs(self):
        hist = Counter()
        dists = []
        for i in range(len(self.reads)-1):
            this_r = self.reads[i]
            next_r = self.reads[i+1]            
            dist = next_r-this_r
            hist[dist] += 1
            dists.append(dist)

        self.set_distance_map(hist)

        self.rawX = []
        self.rawY = []
        self.dataX = []
        self.dataY = []
        n = len(dists) - SSD_WINDOW_SZ
        for i in range(n):
            rX = np.empty(SSD_WINDOW_SZ, dtype=int)
            X    = np.empty((SSD_WINDOW_SZ, SSD_N_CLASSES+1), dtype=int)
            for j in range(SSD_WINDOW_SZ):
                rX[j] = dists[i+j]
                X[j] = self.get_onehot(dists[i+j])
            Y = self.get_onehot(dists[i+SSD_WINDOW_SZ])
            self.rawX.append(rX)
            self.dataX.append(X)
            self.dataY.append(Y)

    def split_training(self, train_pct=0.7):
        #do soemthing simple and reproducible
        #take train_n out of each 10 for training
        train_n = (train_pct*10)-1

        self.valX = []
        self.valY = []
        self.trainX = []
        self.trainY = []

        for i in range(len(self.dataX)):
            j = i % 10
            if j < train_n:
                self.trainX.append(self.dataX[j])
                self.trainY.append(self.dataY[j])
            else:
                self.valX.append(self.dataX[j])
                self.valY.append(self.dataY[j])

        self.trainX = np.array(self.trainX)
        self.trainY = np.array(self.trainY)
        self.valX = np.array(self.valX)
        self.valY = np.array(self.valY)

        print(f"Split shapes {self.trainX.shape}, {self.trainY.shape}  -  {self.valX.shape}, {self.valY.shape}")

    def create_model(self):
        layers = 200
        dropout = 0
        
        model = Sequential()
        model.add(Input(shape=(SSD_WINDOW_SZ, SSD_N_CLASSES+1)))
        model.add(LSTM(layers, input_shape=(SSD_WINDOW_SZ, SSD_N_CLASSES+1), return_sequences=True))  #,  recurrent_dropout=dropout))
        model.add(LSTM(layers))
        model.add(Dense(SSD_N_CLASSES+1, activation='softmax'))
        print(f"LSTM output: {model.output_shape}")

        model.summary()
        model.compile(optimizer='rmsprop',
              loss="categorical_crossentropy",
              metrics="categorical_accuracy")
        self.model = model

    def train(self):
        self.split_training(train_pct)
        self.create_model()
        self.model.fit( self.trainX, self.trainY, epochs = SSD_EPOCHS, 
            validation_data=(self.valX, self.valY) 
        )

    def test_inference(self, n):
        random.seed(0)

        for _ in range(min(n, len(self.valX))):
            idx = random.randrange(0,len(self.valX))
            inp = np.expand_dims(self.valX[idx], axis=0)   
            pred = self.model.predict(inp)
        
            print("\n")
            for x in self.valX[idx]:
                print(f"{self.onehot_to_dist(x)},", end="")

            print(f"\npredicted {self.onehot_to_dist(pred)}")

    def inference_window(self):
        # get tuple of distances so we can check our cache
        st = timer()
        raw_dists = []
        for i in range(len(self.inf_read_window)-1):
            this_r = self.inf_read_window[i]
            next_r = self.inf_read_window[i+1]            
            dist = next_r-this_r
            raw_dists.append(dist)
        raw_dists = tuple(raw_dists)

        cached = self.get_cache(raw_dists)
        if cached is not None:
            print("hit")
            #elaps = timer() - st
            #print(f"cache took {elaps*1000}ms")
            return cached
        else:
            print("miss")
            x = np.empty((SSD_WINDOW_SZ, SSD_N_CLASSES+1), dtype=int)
            for i in range(len(self.inf_read_window)-1):
                this_r = self.inf_read_window[i]
                next_r = self.inf_read_window[i+1]            
                dist = next_r-this_r
                x[i] = self.get_onehot(dist)

            x = np.expand_dims(x, axis=0)
            pred = self.model.predict(x)
            predicted_dist = self.onehot_to_dist(pred)
            elaps = timer() - st
            print(f"{self.inf_read_window[0]} inf took {elaps*1000}ms")
            self.put_cache(raw_dists, predicted_dist)
            return predicted_dist

    def get_cache(self, k):
        ret = self.inference_cache.get(k)
        return None if ret == -1 else ret

    def put_cache(self, k, dist):
        self.inference_cache.put(k, dist)

    def readahead(self, idx, is_majfault):
        cur_page = self.reads[idx]
        self.inf_read_window.append(cur_page)
        #maintain SSD_WINDOW_SZ+1 reads (which is SSD_WINDOW_SZ distances)
        if len(self.inf_read_window) > SSD_WINDOW_SZ+1:
            self.inf_read_window = self.inf_read_window[1:]

        if is_majfault:
            ras = []
            # if we dont have enough data yet
            if len(self.inf_read_window) < SSD_WINDOW_SZ+1:
                return [x for x in range(cur_page+1, cur_page+self.readahead_size+1)]
                pass
            # we do have enough, inference
            else:
                predicted = self.inference_window()
                #fetch readahead_size after predicted
                for page in range(predicted,predicted+self.readahead_size):
                    if page == len(self.reads): break
                    ras.append(page)
            return ras
        return []
