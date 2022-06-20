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
        self.dist_to_onehot = {}
        self.class_to_dist = {}
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
            self.dist_to_onehot[d] = to_categorical(i, num_classes=SSD_N_CLASSES+1, dtype=int)
            #print(f"dist {d} (count {count}) is categorical: {self.dist_class_map[d]}")
            i += 1
        self.default_dist = to_categorical(SSD_N_CLASSES, num_classes=SSD_N_CLASSES+1, dtype=int)

    def get_onehot(self, d):
        return self.dist_to_onehot.get(d, self.default_dist)

    def prediction_to_dist(self, pred):
        x = np.argmax(pred)
        return self.class_to_dist.get(x, None)

    def prepare_inputs(self):
        hist = Counter()
        dists = []
        for i in range(0, len(self.reads)-1, 3):
            this_r = self.reads[i]
            next_r = self.reads[i+1]            
            dist = next_r-this_r
            hist[dist] += 1
            dists.append(dist)

        self.set_distance_map(hist)
        self.dataX = []
        self.dataY = []
        n = len(dists) - SSD_WINDOW_SZ
        for i in range(n):
            X  = np.empty((SSD_WINDOW_SZ, SSD_N_CLASSES+1), dtype=int)
            for j in range(SSD_WINDOW_SZ):
                X[j] = self.get_onehot(dists[i+j])
            Y = self.get_onehot(dists[i+SSD_WINDOW_SZ])
            self.dataX.append(X)
            self.dataY.append(Y)

        self.split_training()

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
        #cleanup
        del self.dataX
        del self.dataY
        #print(f"Split shapes {self.trainX.shape}, {self.trainY.shape}  -  {self.valX.shape}, {self.valY.shape}")

    def create_model(self):
        layers = 200
        dropout = 0
        
        model = Sequential()
        model.add(Input(shape=(SSD_WINDOW_SZ, SSD_N_CLASSES+1)))
        model.add(LSTM(layers, input_shape=(SSD_WINDOW_SZ, SSD_N_CLASSES+1), return_sequences=True, recurrent_dropout=dropout))
        model.add(LSTM(layers))
        model.add(Dense(SSD_N_CLASSES+1, activation='softmax'))
        print(f"LSTM output: {model.output_shape}")

        model.summary()
        model.compile(optimizer='rmsprop',
              loss="categorical_crossentropy",
              metrics="categorical_accuracy")
        self.model = model

    def train(self):
        self.create_model()
        self.model.fit( self.trainX, self.trainY, epochs = SSD_EPOCHS, 
            validation_data=(self.valX, self.valY) 
        )

    def test_inference(self, n):
        random.seed(0)
        verX, verY = [], []
        for _ in range(min(n, len(self.reads))):
            readidx = random.randrange(SSD_WINDOW_SZ,len(self.reads)-SSD_WINDOW_SZ)
            window = self.reads[readidx:readidx+SSD_WINDOW_SZ+1]          
            pred = self.inference_window(window)

            #print(f" Inference on {window}")
            truth = self.reads[readidx+SSD_WINDOW_SZ+1] - self.reads[readidx+SSD_WINDOW_SZ] 
            #print(f"  {pred}  vs. {truth} (truth)")
            print(f"missed by: {truth-pred}  {pred} vs {truth}")

    def inference_window(self, reads, use_cache=False):
        # get tuple of distances so we can check our cache
        #st = timer()
        if use_cache:
            raw_dists = []
            for i in range(len(reads)-1):
                dist = reads[i+1] - reads[i]
                raw_dists.append(dist)
            raw_dists = tuple(raw_dists)
            cached = self.get_cache(raw_dists)
        else:
            cached = None

        if cached is not None:
            #print("hit")
            return cached
        else:
            x = np.empty((SSD_WINDOW_SZ, SSD_N_CLASSES+1), dtype=int)
            for i in range(len(reads)-1):
                dist = reads[i+1] - reads[i]
                x[i] = self.get_onehot(dist)

            x = np.expand_dims(x, axis=0)
            pred = self.model.predict(x)[0]
            predicted_dist = self.prediction_to_dist(pred)
            #elaps = timer() - st
            #print(f"{self.inf_read_window[0]} inf took {elaps*1000}ms")
            if use_cache:
                #print("miss")
                self.put_cache(raw_dists, predicted_dist)
            return predicted_dist

    def get_cache(self, k):
        return self.inference_cache.get(k)

    def put_cache(self, k, dist):
        self.inference_cache.put(k, dist)

    def readahead(self, idx, is_majfault):
        cur_page = self.reads[idx]
        if is_majfault:
            ras = []
            # if we dont have enough data yet
            if idx < SSD_WINDOW_SZ+1:
                return [x for x in range(cur_page, cur_page+self.readahead_size)]
            # we do have enough, inference
            else:
                window = self.reads[idx:idx+SSD_WINDOW_SZ+1] 
                predicted_dist = self.inference_window(window, use_cache=True)
                ra_offset = int(predicted_dist / self.readahead_size)
                # handle the small negative case
                if ra_offset == 0 and predicted_dist < 0:
                    ra_offset = -1              
                start = cur_page + ra_offset * self.readahead_size
                #read the page that majfaulted
                ras.append(cur_page)
                #fetch readahead_size-1 after predicted
                for page in range(start, start+self.readahead_size-1):
                    if page == len(self.reads): break
                    ras.append(page)
            return ras
        return []
