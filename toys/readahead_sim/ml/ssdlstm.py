#!/usr/bin/env python3
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


class LSTM_SSD (BaseModel):
    def __init__(self):
        self.dist_categ_onehot_map = {}
        self.class_to_dist = {}

    def set_distance_map(self, hist):
        #transform top deltas
        top = hist.most_common(SSD_N_CLASSES)
        
        i = 0
        for delta, _ in top:
            self.class_to_dist[i] = delta

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

    def transform_input(self, reads):
        hist = Counter()
        dists = []
        for i in range(len(reads)-1):
            this_r = reads[i]
            next_r = reads[i+1]            
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

    def prepare_inputs(self, fname, train_pct=0.7):
        reads = self.parse_input(fname)
        self.transform_input(reads)

        print(f"Examples for training")
        for i in range(min(10, len(self.rawX))):
            print(f"{self.rawX[i]}")
            print(f"  -> {self.dataY[i]}")

        self.split_training(train_pct)

    def split_training(self, train_pct):
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
        self.create_model()
        self.model.fit( self.trainX, self.trainY, epochs = SSD_EPOCHS, 
            validation_data=(self.valX, self.valY) 
        )

    def inference(self, n):
        random.seed(0)

        for _ in range(min(n, len(self.valX))):
            idx = random.randrange(0,len(self.valX))
            inp = np.expand_dims(self.valX[idx], axis=0)   
            pred = self.model.predict(inp)
        
            print("\n")
            for x in self.valX[idx]:
                print(f"{self.onehot_to_dist(x)},", end="")

            print(f"\npredicted {self.onehot_to_dist(pred)}")

