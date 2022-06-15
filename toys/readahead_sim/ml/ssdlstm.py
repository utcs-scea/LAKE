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
        self.dist_class_map = {}

    def set_distance_map(self, hist):
        #transform top deltas
        top = hist.most_common(SSD_N_CLASSES)
        i = 0
        for d, count in top:
            self.dist_class_map[d] = to_categorical(i, num_classes=SSD_N_CLASSES+1, dtype=int)
            #print(f"dist {d} (count {count}) is categorical: {self.dist_class_map[d]}")
            i += 1
        self.default_dist = to_categorical(SSD_N_CLASSES, num_classes=SSD_N_CLASSES+1, dtype=int)

    def get_class(self, d):
        return self.dist_class_map.get(d, self.default_dist)

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
                X[j] = self.get_class(dists[i+j])
            Y = self.get_class(dists[i+SSD_WINDOW_SZ])
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


    # def model_lstm(self, inp_shape):
    #     layers = 500
    #     dropout = 0
    #     model = Sequential()
    #     model.add(Input(shape=inp_shape) )
    #     model.add(LSTM(layers, input_shape=(SLICE_LEN, MAX_DIST+1), return_sequences=True, recurrent_dropout=dropout))
    #     model.add(LSTM(layers))
    #     model.add(Dense(MAX_DIST+1, activation='softmax'))
    #     print(f"LSTM output: {model.output_shape}")

    #     model.summary()
    #     model.compile(optimizer=SGD(lr=LEARN_RATE), 
    #             loss='categorical_crossentropy', 
    #             #loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #             metrics=['categorical_accuracy'])
    #     self.model = model


    # def train(self):
    #     self.model_lstm(self.t[0].shape)
    #     self.model.fit( train_xv[0], train_xv[1], epochs = LSTM_EPOCHS, 
    #         validation_data=(verif_xv[0],verif_xv[1]) 
    #     )

    # def load_model(self, fpath):
    #     self.model = load_model(fpath)

    # def save_model(self, fpath):
    #     self.model.save(fpath)

    # def inference(self, n):
    #     random.seed(0)

    #     for _ in range(min(n, len(self.t))):
    #         idx = random.randrange(0,len(self.t))
    #         inp = np.expand_dims(self.t[idx], axis=0)   
    #         pred = self.model.predict(inp)

    #         #print(f"pred shape: {pred.shape}")
    #         #pred = np.array([np.argmax(x) for x in pred])

    #         print("\n")
    #         for x in self.t[idx]:
    #             print(f"{self.dist_denorm(x)},", end="")

    #         print(f"\npredicted {self.dist_denorm(pred)}")

