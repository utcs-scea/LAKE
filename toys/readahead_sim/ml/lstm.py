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

class LSTM_v1 (BaseModel):
    def __init__(self, dist_fn, denorm_fn):
        self.dist_fn = dist_fn
        self.denorm_fn = denorm_fn

    def dist_denorm(self, d):
        return self.denorm_fn(np.argmax(d))

    def model_lstm(self, inp_shape):
        layers = 16
        dropout = 0
        model = Sequential()
        model.add(Input(shape=inp_shape) )
        model.add(LSTM(layers, input_shape=(SLICE_LEN, MAX_DIST+1), return_sequences=True, recurrent_dropout=dropout))
        model.add(LSTM(layers))
        model.add(Dense(MAX_DIST+1, activation='softmax'))
        print(f"LSTM output: {model.output_shape}")

        model.summary()
        model.compile(optimizer=SGD(lr=LEARN_RATE), 
                loss='categorical_crossentropy', 
                #loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['categorical_accuracy'])
        self.model = model

    def prepare_inputs(self, fname, train_pct=0.7):
        reads = self.parse_input(fname)
        cat = self.transform_input_categorical(reads)
        self.t, self.o = self.slice_categorical(cat)
        train_xv, verif_xv = self.split_training(self.t, self.o, train_pct)

        print(f"Examples of training")
        for i in range(min(10, len(self.t))):
            print(f"{train_xv[0][i]}")
            d = self.dist_denorm(train_xv[1][i])
            print(f"  -> {d}")

    def train(self):
        self.model_lstm(self.t[0].shape)
        self.model.fit( train_xv[0], train_xv[1], epochs = LSTM_EPOCHS, 
            validation_data=(verif_xv[0],verif_xv[1]) 
        )

    def load_model(self, fpath):
        self.model = load_model(fpath)

    def save_model(self, fpath):
        self.model.save(fpath)

    def inference(self, n):
        random.seed(0)

        for _ in range(min(n, len(self.t))):
            idx = random.randrange(0,len(self.t))
            inp = np.expand_dims(self.t[idx], axis=0)   
            pred = self.model.predict(inp)

            #print(f"pred shape: {pred.shape}")
            #pred = np.array([np.argmax(x) for x in pred])

            print("\n")
            for x in self.t[idx]:
                print(f"{self.dist_denorm(x)},", end="")

            print(f"\npredicted {self.dist_denorm(pred)}")

