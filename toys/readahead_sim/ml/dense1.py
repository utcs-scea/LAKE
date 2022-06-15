#!/usr/bin/env python3
#!/usr/bin/env python3
import os, sys
import argparse
from math import floor
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam, SGD

from base import *

class Dense_v1 (BaseModel):
    def __init__(self, dist_fn, denorm_fn):
        self.dist_fn = dist_fn
        self.denorm_fn = denorm_fn

    def model_dense(self, inp_shape):
        print(f"Input shape: {inp_shape}")
        model = Sequential([
            Input(shape=inp_shape),
            Dense(16, input_shape=inp_shape, activation='relu'),
            Dense(8, activation='relu'),
            #Dense(32, activation='relu'),
            #Dense(MAX_DIST+1, activation='softmax')
            #Dense(MAX_DIST+1, activation='sigmoid')
            Dense(1, activation='softmax')
        ])

        print(f"Dense output: {model.output_shape}")

        learning_rate = LEARN_RATE
        model.compile(optimizer=SGD(lr=learning_rate), 
                loss='categorical_crossentropy', 
                metrics=['categorical_accuracy'])

        self.model = model

    def train(self, fname, train_pct=0.7):
        reads = self.parse_input(fname)
        cat = self.transform_input_regular(reads)
        self.t, self.o = self.slice_regular(cat)
        train_xv, verif_xv = self.split_training(self.t, self.o, train_pct)

        print("training ", verif_xv[0])
        print("label", verif_xv[1])

        self.model_dense((SLICE_LEN,))
        self.model.fit( train_xv[0], train_xv[1], epochs = EPOCHS, 
            validation_data=(verif_xv[0],verif_xv[1]) 
        )

    def inference(self, n):
        #for i in range(len(self.t)):
        for i in range( min(n, len(self.t))):
            inp = np.expand_dims(self.t[i], axis=0)   
            pred = self.model.predict(inp)

            #print(f"pred shape: {pred.shape}")
            #pred = np.array([np.argmax(x) for x in pred])
            print(f"pred for {self.t[i]}: {pred}")

