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

# the granularity distances are measured
BLOCK_SZ = 32
MAX_DIST = 1024
#MAX_DIST = 8
#the max dist measurable will be BLOCK_SZ*MAX_DIST pages
#32 and 1024 have  range of ~134 MB (half each way)

#how many distances are input to the model
SLICE_LEN = 8


def norm_dist(a, b):
    d = floor((b-a)/BLOCK_SZ)
    # cap distances
    dn = max(d,  -(MAX_DIST/2) )
    dn = min(dn,  (MAX_DIST/2) )
    dn = dn+MAX_DIST/2
    return (d, dn)

dist_fn = norm_dist

def parse_transform_input(fname):
    reads = []
    with open(sys.argv[1], "r") as f:
        for line in f:
            offset = line.split(",")[-1].rstrip()
            reads.append(int(offset))
    print(f"Parsed {len(reads)} inputs")

    dists = np.empty(len(reads)-1, dtype=int)
    rawdists = np.empty(len(reads)-1, dtype=int)
    for i in range(len(reads)-1):
        this_r = reads[i]
        next_r = reads[i+1]
        d, dn = dist_fn(this_r, next_r)
        rawdists[i] = d
        dists[i] = dn

    cat = to_categorical(dists, num_classes=MAX_DIST+1, dtype=int)
    #for i in range(len(dists)):
    #    print(f"{rawdists[i]} -> {dists[i]} : {cat[i]}")
    return cat

def slice_categorical(cat):
    n = len(cat) - SLICE_LEN
    train = []
    out = []
    for i in range(n):
        train_row = np.empty( (SLICE_LEN, cat[0].shape[0]), dtype=int)
        out_row   = np.empty(             cat[0].shape[0],  dtype=int)
        for j in range(SLICE_LEN):
            train_row[j] = cat[i+j]
        out_row = cat[i+SLICE_LEN]

        train.append(train_row)
        out.append(out_row)

    print(f"after slice, shapes: {train[0].shape} {out[0].shape}")
    return (train, out)

def split_training(train, out, train_pct):
    #do soemthing simple and reproducible
    #take train_n out of each 10 for training
    train_n = (train_pct*10)-1

    train_x = []
    train_v = []
    verif_x = []
    verif_v = []

    for i in range(len(train)):
        j = i % 10

        if j < train_n:
            train_x.append(train[j])
            train_v.append(out[j])
        else:
            verif_x.append(train[j])
            verif_v.append(out[j])

    train_x = np.array(train_x)
    train_v = np.array(train_v)
    verif_x = np.array(verif_x)
    verif_v = np.array(verif_v)

    print(f"split shapes {train_x.shape}, {train_v.shape}  -  {verif_x.shape}, {verif_v.shape}")

    return (( train_x, train_v),(verif_x, verif_v))


def train(train, verif):
    layers = 128
    learning_rate = 0.00001
    dropout = 0
    model = Sequential()

    print(f"Input shape: {train[0][0].shape}")
    model.add(Input(shape=train[0][0].shape) )
    model.add(LSTM(layers, input_shape=(SLICE_LEN, MAX_DIST+1), return_sequences=True, recurrent_dropout=dropout))
    model.add(LSTM(layers))
    model.add(Dense(MAX_DIST+1, activation='softmax'))

    model.summary()

    print(f"train shapes: {train[0].shape} {train[1].shape}")

    model.compile(optimizer=SGD(lr=learning_rate), 
            loss='categorical_crossentropy', 
            #loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['categorical_accuracy'])

    model.fit( train[0], train[1], epochs = 100, validation_data=(verif[0],verif[1]) )


def main(args):
    cat = parse_transform_input(args.trace)
    t,o = slice_categorical(cat)
    train_xv, verif_xv = split_training(t, o, 0.7)
    train(train_xv, verif_xv)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="path to trace file", type=str)
    parser.add_argument("--train", help="train the model")
    args = parser.parse_args()

    main(args)
