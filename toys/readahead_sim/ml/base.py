import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam, SGD
import sys
from config import *

class BaseModel:
        
    def parse_input(self, fname):
        reads = []
        with open(sys.argv[1], "r") as f:
            for line in f:
                offset = line.split(",")[-1].rstrip()
                reads.append(int(offset))
        print(f"Parsed {len(reads)} inputs")
        return reads

    def transform_input_categorical(self, reads):
        dists = np.empty(len(reads)-1, dtype=int)
        rawdists = np.empty(len(reads)-1, dtype=int)
        for i in range(len(reads)-1):
            this_r = reads[i]
            next_r = reads[i+1]
            d, dn = self.dist_fn(this_r, next_r)
            rawdists[i] = d
            dists[i] = dn

        cat = to_categorical(dists, num_classes=MAX_DIST+1, dtype=int)
        #for i in range(len(dists)):
        #    print(f"{rawdists[i]} -> {dists[i]} : {cat[i]}")
        return cat

    def transform_input_regular(self, reads):
        dists = np.empty(len(reads)-1, dtype=int)
        rawdists = np.empty(len(reads)-1, dtype=int)
        for i in range(len(reads)-1):
            this_r = reads[i]
            next_r = reads[i+1]
            d, dn = self.dist_fn(this_r, next_r)
            rawdists[i] = d
            dists[i] = dn

        cat = to_categorical(dists, num_classes=MAX_DIST+1, dtype=int)
        #for i in range(len(dists)):
        #    print(f"{rawdists[i]} -> {dists[i]} : {cat[i]}")
        return dists

    def slice_regular(self, dists):
        n = len(dists) - SLICE_LEN
        trains = []
        labels = []
        for i in range(n):
            train_row = np.empty( SLICE_LEN, dtype=int)
            out_row   = np.empty( 1,  dtype=int)
            for j in range(SLICE_LEN):
                train_row[j] = dists[i+j]
            out_row[0] = dists[i+SLICE_LEN]
            trains.append(train_row)
            labels.append(out_row)

        print(f"after slice regular, shapes: {trains[0].shape} {labels[0].shape}")
        return (trains, labels)

    def slice_categorical(self, cat):
        n = len(cat) - SLICE_LEN
        trains = []
        labels = []
        for i in range(n):
            train_row = np.empty( (SLICE_LEN, cat[0].shape[0]), dtype=int)
            out_row   = np.empty(             cat[0].shape[0],  dtype=int)
            for j in range(SLICE_LEN):
                train_row[j] = cat[i+j]
            out_row = cat[i+SLICE_LEN]

            trains.append(train_row)
            labels.append(out_row)

        print(f"after slice, shapes: {trains[0].shape} {labels[0].shape}")
        return (trains, labels)

    def split_training(self, train_data, label_data, train_pct):
        #do soemthing simple and reproducible
        #take train_n out of each 10 for training
        train_n = (train_pct*10)-1
        train_x = []
        train_v = []
        verif_x = []
        verif_v = []

        for i in range(len(train_data)):
            j = i % 10
            if j < train_n:
                train_x.append(train_data[j])
                train_v.append(label_data[j])
            else:
                verif_x.append(train_data[j])
                verif_v.append(label_data[j])

        train_x = np.array(train_x)
        train_v = np.array(train_v)
        verif_x = np.array(verif_x)
        verif_v = np.array(verif_v)
        print(f"split shapes {train_x.shape}, {train_v.shape}  -  {verif_x.shape}, {verif_v.shape}")
        return (( train_x, train_v),(verif_x, verif_v))