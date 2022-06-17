import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam, SGD
import sys, os
from config import *
import string, random

LIMIT_SIZE = None

def norm_dist(a, b):
    d = floor((b-a)/BLOCK_SZ)
    # cap distances
    dn = max(d,  -(MAX_DIST/2) )
    dn = min(dn,  (MAX_DIST/2) )
    dn = dn+MAX_DIST/2
    return (d, dn)

def denorm_dist(d):
    return d - MAX_DIST/2

class BaseModel:
    def parse_input(self, fname):
        reads = []
        count = 0
        with open(fname, "r") as f:
            for line in f:
                offset = line.split(",")[-1].rstrip()
                reads.append(int(offset))
                count += 1
                if LIMIT_SIZE is not None and count == LIMIT_SIZE: break 
        print(f"Parsed {len(reads)} inputs")
        return reads

    def load_model(self, fpath):
        self.model = load_model(fpath)

    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))

    def save_model(self, fpath):
        if hasattr(self, "base_model_name") and hasattr(self, "trace_name"):
            final_path = os.path.join(fpath, self.base_model_name+self.trace_name)
        else:
            print("Class does not have base_model_name or trace_name, using random name")
            final_path = os.path.join(fpath, id_generator())

        print(f"Saving model to {final_path}")
        self.model.save(final_path)