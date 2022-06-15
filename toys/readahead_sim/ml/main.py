#!/usr/bin/env python3
#!/usr/bin/env python3
import os, sys
import argparse
from math import floor
import numpy as np

from dense1 import Dense_v1
from lstm import LSTM_v1
from ssdlstm import LSTM_SSD
from config import *


def norm_dist(a, b):
    d = floor((b-a)/BLOCK_SZ)
    # cap distances
    dn = max(d,  -(MAX_DIST/2) )
    dn = min(dn,  (MAX_DIST/2) )
    dn = dn+MAX_DIST/2
    return (d, dn)

def denorm_dist(d):
    return d - MAX_DIST/2

def main(args):
    l = LSTM_SSD()
    l.prepare_inputs(args.trace)

    #l = LSTM_v1(norm_dist, denorm_dist)
    #l.prepare_inputs(args.trace)
    #l.train()
    #l.save_model("lstm_v1_model_10epoch_16")
    #l.load_model("lstm_v1_model_10epoch")
    #l.inference(100)

    #d = Dense_v1(norm_dist)
    #d.train(args.trace)
    #d.inference(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="path to trace file", type=str)
    parser.add_argument("--train", help="train the model")
    args = parser.parse_args()

    main(args)
