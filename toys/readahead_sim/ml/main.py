#!/usr/bin/env python3
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

import argparse
from math import floor
import numpy as np

#from dense1 import Dense_v1
#from lstm import LSTM_v1
from ssdlstm import LSTM_SSD
from config import *

def main(args):
    fname = args.trace
    trace = fname.split("/")[-1]
    trace = trace.split(".")[-2]

    l = LSTM_SSD(reads=args.trace, trace=trace)

    if args.train:
        print("Training model..")
        l.train()
        if args.train != '':
            print(f"Saving model at {args.train}")
            l.save_model(args.train)
    elif args.model:
        print(f"Loading model from {args.model}")
        l.load_model(args.model)
    else:
        print("Argument error, need either -train <path> or -model <path")

    l.test_inference(100)


if __name__ == "__main__" and not __package__:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="path to trace file", type=str)
    parser.add_argument('-m', '--model', nargs='?', help='Path to model')
    parser.add_argument('-t', '--train', nargs='?', const='', help='Where to save model')
    args = parser.parse_args()

    main(args)
