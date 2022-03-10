#! /usr/bin/env python3

import sys
from training_config import *
from keras_conf import *
from prep import preprocess
from keras_lb import keras_train
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tags', nargs='+', help='tags list', required=True)
parser.add_argument('-o', '--object', help='tag for model object')
parser.add_argument('-b', '--balance', action='store_true', help='balance multiple dumps')
parser.add_argument('-d', '--dump', action='store_true')

args = parser.parse_args()

tags = args.tags
model_tag = args.object

preprocess(tags, balance=args.balance, out_tag=model_tag)
keras_train(model_tag=model_tag, dump=args.dump)
