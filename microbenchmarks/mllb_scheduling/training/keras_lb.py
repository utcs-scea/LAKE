import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf

import logging
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers
import os.path
import pickle
import json

from timer import Timer

from training_config import features, label
from keras_conf import *
from predict_ana import predict_ana
from tensorflow.keras.callbacks import EarlyStopping

tf.get_logger().setLevel('DEBUG')
pd.options.mode.chained_assignment = None
from tensorflow.keras.callbacks import Callback

class Histories(Callback):
    DUMP_DELTA=100
    def on_train_begin(self,logs={}):
        self.losses = []
        self.accuracies = []
        self.dump_cd = 10
        self.batches = []
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        self.dump_cd -= 1
        self.batch += 1
        if self.dump_cd == 0:
            self.losses.append(logs.get('loss'))
            self.accuracies.append(logs.get('acc'))
            self.dump_cd = self.DUMP_DELTA
            self.batches.append(self.batch)


early_stop = EarlyStopping()


def print_config():
    msg = []
    DO_LOAD and msg.append('load')
    DO_TRAIN and msg.append('train')
    DO_SAVE and msg.append('save')
    DO_DUMP and msg.append('dump')
    DO_EVALUATE and msg.append('evaluate')
    LOAD_EVALUATE and msg.append(f'(on {EVALUATE_SET})')
    print(*msg)


def random_split(data, proportion=0.1):
    nrow = len(data)
    indices = np.random.permutation(nrow)
    test_size = int(nrow * proportion)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    test_data = data.iloc[test_indices]
    train_data = data.iloc[train_indices]
    return train_data, test_data


def get_model(input_dim):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_dim=input_dim))
    #  model.add(Dense(5, activation='relu'))
    model.add(Dense(1))
    #  sgd = optimizers.SGD()
    #  rmsprop = optimizers.RMSprop(learning_rate=0.001, rho=0.9, decay=0.01)
    adam = optimizers.Adam(learning_rate=0.0005, decay=0.000003)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def keras_train(it=0, model_tag=None, dump=None, plot_loss=False):
    global EVALUATE_TAG, EVALUATE_SET, DO_DUMP, DO_SAVE
    model_tag = model_tag or 'default'
    DATA_FILE = './post_' + model_tag + '.csv'
    WEIGHT_FILE = 'weights_' + model_tag + '.h5'
    MODEL_FILE = 'model_' + model_tag + '.h5'
    DO_DUMP = dump or DO_DUMP
    DO_SAVE = DO_TRAIN and DO_SAVE or False
    DO_READ = X_val or DO_TRAIN

    if DO_READ:
        print(f"reading {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
        head_df = df[:TEST_SIZE]
        tail_df = df[-TEST_SIZE:]
        head_X, head_y = head_df[features], head_df[label]
        tail_X, tail_y = tail_df[features], tail_df[label]
        train_df, test_df = random_split(df, 0.1)
        train_X = train_df[features]
        train_y = train_df[label]
        test_X = test_df[features]
        test_y = test_df[label]
        n_features = train_X.shape[1]
        model = get_model(n_features)

    print(f"X_val? {X_val}")

    if X_val:
        sum_acc = 0
        sum_loss = 0
        for time in range(X_val):
            train_df, test_df = random_split(df, 0.1)
            train_X = train_df[features]
            train_y = train_df[label]
            test_X = test_df[features]
            test_y = test_df[label]
            n_features = train_X.shape[1]
            model = get_model(n_features)
            model.fit(train_X, train_y, batch_size=BATCH_SIZE, validation_split=0.1,
                      epochs=100, callbacks=[early_stop])
            loss, acc = model.evaluate(test_X, test_y)
            sum_acc += acc
            sum_loss += loss
            print(time, loss, acc, sum_acc / (time+1))
        print('avg acc', sum_acc / X_val)
        print('avg loss', sum_loss / X_val)
        exit()

    if LOAD_EVALUATE:
        EVALUATE_SET = f'./post_{EVALUATE_TAG}.csv'
        extra_test_df = pd.read_csv(EVALUATE_SET)


    print('Model', model_tag)
    print_config()

    if DO_LOAD:
        try:
            model.load_weights(WEIGHT_FILE)
        except:
            try:
                model = tf.keras.models.load_model(MODEL_FILE)
            except:
                print('load model failed')
                pass

    if DO_TRAIN:
        if plot_loss:
            histories = Histories()
            history = model.fit(train_X, train_y, batch_size=BATCH_SIZE,
                                validation_split=0.1, epochs=20, callbacks=[early_stop, histories])

            import matplotlib.pyplot as plt
            plt.plot(histories.batches, histories.losses)
            plt.title("Training history of loss vs. batches")
            plt.xlabel('Training batches')
            plt.ylabel('Loss')
            batches_file = 'batches_' + model_tag
            loss_file = 'loss_' + model_tag
            try:
                with open(batches_file, 'wb') as f:
                    pickle.dump(histories.batches, f)
                with open(loss_file, 'wb') as f:
                    pickle.dump(histories.losses, f)
                    #  plt.yscale('log')
            except Exception as e:
                print("histories save failed")
                print(e)
            plt.show()
        else:
            print(f"{len(train_X)} data points, {BATCH_SIZE} batch")
            with Timer.get_handle("model_fit_"+str(it)):
                model.fit(train_X, train_y, batch_size=BATCH_SIZE, validation_split=0.1, epochs=EPOCHS)
        if DO_SAVE:
            model.save_weights(WEIGHT_FILE)
            model.save(MODEL_FILE)

    if DO_DUMP:
        pickle_file = 'pickle_' + model_tag + '.weights'
        print('Dumped', pickle_file)
        with open(pickle_file, 'wb') as f:
            weights = model.get_weights()
            pickle.dump(weights, f)


    if DO_EVALUATE:
        print(model.evaluate(test_X, test_y))
        print(model.evaluate(head_X, head_y))
        print(model.evaluate(tail_X, tail_y))
        if LOAD_EVALUATE:
            print('Evaluate', EVALUATE_TAG)
            test_df = extra_test_df
            test_X, test_y = test_df[features], test_df[label]
            print(model.evaluate(test_df[features], test_df[label]))

        EVALUATE_TAG = LOAD_EVALUATE and EVALUATE_TAG or model_tag
        test_df = test_df[features + label]
        output = model.predict(test_X)
        test_df['prediction'] = np.where(output > 0.5, 1, 0)
        test_df['output'] = output
        test_df.to_csv(f'predict_{EVALUATE_TAG}.csv', index=False)
        print('Generated prediction', EVALUATE_TAG)

        predict_ana(test_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--object', help='object model', required=True)
    parser.add_argument('-e', '--evaluate')
    parser.add_argument('-t', '--train', action='store_true', help='train model')
    parser.add_argument('-l', '--load', action='store_true', help='load model')
    parser.add_argument('-s', '--save', action='store_true', help='save model')
    parser.add_argument('-d', '--dump', action='store_true', help='dump weights')
    parser.add_argument('-p', '--plot', action='store_true', help='plot training history')

    parser.add_argument('-g', '--gpu', action='store_true', help='use GPU')
    args = parser.parse_args()

    USE_GPU = args.gpu

    print(f"Using GPU? {USE_GPU}")
    TF_DEVICE = "/GPU:0" if USE_GPU else "/device:CPU:0"

    #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #tf.debugging.set_log_device_placement(True)

    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())


    MODEL_TAG = args.object or MODEL_TAG
    DO_LOAD = args.load
    DO_DUMP = args.dump
    DO_TRAIN = args.train
    DO_SAVE = args.save
    if args.evaluate:
        DO_EVALUATE = 1
        EVALUATE_TAG = args.evaluate
        LOAD_EVALUATE = 1
        DO_LOAD = 1
    else:
        DO_EVALUATE = 0

    with tf.device(TF_DEVICE):
        for i in range(2):
            keras_train(it=i, model_tag=MODEL_TAG, plot_loss=args.plot)


    Timer.print()