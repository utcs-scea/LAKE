#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import matplotlib.pyplot as plt
import numpy as np
import time
import os
# from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM
# from keras.models import Sequential
# from keras.models import model_from_json

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import preprocess

# Global hyper-parameters
sequence_length = 20
epochs = 5
batch_size = 16
feature_dimension = 341


def save_model_weight_into_file(model, modelname="model.json", weight="model.h5"):
    model_json = model.to_json()
    with open(modelname, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weight)
    print("Saved model to disk in {} and {}".format(modelname,weight))


def load_model_and_wieght_from_file(modelname="model.json", weight="model.h5"):

    json_file = open(modelname, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight)
    print("Loaded model from disk, you can do more analysis more")

    pass

def build_model():
    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(20, feature_dimension)))
    # layers = {'input': feature_dimension, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': feature_dimension}

    # model.add(layers.Dense(units=feature_dimension))
    model.add(layers.LSTM(units=64, return_sequences=True, dropout=0.3))
    model.add(layers.LSTM(units=256, return_sequences=True, dropout=0.3))
    model.add(layers.LSTM(units=128, dropout=0.2))
    model.add(layers.Dense(units=64, activation='relu'))    # Absolutely no idea why this worked
    model.add(layers.Dense(units=1, activation='sigmoid'))

    # model.add(LSTM(
    #         input_length=sequence_length,
    #         input_dim=layers['input'],
    #         output_dim=layers['hidden1'],
    #         return_sequences=True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(
    #         layers['hidden2'],
    #         return_sequences=True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(
    #         layers['hidden3'],
    #         return_sequences=False))
    # model.add(Dropout(0.2))

    # model.add(Dense(
    #         output_dim=layers['output'],activation='softmax'))
    #model.add(Activation("linear"))

    start = time.time()

    # model.compile(loss="categorical_crossentropy", optimizer='rmsprop',  metrics=['accuracy'])
    model.compile(loss="binary_crossentropy", optimizer='adam',  metrics=['accuracy'])
    #model.compile(loss="mse", optimizer="rmsprop")

    print ("Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, data=None):

    X_train, y_train  = preprocess.preprocess()
    print(X_train.shape)
    print(y_train.shape)
    print(X_train)
    print(y_train)
    # global_start_time = time.time()

    # if data is None:
    #     print 'Loading data... '
    #     # train on first 700 samples and test on next 300 samples (has anomaly)
    #     X_train, y_train  = preprocess.preprocess()
    # else:
    #     X_train, y_train = data

    print ("X_train, y_train,shape")
    print (X_train.shape)
    print (y_train.shape)
    print ('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model()
        print("Training...")
        model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.05)
        model.summary()
        os.chdir(os.path.dirname(__file__))
        model.save( os.path.join(os.getcwd(), "model.tf") )
        print("Done Training...")

    # predicted = model.predict(X_test)
    #print("Reshaping predicted")
    #predicted = np.reshape(predicted, (predicted.size,))


    """
    except KeyboardInterrupt:
        print("prediction exception")
        print 'Training duration (s) : ', time.time() - global_start_time
        return model, y_test, 0
    try:
        plt.figure(1)
        plt.subplot(311)
        plt.title("Actual Test Signal w/Anomalies")
        plt.plot(y_test[:len(y_test)], 'b')
        plt.subplot(312)
        plt.title("Predicted Signal")
        plt.plot(predicted[:len(y_test)], 'g')
        plt.subplot(313)
        plt.title("Squared Error")
        mse = ((y_test - predicted) ** 2)
        plt.plot(mse, 'r')
        plt.show()
    except Exception as e:
        print("plotting exception")
        print str(e)
    print 'Training duration (s) : ', time.time() - global_start_time
    return model, y_test, predicted
    """

if __name__ == "__main__":
    run_network()
