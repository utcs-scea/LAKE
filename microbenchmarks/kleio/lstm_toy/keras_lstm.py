import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from numpy import array
import numpy as np


def inputs1():
    with tf.device('/cpu:0'):
        # define model
        inputs1 = Input(shape=(1, 1))
        lstm1 = LSTM(1, return_sequences=True)(inputs1)
        model = Model(inputs=inputs1, outputs=lstm1)

        #manually set so we always get the same result
        #ws = [np.array([[-1.0774542 ,  1.0071883 , -1.067705  ,  0.82218635]],
        #dtype=np.float32), np.array([[-0.7084664 ,  0.19819692,  0.23885503, -0.633831  ]],
        #dtype=np.float32), np.array([0., 1., 0., 0.], dtype=np.float32)]
        #print(model.layers[1].set_weights(ws))

        # define input data
        #data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
        data = array([0.1]).reshape((1,1,1))

        # make and show prediction
        print(model.predict(data))
        #print(model.layers[1].get_weights())


def test2():
    with tf.device('/cpu:0'):
        data = array([0.1, 0.2]).reshape((1,1,2))
        lstm1 = LSTM(4, return_sequences=True)
        #input to LSTM is [batch, timesteps, feature]
        print(lstm1(data))

        data = array([0.1, 0.2, 0.3, 0.4]).reshape((2,1,2))
        lstm1 = LSTM(4, return_sequences=True)
        #input to LSTM is [batch, timesteps, feature]
        print(lstm1(data))



if __name__ == "__main__":
    #inputs1() 
    test2()