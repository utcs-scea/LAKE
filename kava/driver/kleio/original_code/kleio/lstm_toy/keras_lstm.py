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
        #input to LSTM is [batch, timesteps, feature]
        units = 2
        inputs1 = Input(shape=(3, 1))
        lstm1 = LSTM(units, return_sequences=True)(inputs1)
        model = Model(inputs=inputs1, outputs=lstm1)

        #manually set so we always get the same result
        #ws = [np.array([[-1.0774542 ,  1.0071883 , -1.067705  ,  0.82218635]],
        #dtype=np.float32), np.array([[-0.7084664 ,  0.19819692,  0.23885503, -0.633831  ]],
        #dtype=np.float32), np.array([0., 1., 0., 0.], dtype=np.float32)]
        #print(model.layers[1].set_weights(ws))

        # define input data
        data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
        #data = array([0.1]).reshape((1,1,1))

        # make and show prediction
        print("prediction: ", model.predict(data))
        #print(model.layers[1].get_weights())

        #for e in zip(model.layers[1].trainable_weights, model.layers[1].get_weights()):
        #    print('Param %s:\n%s' % (e[0],e[1]))


        #https://stackoverflow.com/questions/42861460/how-to-interpret-weights-in-a-lstm-layer-in-keras
        # each output is 4 * number_of_units
        # each 4 is 
        #     i (input), f (forget), c (cell state) and o (output)

        for e in model.layers[1].trainable_weights:
            print(f"Trainable: {e}")

        print("Formatted: ")

        W = model.layers[1].get_weights()[0]
        U = model.layers[1].get_weights()[1]
        b = model.layers[1].get_weights()[2]

        W_i = W[:, :units]
        W_f = W[:, units: units * 2]
        W_c = W[:, units * 2: units * 3]
        W_o = W[:, units * 3:]

        print(f"Wi {W_i}")
        print(f"Wf {W_f}")
        print(f"Wc {W_c}")
        print(f"Wo {W_o}")

        U_i = U[:, :units]
        U_f = U[:, units: units * 2]
        U_c = U[:, units * 2: units * 3]
        U_o = U[:, units * 3:]

        b_i = b[:units]
        b_f = b[units: units * 2]
        b_c = b[units * 2: units * 3]
        b_o = b[units * 3:]



def test2():
    with tf.device('/cpu:0'):
        #data = array([0.1, 0.2]).reshape((1,2,2))
        #lstm1 = LSTM(5, return_sequences=True)
        #input to LSTM is [batch, timesteps, feature]
        #print(lstm1(data))

        data = array([0.1, 0.2, 0.3, 0.4]).reshape((2,1,2))
        lstm1 = LSTM(3, return_sequences=True)
        #input to LSTM is [batch, timesteps, feature]
        print(f"return_sequences True: {np.array(lstm1(data))}\n\nend")

        data = array([0.1, 0.2, 0.3, 0.4]).reshape((2,1,2))
        lstm1 = LSTM(3, return_sequences=False)
        #input to LSTM is [batch, timesteps, feature]
        print(f"return_sequences False: {np.array(lstm1(data))}\n\nend")


if __name__ == "__main__":
    #inputs1() 
    test2()