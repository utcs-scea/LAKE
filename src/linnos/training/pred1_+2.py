import pandas as pd
import numpy as np
import os
from sklearn.utils import class_weight
from keras import regularizers
# from keras.models import Sequential
# from keras.layers import Dense
from itertools import product
import keras.backend as K
from functools import partial
#from keras import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
import sys


train_input_path = sys.argv[1]
percentile_threshold = float(sys.argv[2])

custom_loss = 5.0

train_data = pd.read_csv(train_input_path, dtype='float32',sep=',', header=None)
train_data = train_data.sample(frac=1).reset_index(drop=True)
train_data = train_data.values

train_input = train_data[:,:31]
train_output = train_data[:,31]

lat_threshold = np.percentile(train_output, percentile_threshold)
print("lat_threshold: ",lat_threshold)
num_train_entries = int(len(train_output) * 0.80)
print("num train entries: ",num_train_entries)

train_Xtrn = train_input[:num_train_entries,:]
train_Xtst = train_input[num_train_entries:,:]
train_Xtrn = np.array(train_Xtrn)
train_Xtst = np.array(train_Xtst)

#Classification
train_y = []
for num in train_output:
    labels = [0] * 2
    if num < lat_threshold:
        labels[0] = 1
    else:
        labels[1] = 1
    train_y.append(labels)


#print(y)
train_ytrn = train_y[:num_train_entries]
train_ytst = train_y[num_train_entries:]
train_ytrn = np.array(train_ytrn)
train_ytst = np.array(train_ytst)

print(type(train_ytrn))
print(type(train_Xtrn))

#-------------------------Custom Loss--------------------------
def w_categorical_crossentropy(y_true, y_pred, weights):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    weights = weights.astype(float)
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    cross_ent = tf.keras.losses.categorical_hinge(y_true,y_pred) #K.categorical_crossentropy(y_true,y_pred, from_logits=False)
    return cross_ent * final_mask

w_array = np.ones((2,2))
w_array[1, 0] = custom_loss   #Custom Loss Multiplier
#w_array[0, 1] = 1.2

ncce = partial(w_categorical_crossentropy, weights=w_array)
#----------------------------------------------------------------------
model = Sequential()
model.add(Dense(256, input_dim=31, activation='relu'))
model.add(Dense(256, input_dim=256, activation='relu'))
model.add(Dense(256, input_dim=256, activation='relu'))
model.add(Dense(2, activation='linear'))#,kernel_regularizer=regularizers.l2(0.001)))
model.compile(optimizer='adam', loss=ncce, metrics=['accuracy'])

for i in range(50):
    model.fit(train_Xtrn, train_ytrn, epochs=1, batch_size=128, verbose=0) 
    print('Iteration '+str(i)+'\n')
    print('On test dataset:\n')
    train_Y_test = np.argmax(train_ytst, axis=1) # Convert one-hot to index
    train_y_pred = np.argmax(model.predict(train_Xtst), axis=-1)
    #model.predict_classes(train_Xtst)
    print(classification_report(train_Y_test, train_y_pred, digits=4))
 
    count = 0
    for layer in model.layers: 
        weights = layer.get_weights()[0]
        biases = layer.get_weights()[1]
        name = train_input_path +'.weight_' + str(count) + '.csv' #output_dir_path+'/SSD_31col_iteration_'+str(i)+'_weightcustom1_' + str(count) + '.csv'
        name_b = train_input_path + '.bias_' + str(count) + '.csv' #output_dir_path+'/SSD_31col_iteration_'+str(i)+'_biascustom1_' + str(count) + '.csv'
        np.savetxt(name, weights, delimiter=',')
        np.savetxt(name_b, biases, delimiter=',')
        count += 1


'''
from keras.models import model_from_json
import os

# save model to JSON
model_json = model.to_json()
with open("preidctionmodelcl19.json", "w") as json_file:
    json_file.write(model_json)
#save weights to HDF5
model.save_weights("predictionmodelcl19.h5")
print("Saved model to disk")
'''
