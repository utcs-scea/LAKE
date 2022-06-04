import numpy as np
import io_helper
import os

random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function

def dropin(X, y):
    """
    The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
    """
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)



def preprocess(dir=None):
    if dir is None:
        dir = os.getcwd();

    normalfile = dir + "/" + "normal_test.pickle"
    x_train1 = io_helper.loadfrompickle(normalfile)
    
    # x_train1 = array[:,:-1]
    # x_train1 = array
    # y_train2 = array[:,-1]
    normallabelfile = dir + "/" + "normal_label.pickle"
    y_train1 = io_helper.loadfrompickle(normallabelfile)
    # y_train1 = arrayLabel

    attackfile = dir + "/" + "attack_test.pickle"
    x_train2 = io_helper.loadfrompickle(attackfile)
    
    # x_train1 = array[:,:-1]
    # x_train2 = array2
    # y_train2 = array[:,-1]
    attacklabelfile = dir + "/" + "attack_label.pickle"
    y_train2 = io_helper.loadfrompickle(attacklabelfile)
    # y_train2 = array2Label

    x_train = np.concatenate((x_train1, x_train2), axis=0)
    y_train = np.concatenate((y_train1, y_train2), axis=0)

    assert(len(x_train) == len(y_train))

    shuffler = np.random.permutation(len(y_train))
    y_train = y_train[shuffler]
    x_train = x_train[shuffler]
    # print(y_train)

    # print ("The train data size is that ")
    # print (x_train1.shape)
    # print (y_train1.shape)
    # print (x_train2.shape)
    # print (y_train2.shape)
    # print (x_train.shape)
    # print (y_train.shape)
    # print(y_train)

    return (x_train.astype(np.float), y_train.astype(np.float))


if __name__ =="__main__":

    preprocess()
