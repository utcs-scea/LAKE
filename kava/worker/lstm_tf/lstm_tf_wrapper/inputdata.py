#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

import io_helper

# Return a list of file names in the given directory
def readfilesfromAdir(dataset):
    #read a list of files
    files = os.listdir(dataset)
    files_absolute_paths = []
    for i in files:
        files_absolute_paths.append(dataset+str(i))
    return files_absolute_paths


file = "ADFA-LD/Training_Data_Master/UTD-0001.txt"

#this is used to read a char sequence from
# Return a list of syscall number in the given file
def readCharsFromFile(file):
    channel_values = open(file).read().split()
    #print (len(channel_values))
    #channel_values is a list
    return channel_values
    #print (channel_values[800:819])

def get_attack_subdir(path):

    subdirectories = os.listdir(path)
    # Remove DS_Store
    subdirectories = [i for i in subdirectories if "DS_Store" not in i]
    for i in range(0,len(subdirectories)):
        subdirectories[i] = path + subdirectories[i] + '/'

    # print (subdirectories)
    return (subdirectories)

def get_anamoly_call_sequences(dir):
    subdirs = get_attack_subdir(dir)
    allthelist = []
    for d in subdirs:
        allthelist = allthelist + get_all_call_sequences(d)
    print(len(allthelist))
    return allthelist


def get_all_call_sequences(dire):
    files = readfilesfromAdir(dire)     # files are a list of full path name to each training file
    allthelist = []             # allthelist is a list of list of syscall sequences
    print (len(files))          # 834

    for eachfile in files:
        if not eachfile.endswith("DS_Store"):
            allthelist.append(readCharsFromFile(eachfile))
        else:
            print ("Skip the file "+ str(eachfile))

    elements = []       # This is used to keep track of unique syscall id
    for item in allthelist:
        for key in item:
            if key not in elements:
                elements.append(key)

    elements = map(int, elements)
    elements = sorted(elements)

    print ("All syscall id:")
    print (elements)

    print ("Max syscall id:")
    print (max(elements))

    #print ("The length elements:")
    #print (len(elements))
    print ("Total number of files:")
    print (len(allthelist))

    #clean the all list data set
    _max = 0
    for i in range(0,len(allthelist)):
        _max = max(_max,len(allthelist[i]))
        allthelist[i] = list(map(int,allthelist[i]))


    # The max number of syscalls for one application
    print ("The maximum length of a sequence is: {}".format(_max))
    return (allthelist)

## shift the data for analysis
def shift(seq, n):
    n = n % len(seq)
    return seq[n:] + seq[:n]


def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.
    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v
        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1  # The last element 0, is reserverd as normal target???
    return result.astype(int)

"""
The num_class here is set as 341
"""

#one function do one thing
def sequence_n_gram_parsing(alist,n_gram=20,num_class=341):
    if len(alist) <= n_gram:
        return alist

    ans = []
    for i in range(0,len(alist)-n_gram+1,1):
        tmp = alist[i:i+n_gram]
        oneHot = convertToOneHot(np.asarray(tmp), num_class)
        ans.append(oneHot)

    #transform into nmup arrray
    ans = np.array(ans) # ans is an array of 20 one-hot encoding for a given syscall file
    return (ans)

def lists_of_list_into_big_matrix(allthelist,n_gram=20, attack=False):

    """
    allthelist is a list of list of syscall numbers, each element in
    allthelist is a list of syscalls for one particular application
    """
    array = sequence_n_gram_parsing(allthelist[0])

    for i in range(1,len(allthelist),1):
        tmp = sequence_n_gram_parsing(allthelist[i])

        #print ("tmp shape")
        #print (tmp.shape)

        array = np.concatenate((array, tmp), axis=0)    # array is a list of 20 one-hot syscall sequences all together

        percent = (i+0.0)/len(allthelist)
        io_helper.drawProgressBar(percent)

        if (len(array)> 30000):
            break
        #print ("array shape")
        #print (array.shape)


    # print (array.shape)
    if not attack:
        labels = np.zeros(array.shape[0])
        io_helper.saveintopickle(array,"normal_test.pickle")
        io_helper.saveintopickle(labels, "normal_label.pickle")
    else:
        labels = np.ones(array.shape[0])
        io_helper.saveintopickle(array,"attack_test.pickle")
        io_helper.saveintopickle(labels, "attack_label.pickle")
    print ("done")



if __name__ == "__main__":
    dirc = "ADFA-LD/Training_Data_Master/"
    dirc_val = "ADFA-LD/Validation_Data_Master/"
    dic_attack ="ADFA-LD/Attack_Data_Master/"
    #train1 = get_all_call_sequences(dirc)

    #test = [i for i in range(0,300)]
    #array = sequence_n_gram_parsing(test)
    #print (type(array))
    #print (array.shape)

    att2 = get_anamoly_call_sequences(dic_attack)
    # print ("XxxxxxxXXXXXXXXXXX")
    #val1 = get_all_call_sequences(dirc_val)

    # This stores only harmless training data
    att = get_all_call_sequences(dirc)
    lists_of_list_into_big_matrix(att, attack=False)
    lists_of_list_into_big_matrix(att2, attack=True)
