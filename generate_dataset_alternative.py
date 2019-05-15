# ------------------------------------------------------------------------------
# We generate the h5 file as our initial data. We have 90% training samples and 
# 10% test/dev samples. The size of each image is 32x32.
# ------------------------------------------------------------------------------

########################## first part: prepare data ###########################
from random import shuffle
import glob
import cv2
import datetime as dt
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
import os
import pandas as pd
from glob import glob
import string

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

shuffle_data = True  # shuffle the addresses
createFolder('./datasets/')
hdf5_train_path = './datasets/train_data.h5'  # file path for the created .hdf5 file
hdf5_test_path = './datasets/test_data.h5'  # file path for the created .hdf5 file
#cat_dog_train_path = 'E:/All about AI/train/*.jpg'  # the original data path

PATH = os.path.abspath(os.path.join('.', 'images'))
    # ../input/sample/images/
#SOURCE_IMAGES = os.path.join(PATH, "sample", "images")
SOURCE_IMAGES = os.path.join(PATH, "*.png")

# get all the image paths
address_text = PATH
addrs = []
for i in range(1, 62001):
#for i in range(1, 18601):

    addrs.append(address_text + "/" + str(i) + ".png")

classes = []

#for number in range(0,62):
for number in range(0,62):
    classes.append(number)

column_names = ['count', 'character']
data  = pd.read_csv('./data.csv', names = column_names)
labels = data.character.tolist()

#labels = [0 if 'cat' in addr else 1 for addr in addrs]
labels.remove('character')
# shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))  # use zip() to bind the images and labels together
   # print(c)
    shuffle(c)

    (addrs, labels) = zip(*c)  # *c is used to separate all the tuples in the list c,
    # "addrs" then contains all the shuffled paths and
    # "labels" contains all the shuffled labels.

# Divide the data into 80% for train and 20% for test
train_addrs = addrs[0:int(0.9 * len(addrs))]
train_labels = labels[0:int(0.9 * len(labels))]

train_labels = np.array(train_labels).astype(int)

#print(train_labels)

test_addrs = addrs[int(0.9 * len(addrs)):]
test_labels = labels[int(0.9 * len(labels)):]

test_labels = np.array(test_labels).astype(int)
#print(test_labels)
##################### second part: create the h5py object #####################
import numpy as np
import h5py

#train_shape = (len(train_addrs), 64, 64, 3)
train_shape = (len(train_addrs), 32, 32)
#test_shape = (len(test_addrs), 64, 64, 3)
test_shape = (len(test_addrs), 32, 32)

# open a hdf5 file and create arrays
f_train = h5py.File(hdf5_train_path, mode='w')
f_test = h5py.File(hdf5_test_path, mode='w')


# PIL.Image: the pixels range is 0-255,dtype is uint.
# matplotlib: the pixels range is 0-1,dtype is float.
f_train.create_dataset("train_set_x", train_shape, np.uint8)
f_test.create_dataset("test_set_x", test_shape, np.uint8)

# the ".create_dataset" object is like a dictionary, the "train_labels" is the key.
f_train.create_dataset("train_labels", (len(train_addrs),), np.uint8)
f_train["train_set_y"] = train_labels
#print(train_labels)
f_test.create_dataset("test_labels", (len(test_addrs),), np.uint8)
f_test["test_set_y"] = test_labels

f_test.create_dataset("classes", (len(classes),), np.uint8)
f_test["list_classes"] = classes

######################## third part: write the images #########################
import cv2


# loop over train paths
for i in range(len(train_addrs)):

    if i % 1000 == 0 and i > 1:
        print('Train data: {}/{}'.format(i, len(train_addrs)))

    addr = train_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)  # resize to (128,128)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 load images as BGR, convert it to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_train["train_set_x"][i,...] = img[None]

# loop over test paths
for i in range(len(test_addrs)):

    if i % 1000 == 0 and i > 1:
        print('Test data: {}/{}'.format(i, len(test_addrs)))

    addr = test_addrs[i]
    img = cv2.imread(addr)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f_test["test_set_x"][i, ...] = img[None]


f_train.close()
f_test.close()
