import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf

from skimage.exposure import rescale_intensity
from matplotlib.colors import rgb_to_hsv


import six
import keras
from keras import applications
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Add,
    Convolution2D,
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    LSTM,
    Reshape,
    TimeDistributed,
    ELU,
    Bidirectional
)
from keras.layers.convolutional import (
    Conv2D,
    Conv3D,
    MaxPooling2D,
    MaxPooling3D,
    AveragePooling2D,
    AveragePooling3D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.engine import InputSpec, Layer
'''
#############################
#                           #
#  Define hyperparameters   #
#                           #
#############################
'''
BATCH_SIZE = 20
# Images size 180x180x1 (since it's grayscaled)
IMG_H = 180
IMG_W =  180
IMG_C = 1
# Dropout probability
keep_prob = .2
# Sequence length is 10 (plus 5, because they'll be eaten by CNN)
SEQ = 10 
LEFT_PAD = 5 


'''
#############################
#                           #
#       Load data           #
#                           #
#############################
'''
# Load angle data
test_angles = pd.read_csv("/testing_cleaned.csv").steering_angle
train_angles = pd.read_csv("/training_cleaned.csv").angle


'''
This function will construct a sequence dict that holds images name with correspoding angle.
sequences at index i will look like [ [image_0, images_1, ... image_n] , Angle ]
'''
def construct_seq (path, angles):
    folder = os.listdir(path)
    folder.sort()
    size = len(folder)
    data = []
    for i in range (size):
        start = i-((SEQ+LEFT_PAD)-1)
        seq = []
        for j in range (start, i+1):
            j = 0 if j < 0 else j
            img = folder[j]
            seq.append(img)
        data.append([seq, angles[i]])
    return data

'''
    This function will load all images into memory.    
'''
def load_images (path, gray=True):
    dic = {}
    folder = os.listdir(path)
    folder.sort()
    for i, file in enumerate(folder):
        img = cv2.imread(path+"/"+file, cv2.IMREAD_GRAYSCALE) if gray else cv2.imread(path+"/"+file)
        img = cv2.resize(img, (IMG_W, IMG_H))
        dic[file] = img
    return dic
    
# Load images into memory
train_images = load_images(path_train_images)
test_images = load_images(path_test_images)

# Construct a dict that hold sequences data  (will load image names only -memory efficient-)
train_seq = construct_seq(path_train_images, train_angles)
test_seq = construct_seq(path_test_images, test_angles)

# Shuffle training sequences to perform better training
random.shuffle(train_seq)


'''
#############################
#                           #
#       Design Model        #
#                           #
#############################
'''
input_layer = Input((SEQ+LEFT_PAD, IMG_H, IMG_W, IMG_C))

conv1 = Conv3D(64, kernel_size=(3, 12, 12), strides=(1, 1, 1))(input_layer)
conv1 = MaxPooling3D(pool_size=(1,6,6))(conv1)
conv1 = Activation('relu')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = Dropout(keep_prob)(conv1)

conv2 = Conv3D(64, kernel_size=(2, 5, 5), strides=(1, 1, 1))(conv1)
conv2 = MaxPooling3D(pool_size=(1,2,2))(conv2)
conv2 = Activation('relu')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = Dropout(keep_prob)(conv2)

conv3 = Conv3D(64, kernel_size=(2, 5, 5), strides=(1, 1, 1))(conv2)
conv3 = Activation('relu')(conv3)

conv_31 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1))(conv1)
conv_31 = MaxPooling3D(pool_size=(1,3,3))(conv_31)
conv_31 = Activation('relu')(conv_31)
conv_31 = Add()([conv3, conv_31])
conv_31 = BatchNormalization()(conv_31)
conv_31  = Dropout(keep_prob)(conv_31)

conv4 = Conv3D(64, kernel_size=(2, 6, 6), strides=(1, 1, 1))(conv_31)
conv4 = Activation('relu')(conv4)

conv_42 = Conv3D(64, kernel_size=(3, 3, 3), strides=(1, 1, 1))(conv2)
conv_42 = MaxPooling3D(pool_size=(1,3,3))(conv_42)
conv_42 = Activation('relu')(conv_42)
conv_42 = Add()([conv4, conv_42])
conv_42 = BatchNormalization()(conv_42)
conv_42  = Dropout(keep_prob)(conv_42)


reshape = Reshape((SEQ, -1))(conv_42)
lstm = LSTM(10)(reshape)
net = Dense(100, activation='relu')(lstm)
net = Dropout(keep_prob)(net)
net = Dense(50, activation='relu')(net)
net = Dropout(keep_prob)(net)
net = Dense(10, activation='relu')(net)
net = Dropout(keep_prob)(net)
net = Dense(1) (net)

model = Model(inputs=input_layer, outputs=net)

optimizer = keras.optimizers.Adam(lr= 0.0005)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.summary()



# Training data generator
train_gen = DataGenerator(train_seq, train_images, batch_size=BATCH_SIZE, dim=(SEQ+LEFT_PAD, IMG_H, IMG_W), n_channels=IMG_C)


mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', save_best_only=True)
# Start training
hist = model.fit_generator(epochs=15, generator=train_gen, validation_data=test_gen, callbacks=[mc])



'''
#############################
#                           #
#     Data Generator        #
#                           #
#############################
'''
# Since loading all sequence will consume too much memory
#   will use mini batch gradient descent.
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, sequences_ids, img_dict, batch_size=32, dim=(15, 200, 200), n_channels=1, shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.img_dict = img_dict
        self.dim = dim
        self.sequences_ids = sequences_ids
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.sequences_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        seq_tmp = [self.sequences_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(seq_tmp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.sequences_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, seq_tmp):
        
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size))

        for i, pack in enumerate(seq_tmp):
            # Read angle, and sequence image names
            angle = pack[1]
            seq = pack[0]
            # Load images from image dictionary
            images = []
            for img_name in seq:
                img = self.img_dict[img_name]
                images.append(img)
            # Convert images to numpy array
            images = np.array(images).reshape(-1, *self.dim, self.n_channels)
            images = images.astype('float32')
            # Normalize images
            images /= 255.0

            X[i,] = images
            y[i] = angle
        
        return X, y


