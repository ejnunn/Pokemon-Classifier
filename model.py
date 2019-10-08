import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time

# name of model to keep track of changing model architectures
NAME = 'Pokemon-classifier-cnn-64x2-{}'.format(int(time.time()))

# create a new log of the model to analyze results
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# load dataset and labels
print('Loading Model')

# TODO - Get different pickle datasets to train on
X = pickle.load(open('X.pickle','rb'))
y = pickle.load(open('y.pickle','rb'))

print('X.shape =', X.shape)
print('y.shape =', y.shape)

# normalize pixel data
X = X/255.0

# define model
model = Sequential()

# 1st convolution
model.add(Conv2D(64, (3,3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 2nd convolution
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# flatten model for fully connected layers
model.add(Flatten())

# fully-connected layer
model.add(Dense(256)) # 256 is arbitrary
model.add(Activation('relu'))

# fully-connected layer
model.add(Dense(151))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, batch_size=16, epochs=10, validation_split=0.1, callbacks=[tensorboard])














