import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time

# load dataset and labels
print('Loading Model')

# TODO - Get different pickle datasets to train on
X = pickle.load(open('X_10_pokemon.pickle','rb'))
y = pickle.load(open('y_10_pokemon.pickle','rb'))

print('X.shape =', X.shape)
print('y.shape =', y.shape)

# normalize pixel data
X = X/255.0

# test parameters
#create a log for models using every combination of test variables
conv_layers = [1, 2, 3]
layer_sizes = [32, 64, 128]
dense_layers = [0, 1, 2]

for conv_layer in conv_layers:
	for layer_size in layer_sizes:
		for dense_layer in dense_layers:
			# name of model to keep track of changing model architectures
			NAME = '{}-conv-{}-nodes-{}-dense-{}'.format(conv_layer, layer_size, dense_layer, int(time.time()))
			# create a new log of the model to analyze results
			tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

			# define model
			model = Sequential()

			# 1st convolution
			model.add(Conv2D(layer_size, (3,3), input_shape=X.shape[1:]))
			model.add(Activation('relu'))
			model.add(MaxPooling2D(pool_size=(2,2)))

			# add additional conv layers
			for _ in range(conv_layer-1):
				# convolution
				model.add(Conv2D(layer_size, (3,3)))
				model.add(Activation('relu'))
				model.add(MaxPooling2D(pool_size=(2,2)))

			# flatten model for fully connected layers
			model.add(Flatten())
			for _ in range(dense_layer):
				# fully-connected layer
				model.add(Dense(layer_size))
				model.add(Activation('relu'))

			# final fully-connected layer (151 matches number of Pokemon categories)
			model.add(Dense(151))
			model.add(Activation('softmax'))

			model.compile(loss='sparse_categorical_crossentropy',
			              optimizer='adam',
			              metrics=['accuracy'])

			model.fit(X, y, batch_size=16, epochs=10, validation_split=0.1, callbacks=[tensorboard])














