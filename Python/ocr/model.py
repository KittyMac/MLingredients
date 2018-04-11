from __future__ import division

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

import train

IMG_SIZE = [28,28,1]

def cnn_model(justTheImage=False):
	
	model = Sequential()

	model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMG_SIZE[1], IMG_SIZE[0], IMG_SIZE[2]), activation='relu'))
	model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(784, activation='relu'))
	model.add(Dense(len(train.ALPHABET), activation='softmax'))
	
	print(model.summary())
	
	lr = 0.01
	sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy',
				  optimizer=sgd,
				  metrics=['accuracy'])
	
	return model