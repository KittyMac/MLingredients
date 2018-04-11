from __future__ import division

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU

MAX_LEN = 80

def create_model(vocabularySize):
    
	model = Sequential()

	model.add(Embedding(vocabularySize, 512, input_length=MAX_LEN))
	model.add(SpatialDropout1D(0.2))
	model.add(GRU(512, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(1))

	model.add(Activation('sigmoid'))    

	model.compile(loss='binary_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	return model
