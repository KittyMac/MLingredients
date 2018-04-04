from __future__ import division

from keras import backend as K

from keras.preprocessing import sequence
import numpy as np
import os
import coremltools
import model
import json
import operator

def tokenize(sentence,mapping):
    words = sentence.split(' ')
    tokenizedSentence = []
    for word in words:
        if mapping.get(word) != None:
            tokenizedSentence.append(mapping[word])
        else:
            tokenizedSentence.append(0)
    return tokenizedSentence


def Learn():
	
	model_h5_name = "ingredients.h5"
	model_coreml_name = "ingredients.mlmodel"
	
	# 1. load in the product json
	print("loading the product json")
	productData = json.load(open('product.json'))
	
	# 2. load in the ingredient word dictionary where the key is the word and the value is the index
	print("loading the ingredient vocabulary")
	with open('words.txt') as file:
		ingredientsWordList = file.readlines()
	ingredientsWordList = [x.strip() for x in ingredientsWordList]
	
	ingredientWordToIndex = {}
	index = 0
	for ingredient in ingredientsWordList:
		index += 1
		ingredientWordToIndex[ingredient] = index
		
		
	# 3. create the model
	print("creating the RNN")
	_model = model.create_model(len(ingredientsWordList) + 3)
	
	
	# 4. generate the training data
	print("generating the training data")
	
	unlabeledInput = []
	unlabeledInputProducts = []
	
	trainingInput = []
	trainingOutput = []
	for product in productData:
		healthIndex = float(product['health']) / 10
		# ignore unlabled products
		if healthIndex > 0:
			trainingInput.append(tokenize(product['ingredients'], ingredientWordToIndex))
			trainingOutput.append(healthIndex)
		else:
			unlabeledInput.append(tokenize(product['ingredients'], ingredientWordToIndex))
			unlabeledInputProducts.append(product)
	
	
	# add some base truth ( a product with all unrecognized ingredients is worht 0 health? )
	trainingInput.append([])
	trainingOutput.append(0)
	
	
	# pad all of our input sentences to maximum sentence length
	trainingInput = sequence.pad_sequences(trainingInput, maxlen=model.MAX_LEN)
	unlabeledInput = sequence.pad_sequences(unlabeledInput, maxlen=model.MAX_LEN)
	
	
	
	# 5. train model on word index arrays with expected normalized health output (note: health value of 0 should be skipped)
	for i in range(0,10):
		_model.fit(trainingInput, trainingOutput,
	        batch_size=32,
	        epochs=10,
	        shuffle=True,
	        verbose=1,
			validation_split = 0.1
	        )
	
	
	# 6. predict against some of our unlabelled products to see how well it does?
	predictions = _model.predict(unlabeledInput)

	for i in range(0,60):
		print(predictions[i], unlabeledInputProducts[i]['name'])
	
	_model.save(model_h5_name)

	# 7. export to coreml
	coreml_model = coremltools.converters.keras.convert(model_h5_name,input_names=['ingredients'], output_names=['health'])   
	coreml_model.author = 'Rocco Bowling'   
	coreml_model.short_description = 'Classify how healthy a food is by examining its ingredients list'
	coreml_model.input_description['ingredients'] = 'A properly tokenized string of the ingredients to classify'
	coreml_model.save(model_coreml_name)
	print("Conversion to coreml finished...")
	
	

Learn()