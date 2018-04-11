from __future__ import division

from keras import backend as K

import time
import numpy as np
import subprocess

import os
import coremltools   
import h5py
import struct

import model
import images

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890,"
ALPHABET_LEN = len(ALPHABET)

ACCURACY_THRESHOLD = 0.01
MAX_SAMPLES_TO_TRAIN = 0
TRAIN_PATH = "./train/"
PERMANENT_PATH = "./permanent/"

def Learn():
	
	model_h5_name = "ocr.h5"
	model_coreml_name = "ocr.mlmodel"
			
	# 2. Load the samples
	print("Loading samples...")
	total_labels = []
	total_imgs = images.generate_image_array(TRAIN_PATH, MAX_SAMPLES_TO_TRAIN)
	images.load_images(total_imgs, total_labels, TRAIN_PATH, MAX_SAMPLES_TO_TRAIN)
	
	permanent_labels = []
	permanent_imgs = images.generate_image_array(PERMANENT_PATH, MAX_SAMPLES_TO_TRAIN)
	images.load_images(permanent_imgs, permanent_labels, PERMANENT_PATH, MAX_SAMPLES_TO_TRAIN)

	
	total_imgs = np.concatenate((total_imgs,permanent_imgs), axis=0) if len(permanent_imgs) > 0 else total_imgs
	total_labels = np.concatenate((total_labels,permanent_labels), axis=0) if len(permanent_labels) > 0 else total_labels

		
	#print(total_labels)
	#print(total_imgs)
	
	# 3. properly calculate the class weights
	total_by_class = np.zeros(ALPHABET_LEN, dtype='uint32')
	normalized_weights = np.zeros(len(total_labels), dtype='float32')
	
	# count up the number of each class
	for i in range(0,len(total_labels)):
		for j in range(0,ALPHABET_LEN):
			total_by_class[j] += total_labels[i][j]
		
	# relate each class to each other based on the total amount
	for j in range(0,ALPHABET_LEN):
		if total_by_class[j] == 0:
			total_by_class[j] = 0
		else:
			total_by_class[j] = len(total_labels) / total_by_class[j]
	
	# relate each sample to its class weights
	for i in range(0,len(total_labels)):
		normalized_weights[i] = 0
		for j in range(0,ALPHABET_LEN):
			normalized_weights[i] += total_labels[i][j] * total_by_class[j]
	
	# normalize all sample weights
	max_weight = max(normalized_weights)
	for i in range(0,len(total_labels)):
		normalized_weights[i] /= max_weight
		normalized_weights[i] *= 0.2
	
	# 3. Train the CNN on the samples
	cnn_model = model.cnn_model()
	
	for iteration in range(0,5):
		print("*** iteration ", iteration)
		cnn_model.fit(total_imgs, total_labels,
			batch_size=32,
			epochs=10,
			shuffle=True,
			verbose=1,
			sample_weight=normalized_weights
			)
	
	cnn_model.save(model_h5_name)
	
	output_labels = [] 
	for x in ALPHABET:
		output_labels.append(x)
	print(output_labels)
	coreml_model = coremltools.converters.keras.convert(model_h5_name,input_names='image',image_input_names = 'image',class_labels = output_labels, image_scale=1/255.0)   
	coreml_model.author = 'Rocco Bowling'   
	coreml_model.short_description = 'Indentify the character in the image'
	coreml_model.input_description['image'] = 'Close cropped image of a character'
	coreml_model.save(model_coreml_name)
	print("Conversion to coreml finished...")
	
	
	
	
if __name__ == '__main__':
	Learn()