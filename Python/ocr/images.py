import numpy as np
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import os
import glob
import gc
from PIL import Image

import model

K.set_image_data_format('channels_last')

def get_labels(img_path):
	filename = img_path.split('/')[-1]
	buttons = filename.split('_')
	
	retVal = []
	for i in range(1,len(buttons)):
		retVal.append(int(buttons[i].split('.')[0]))
	return retVal

def generate_image_array(dir_path, max_size):
	all_img_paths = glob.glob(os.path.join(dir_path, '*.png'))
	size = len(all_img_paths)
	if max_size != 0 and size > max_size:
		size = max_size
	return np.zeros((size, model.IMG_SIZE[1], model.IMG_SIZE[0], model.IMG_SIZE[2]), dtype='float32')

def load_image(imgs_idx, imgs, labels, img_path):
	img = img_to_array(load_img(img_path, grayscale=(model.IMG_SIZE[2] == 1), target_size=[model.IMG_SIZE[1],model.IMG_SIZE[0]]))
	
	img *= 1.0 / 255.0
	
	np.copyto(imgs[imgs_idx],img)
	
	labels.append(get_labels(img_path))
	return imgs_idx

def load_single_image(img_path):
	imgs = np.zeros((1, model.IMG_SIZE[1], model.IMG_SIZE[0], model.IMG_SIZE[2]), dtype='float32')
	img = img_to_array(load_img(img_path, grayscale=(model.IMG_SIZE[2] == 1), target_size=[model.IMG_SIZE[1],model.IMG_SIZE[0]]))
	np.copyto(imgs[0],img)
	return imgs

def load_images(imgs, labels, dir_path, max_size):
	all_img_paths = glob.glob(os.path.join(dir_path, '*.png'))
	np.random.shuffle(all_img_paths)
	n = 0
	for img_path in all_img_paths:
		if n % 10000 == 1:
			gc.collect()
		load_image(n, imgs, labels, img_path)
		n = n + 1
		if max_size != 0 and len(labels) >= max_size:
			return
