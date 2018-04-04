# the purpose of this class is to generate tons of images which are single character in image suitable for training
# ocr NN in use with VNDetectTextRectanglesRequest on iOS

import random
import cairocffi as cairo
import numpy as np
import cv2
from PIL import Image
import uuid
import train
import model

from keras import backend as K
import keras.callbacks

from keras.preprocessing import image
from scipy import ndimage

TRAINING_PATH = "./train/"

def speckle(img):
	severity = np.random.uniform(0, 0.6)
	blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
	img_speck = (img + blur)
	img_speck[img_speck > 1] = 1
	img_speck[img_speck <= 0] = 0
	return img_speck

def four_point_transform(image, pts, size):
	dst = np.array([
		[0, 0],
		[size[0] - 1, 0],
		[size[0] - 1, size[1] - 1],
		[0, size[1] - 1]], dtype = "float32")
 
	M = cv2.getPerspectiveTransform(pts, dst)
	warped = cv2.warpPerspective(image, M, size)
 
	return warped

def GenerateImage(text, img_w=model.IMG_SIZE[1], img_h=model.IMG_SIZE[0]):	
	surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_w, img_h)
	
	with cairo.Context(surface) as context:
		context.set_source_rgb(1, 1, 1)  # White
		context.paint()
		
		box = [0,0,img_w-1, img_h-1]
		origin_x = 0
		origin_y = 0
		
		if text != " ":
			context.select_font_face('Helvetica', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
		
			context.set_font_size(img_w)
			box = context.text_extents(text)
		
			# center the image, slight random rotations and size
			origin_x = (img_w - box[2]) / 2 + random.randint(-2,2)
			origin_y = img_h - (img_h - box[3]) / 2 + random.randint(-2,2)
			context.move_to(origin_x, origin_y)
			#context.move_to(top_left_x - int(box[0]), top_left_y - int(box[1]))
			context.set_source_rgb(0, 0, 0)
			context.show_text(text)
	
	buf = surface.get_data()
	a = np.frombuffer(buf, np.uint8)
	a.shape = (img_h, img_w, 4)
	
	# close crop the letter, expand to the size of the image
	imageCoordA = [box[0] + origin_x + 0, box[1] + origin_y + 0]
	imageCoordB = [box[0] + origin_x + box[2], box[1] + origin_y + 0]
	imageCoordC = [box[0] + origin_x + box[2], box[1] + origin_y + box[3]]
	imageCoordD = [box[0] + origin_x + 0, box[1] + origin_y + box[3]]
	
	a = four_point_transform(a, np.array([imageCoordA,imageCoordB,imageCoordC,imageCoordD], dtype='float32'), (img_w,img_h))
	
	a = a[:, :, 0]  # grab single channel
	a = a.astype(np.float32) / 255
	a = np.expand_dims(a, 0)
	#a = image.random_rotation(a, random.random() * 30 - 15)
	a = speckle(a)
	
	return a

def SaveImage(path,img):
	img = (img * 255).astype(np.uint8)
	im = Image.fromarray(img.reshape(model.IMG_SIZE[0],model.IMG_SIZE[1]))
	im.save(path)

def SaveImageToTraining(img, label):
	filepath = "{}/{}_".format(TRAINING_PATH, str(uuid.uuid4()))
	for x in train.ALPHABET:
		if x == label:
			filepath = filepath + "1_"
		else:
			filepath = filepath + "0_"
	
	filepath = filepath[:-1] + ".png"
			
	SaveImage(filepath,img)
	
for i in range(0,50):
	for x in train.ALPHABET:
		SaveImageToTraining(GenerateImage(x), x)




