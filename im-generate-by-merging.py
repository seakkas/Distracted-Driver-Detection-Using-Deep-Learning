# -*- coding: utf-8 -*-


import numpy as np
import glob
from random import shuffle
import random
import cv2
import shutil
import os



def get_image(filename):
    image = cv2.imread(filename)
    image = np.asarray(image, np.uint8)
    return image



# generated images named starting from 1
# example: gen_1.jpg, gen2_.jpg
# this variable is used to give unique name to each image
img_name = 1

# if this one set to 1.0, it will generate images with the same number of training images
gen_ratio = 2

for i in range(10):

	out_dir = 'dataset/merged/c'+ str(i)

	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	images = glob.glob('dataset/org/train/c' + str(i) +'/*.jpg')
	num_images = len(images)
	shuffle(images)
	for j in range(round(len(images) * gen_ratio)):
		img1 = get_image(images[random.randint(0,num_images -1)])
		img2 = get_image(images[random.randint(0,num_images -1)] )
		gen_image = img1[:,:240,:]
		gen_image = np.concatenate((gen_image, img2[:,240:,:] ),1)
		cv2.imwrite(out_dir + '/' +'gen_' + str(img_name) + '.jpg' ,gen_image)
		img_name += 1
	# copy original images
	for im in images:
		shutil.copy2(im,out_dir)

	print('class #', i, 'completed')


