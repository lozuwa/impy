"""
package: Images2Dataset
class: imageProcess
Author: Rodrigo Loza
Description: Utils methods 
"""
# General purpose
import os 
from tqdm import tqdm
# Image manipulation 
import cv2
import PIL
from PIL import Image
# Utils 
from utils import *
# Tensor manipulation
import numpy as np
from numpy import r_, c_

class preprocess:
	"""Allows operations on images"""
	def __init__(self, 
				data = []):
		self.data = data

	def resizeImages(self, 
					width = 300, 
					height = 300):
		"""
		Resizes all the images in the db. The new images are stored
		in a local folder named "dbResized"
		:param width: integer that tells the width to resize 
		:param height: integer that tells the height to resize
		"""
		# Create new directory
		DB_PATH = os.getcwd() + "/dbResized/"
		assert createFolder(DB_PATH) == True, PROBLEM_CREATING_FOLDER
		# Read Images
		keys = getDictKeys(self.data)
		for key in tqdm(keys):
			imgs = self.data.get(key, None)
			# Create subfolders for each subfolder
			NAME_SUBFOLDER = key.split("/")[-1]
			assert createFolder(DB_PATH + NAME_SUBFOLDER) == True, PROBLEM_CREATING_FOLDER
			# Iterate images 
			for img in imgs:
				# Filter nan values 
				if type(img) == str:
					# Open image
					frame = Image.open(img)
					# Resize image
					frame = frame.resize((width, height), PIL.Image.LANCZOS)
					IMAGE_NAME = "/" + img.split("/")[-1]
					# Save the image 
					frame.save(DB_PATH + NAME_SUBFOLDER + IMAGE_NAME)
				else:
					pass
		print(RESIZING_COMPLETE)

	def rgb2gray(self):
		"""
		Converts the images to grayscale. The new images are stored
		in a local folder named dbGray
		"""
		# Create new directory
		DB_PATH = os.getcwd() + "/dbGray/"
		assert createFolder(DB_PATH) == True, PROBLEM_CREATING_FOLDER
		# Read images 
		keys = getDictKeys(self.data)
		for key in tqdm(keys):
			imgs = self.data.get(key, None)
			# Create subfolders
			NAME_SUBFOLDER = key.split("/")[-1]
			assert createFolder(DB_PATH + NAME_SUBFOLDER) == True, PROBLEM_CREATING_FOLDER
			for img in imgs:
				# Filter nan values 
				if type(img) == str:
					# Read image
					frame = Image.open(img)
					# Convert RGBA to GRAYSCALE
					frame = frame.convert(mode = "1") #dither = PIL.FLOYDSTEINBERG)
					# Save the image
					IMAGE_NAME = "/" + img.split("/")[-1]
					frame.save(DB_PATH + NAME_SUBFOLDER + IMAGE_NAME)
				else:
					pass
		print(RBG2GRAY_COMPLETE)