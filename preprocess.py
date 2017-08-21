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

class preprocess:
	"""Allows operations on images"""
	def __init__(self, data = []):
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
		DB_PATH = os.getcwd()+"/dbResized/"
		if isFolder(DB_PATH):
			pass
		else:
			os.mkdir(DB_PATH)
		# Read Images
		keys = getDictKeys(self.data)
		for key in tqdm(keys):
			imgs = self.data.get(key, None)
			# Create subfolders for each subfolder
			NAME_SUBFOLDER = key.split("/")[-1]
			if isFolder(DB_PATH + NAME_SUBFOLDER):
				pass
			else:
				os.mkdir(DB_PATH + NAME_SUBFOLDER) 
			for img in imgs:
				if type(img) == str:
					frame = Image.open(img)
					frame = frame.resize((width, height), PIL.Image.LANCZOS)
					IMAGE_NAME = "/" + img.split("/")[-1] 
					frame.save(DB_PATH + NAME_SUBFOLDER + IMAGE_NAME)
				else:
					pass
		print(RESIZING_COMPLETE)
