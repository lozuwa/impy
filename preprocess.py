"""
package: Images2Dataset
class: imageProcess
Author: Rodrigo Loza
Description: Preprocess image dataset or singular images.
"""
# General purpose
import os
from tqdm import tqdm
import math
# Image manipulation
import cv2
#import Pixels
from PIL import Image
import PIL
# Tensor manipulation
import numpy as np
from numpy import r_, c_
# Model selection utils from scikit-learn
from sklearn.model_selection import train_test_split
# Utils
from .utils import *

class preprocessImageDataset:
	"""
	Allows operations on images
	"""
	def __init__(self,
				data = []):
		"""
		Constructor of the preprcoessImageDataset
		:param data: pandas dataframe that contains as features the
					name of the classes and as values the paths of
					the images.
		"""
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
		DB_PATH = os.path.join(os.getcwd(), "dbResized")
		assert createFolder(DB_PATH) == True,\
				PROBLEM_CREATING_FOLDER
		# Read Images
		keys = getDictKeys(self.data)
		for key in tqdm(keys):
			imgs = self.data.get(key, None)
			# Create subfolders for each subfolder
			DIR, NAME_SUBFOLDER = os.path.split(key)
			#NAME_SUBFOLDER = key.split("/")[-1]
			assert createFolder(os.path.join(DB_PATH, NAME_SUBFOLDER)) == True,\
					PROBLEM_CREATING_FOLDER
			# Iterate images
			for img in imgs:
				# Filter nan values
				if type(img) == str:
					# Open image
					frame = Image.open(img)
					# Resize image
					frame = frame.resize((width, height),\
											PIL.Image.LANCZOS)
					dir_, IMAGE_NAME = os.path.split(img)
					#IMAGE_NAME = "/" + img.split("/")[-1]
					# Save the image
					frame.save(os.path.join(DB_PATH, NAME_SUBFOLDER, IMAGE_NAME))
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
		assert createFolder(DB_PATH) == True,\
				PROBLEM_CREATING_FOLDER
		# Read images
		keys = getDictKeys(self.data)
		for key in tqdm(keys):
			imgs = self.data.get(key, None)
			# Create subfolders
			NAME_SUBFOLDER = key.split("/")[-1]
			assert createFolder(DB_PATH + NAME_SUBFOLDER) == True,\
					PROBLEM_CREATING_FOLDER
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

	def splitImageDataset(self,
						trainSize = 0.80,
						validationSize = 0.20):
		"""
		Splits the dataset into a training and a validation set
		:param trainSize: int that represents the size of the training set
		:param validationSize: int that represents the size of the
								validation set
		: return: four vectors that contain the training and test examples
					trainImgsPaths, trainImgsClass
					testImgsPaths, testImgsClass
		"""
		# Split dataset in the way that we get the paths of the images
		# in a train set and a test set.
		# ----- | class0, class1, class2, class3, class4
		# TRAIN | path0   path1   path2   path3   path4
		# TRAIN | path0   path1   path2   path3   path4
		# TRAIN | path0   path1   path2   path3   path4
		# TRAIN | path0   path1   path2   path3   path4
		# TEST* | path0   path1   path2   path3   path4
		# TEST* | path0   path1   path2   path3   path4
		trainX, testX, trainY, testY = train_test_split(self.data,\
														self.data,\
														train_size = trainSize,\
														test_size = validationSize)
		print(trainX.shape, testX.shape)
		# Once the dataset has been partitioned, we are going
		# append the paths of the matrix into a single vector.
		# TRAIN will hold all the classes' paths in train
		# TEST will hold all the classes' paths in test
		# Each class wil have the same amount of examples.
		trainImgsPaths = []
		trainImgsClass = []
		testImgsPaths = []
		testImgsClass = []
		# Append TRAIN and TEST images
		keys = self.data.keys()
		for key in keys:
			imgsTrain = trainX[key]
			imgsTest = testX[key]
			for imgTrain in imgsTrain:
				if type(imgTrain) != str:
					pass
				else:
					trainImgsPaths.append(imgTrain)
					trainImgsClass.append(key)
			for imgTest in imgsTest:
				if type(imgTest) != str:
					pass
				else:
					testImgsPaths.append(imgTest)
					testImgsClass.append(key)
		return trainImgsPaths,\
				trainImgsClass,\
				testImgsPaths,\
				testImgsClass

	def saveImageDatasetKeras(self,
								trainImgsPaths,
								trainImgsClass,
								testImgsPaths,
								testImgsClass):
		# Create folder
		DB_PATH = os.getcwd() + "/dbKerasFormat/"
		result = createFolder(DB_PATH)
		assert result == True,\
				PROBLEM_CREATING_FOLDER
		# Create train subfolder
		TRAIN_SUBFOLDER = "train/"
		result = createFolder(DB_PATH + TRAIN_SUBFOLDER)
		assert result == True,\
				PROBLEM_CREATING_FOLDER
		# Create validation subfolder
		VALIDATION_SUBFOLDER = "validation/"
		result = createFolder(DB_PATH + VALIDATION_SUBFOLDER)
		assert result == True,\
				PROBLEM_CREATING_FOLDER
		# Create classes folders inside train and validation
		keys = self.data.keys()
		for key in keys:
			# Train subfolder
			NAME_SUBFOLDER = key.split("/")[-1]
			result = createFolder(DB_PATH + TRAIN_SUBFOLDER +\
								 NAME_SUBFOLDER)
			assert result == True,\
					PROBLEM_CREATING_FOLDER
			# Test subfolder
			NAME_SUBFOLDER = key.split("/")[-1]
			result = createFolder(DB_PATH + VALIDATION_SUBFOLDER +\
								 NAME_SUBFOLDER)
			assert result == True,\
					PROBLEM_CREATING_FOLDER

		######################## OPTIMIZE ########################
		# Save train images
		# Read classes in trainImgsClass
		for i in tqdm(range(len(trainImgsClass))):
			imgClass = trainImgsClass[i].split("/")[-1]
			for key in keys:
				NAME_SUBFOLDER = key.split("/")[-1]
				#print(imgClass, NAME_SUBFOLDER)
				# If they are the same class, then save the image
				if imgClass == NAME_SUBFOLDER:
					NAME_SUBFOLDER += "/"
					NAME_IMG = trainImgsPaths[i]
					frame = Image.open(NAME_IMG)
					NAME_IMG = NAME_IMG.split("/")[-1]
					frame.save(DB_PATH + TRAIN_SUBFOLDER +\
								NAME_SUBFOLDER + NAME_IMG)
				else:
					pass
		# Save test images
		# Read classes in testImgsClass
		for i in tqdm(range(len(testImgsClass))):
			imgClass = testImgsClass[i].split("/")[-1]
			for key in keys: 
				NAME_SUBFOLDER = key.split("/")[-1]
				# If they are the same class, then save the image
				if imgClass == NAME_SUBFOLDER:
					NAME_SUBFOLDER += "/"
					NAME_IMG = testImgsPaths[i]
					frame = Image.open(NAME_IMG)
					NAME_IMG = NAME_IMG.split("/")[-1]
					frame.save(DB_PATH + VALIDATION_SUBFOLDER +\
								NAME_SUBFOLDER + NAME_IMG)
				else:
					pass
		##########################################################

class preprocessImage:
	"""
	Allows operations on single images.
	"""
	def __init__(self):
		"""
		Constructor of preprocessImage class
		"""
		pass

	def divideIntoPatches(self,
						image_width,
						image_height,
						slide_window_size = (0, 0),
						stride_size = (0, 0),
						padding = "VALID",
						number_patches = (1, 1)):
		"""
		Divides the image into N patches depending on the stride size,
		the sliding window size and the type of padding.
		:param image_width: int that represents the width of the image
		:param image_height: int that represents the height of the image
		:param slide_window_size: tuple (width, height) that represents the size
									of the sliding window
		:param stride_size: tuple (width, height) that represents the amount
							of pixels to move on height and width direction
		:param padding: string ("VALID", "SAME", "VALID_FIT_ALL") that tells the type of
						padding
		:param number_of_patches: tuple (number_width, number_height) that 
									contains the number of patches in each axis
		: return: a list containing the number of patches that fill the
				given parameters, int containing the number of row patches,
				int containing the number of column patches
		"""
		# Get sliding window sizes
		slide_window_width, slide_window_height = slide_window_size[0], slide_window_size[1]
		assert (slide_window_height < image_height) and (slide_window_width < image_width),\
				SLIDE_WINDOW_SIZE_TOO_BIG
		# Get strides sizes
		stride_width, stride_height = stride_size[0], stride_size[1]
		assert (stride_height < image_height) and (stride_width < image_width),\
				STRIDE_SIZE_TOO_BIG
		# Start padding operation
		if padding == 'VALID':
			start_pixels_height = 0
			end_pixels_height = slide_window_height
			start_pixels_width = 0
			end_pixels_width = slide_window_width
			patches_coordinates = []
			number_patches_height, number_patches_width = get_valid_padding(slide_window_height,
																		 stride_height,
																		 image_height,
																		 slide_window_width,
																		 stride_width,
																		 image_width)
			print('numberPatchesHeight: ', number_patches_height, 'numberPatchesWidth: ', number_patches_width)
			for i in range(number_patches_height):
				for j in range(number_patches_width):
					patches_coordinates.append([start_pixels_height,\
													start_pixels_width,\
													end_pixels_height,\
													end_pixels_width])
					# Update width with strides
					start_pixels_width += stride_width
					end_pixels_width += stride_width
				# Re-initialize the width parameters 
				start_pixels_width = 0
				end_pixels_width = slide_window_width
				# Update height with height stride size
				start_pixels_height += stride_height
				end_pixels_height += stride_height
			return patches_coordinates,\
					number_patches_height,\
					number_patches_width

		elif padding == 'SAME':
			start_pixels_height = 0
			end_pixels_height = slide_window_height
			start_pixels_width = 0
			end_pixels_width = slide_window_width
			patches_coordinates = []
			# Modify image tensor
			zeros_h, zeros_w = get_same_padding(slide_window_height,
												 stride_height,
												 image_height,
												 slide_window_width,
												 stride_width,
												 image_width)
			image_width += zeros_w
			image_height += zeros_h
			# Valid padding stride should fit exactly
			number_patches_height, number_patches_width = get_valid_padding(slide_window_height,
																		 stride_height,
																		 image_height,
																		 slide_window_width,
																		 stride_width,
																		 image_width)
			for i in range(number_patches_height):
				for j in range(number_patches_width):
					patches_coordinates.append([start_pixels_height,\
													start_pixels_width,\
													end_pixels_height,\
													end_pixels_width])
					# Update width with strides
					start_pixels_width += stride_width
					end_pixels_width += stride_width
				# Re-initialize the width parameters 
				start_pixels_width = 0
				end_pixels_width = slide_window_width
				# Update height with height stride size
				start_pixels_height += stride_height
				end_pixels_height += stride_height
			return patches_coordinates,\
					number_patches_height,\
					number_patches_width,\
					zeros_h,\
					zeros_w

		elif padding == "VALID_FIT_ALL":
			# Get number of patches
			patches_cols = number_patches[0]
			patches_rows = number_patches[1]
			# Determine the size of the windows for the patches
			stride_height = math.floor(image_height / patches_rows)
			slide_window_height = stride_height
			stride_width = math.floor(image_width / patches_cols)
			slide_window_width = stride_width
			#print("Size: ", strideHeigth, slideWindowHeight, strideWidth, slideWindowWidth)
			# Get valid padding
			start_pixels_height = 0
			end_pixels_height = slide_window_height
			start_pixels_width = 0
			end_pixels_width = slide_window_width
			patches_coordinates = []
			number_patches_height, number_patches_width = get_valid_padding(slide_window_height,
																		 stride_height,
																		 image_height,
																		 slide_window_width,
																		 stride_width,
																		 image_width)
			#print('numberPatchesHeight: ', numberPatchesHeight, 'numberPatchesWidth: ', numberPatchesWidth)
			for i in range(number_patches_height):
				for j in range(number_patches_width):
					patches_coordinates.append([start_pixels_height,\
													start_pixels_width,\
													end_pixels_height,\
													end_pixels_width])
					# Update width with strides
					start_pixels_width += stride_width
					end_pixels_width += stride_width
				# Re-initialize the width parameters
				start_pixels_width = 0
				end_pixels_width = stride_width
				# Update height with height stride size
				start_pixels_height += stride_height
				end_pixels_height += stride_height
			return patches_coordinates,\
					number_patches_height,\
					number_patches_width
		else:
			raise Exception("Type of padding not understood")

def get_valid_padding(slide_window_height,
					 stride_height,
					 image_height,
					 slide_window_width,
					 stride_width,
					 image_width):
	"""
	Given the dimensions of an image, the strides of the sliding window
	and the size of the sliding window. Find the number of patches that
	fit in the image if the type of padding is VALID.
	:param slide_window_height: int that represents the height of the slide
								Window
	:param stride_height: int that represents the height of the stride
	:param image_height: int that represents the height of the image
	:param slide_window_width: int that represents the width of the slide
								window
	:param stride_width: int that represents the width of the stride
	:param image_width: int that represents the width of the image
	: return: a tuple containing the number of patches in the height and 
			and the width dimension.
	"""
	number_patches_height_ = 0
	number_patches_width_ = 0
	while(True):
		if slide_window_height <= image_height:
			slide_window_height += stride_height
			number_patches_height_ += 1
		elif slide_window_height > image_height:
			break
		else:
			continue
	while(True):
		if slide_window_width <= image_width:
			slide_window_width += stride_width
			number_patches_width_ += 1	
		elif slide_window_width > image_width:
			break
		else:
			continue
	return (number_patches_height_, number_patches_width_)

def get_same_padding(slide_window_height,
					 stride_height,
					 image_height,
					 slide_window_width,
					 stride_width,
					 image_width):
	""" 
	Given the dimensions of an image, the strides of the sliding window
	and the size of the sliding window. Find the number of zeros needed
	for the image so the sliding window fits as type of padding SAME. 
	Then find the number of patches that fit in the image. 
	:param slide_window_height: int that represents the height of the slide 
								Window
	:param stride_height: int that represents the height of the stride
	:param image_height: int that represents the height of the image
	:param slide_window_width: int that represents the width of the slide
								window
	:param stride_width: int that represents the width of the stride
	:param image_width: int that represents the width of the image
	: return: a tuple containing the amount of zeros
				to add in the height dimension and the amount of zeros
				to add in the width dimension. 
	"""
	# Initialize auxiliar variables
	number_patches_height_ = 0
	number_patches_width_ = 0
	# Calculate the number of patches that fit
	while(True):
		if slide_window_height <= image_height:
			slide_window_height += stride_height
			number_patches_height_ += 1
		elif slide_window_height > image_height:
			break
		else:
			continue
	while(True):
		if slide_window_width <= image_width:
			slide_window_width += stride_width
			number_patches_width_ += 1	
		elif slide_window_width > image_width:
			break
		else:
			continue
	# Fix the excess in slide_window
	slide_window_height -= stride_height
	slide_window_width -= stride_width
	#print(number_patches_height_, number_patches_width_)
	#print(slide_window_height, slide_window_width)
	# Calculate how many pixels to add
	zeros_h = 0
	zeros_w = 0
	if slide_window_width == image_width:
		pass
	else:
		# Pixels left that do not fit in the kernel
		assert slide_window_width < image_width, "Slide window + stride is bigger than width"
		zeros_w = (slide_window_width + stride_width) - image_width
	if slide_window_height == image_height:
		pass
	else:
		# Pixels left that do not fit in the kernel 
		assert slide_window_height < image_height, "Slide window + stride is bigger than height"
		zeros_h = (slide_window_height + stride_height) - image_height
	#print(slideWindowHeight, imageHeight, resid_h, zeros_h)
	# Return amount of zeros
	return (zeros_h, zeros_w)

def lazySAMEpad(frame,
				zeros_h,
				zeros_w,
				padding_type = "ONE_SIDE"):
	"""
	Given an image and the number of zeros to be added in height 
	and width dimensions, this function fills the image with the 
	required zeros.
	:param frame: opencv image of 3 dimensions
	:param zeros_h: int that represents the amount of zeros to be added
					in the height dimension
	:param zeros_w: int that represents the amount of zeros to be added 
					in the width dimension
	: param padding_type: string that determines the side where to pad the image.
					If BOTH_SIDES, then padding is applied to both sides.
					If ONE_SIDE, then padding is applied to the right and the bottom.
					Default: ONE_SIDE
	: return: a new opencv image with the added zeros
	"""
	if padding_type == "BOTH_SIDES":
		rows, cols, d = frame.shape
		# If height is even or odd
		if (zeros_h % 2 == 0):
			zeros_h = int(zeros_h/2)
			frame = r_[np.zeros((zeros_h, cols, 3)), frame,\
						np.zeros((zeros_h, cols, 3))]
		else:
			zeros_h += 1
			zeros_h = int(zeros_h/2)
			frame = r_[np.zeros((zeros_h, cols, 3)), frame,\
						np.zeros((zeros_h, cols, 3))]

		rows, cols, d = frame.shape
		# If width is even or odd 
		if (zeros_w % 2 == 0):
			zeros_w = int(zeros_w/2)
			# Container 
			container = np.zeros((rows,(zeros_w*2+cols),3), np.uint8)
			container[:,zeros_w:container.shape[1]-zeros_w:,:] = frame
			frame = container #c_[np.zeros((rows, zeros_w)), frame, np.zeros((rows, zeros_w))]
		else:
			zeros_w += 1
			zeros_w = int(zeros_w/2)
			container = np.zeros((rows,(zeros_w*2+cols),3), np.uint8)
			container[:,zeros_w:container.shape[1]-zeros_w:,:] = frame
			frame = container #c_[np.zeros((rows, zeros_w, 3)), frame, np.zeros((rows, zeros_w, 3))]
		return frame
	elif padding_type == "ONE_SIDE":
		rows, cols, d = frame.shape
		# Pad height dimension
		frame = r_[frame, np.zeros((zeros_h, cols, 3))]
		# Pad width dimension
		rows, cols, d = frame.shape
		container = np.zeros((rows, cols + zeros_w, 3), np.uint8)
		container[:, :cols, :] = frame
		container[:, cols:, :] = np.zeros((rows, zeros_w, 3), np.uint8)
		return container

def drawGrid(frame,
			patches,
			patchesLabels):
	"""
	Draws the given patches on top of the input image
	:param frame: opencv input image
	:param patches: a list containing the coordinates of the patches
					calculated for the image
	: return: opencv image named frame that contains the same input
				image but with a grid of patches draw on top.
	"""
	# Iterate through patches
	for i in range(len(patches)):
		# Get patch
		patch = patches[i]
		# "Decode" patch
		startHeight, startWidth, endHeight, endWidth = patch[0], patch[1],\
														patch[2], patch[3]
		# Draw grids
		cv2.rectangle(frame, (startWidth, startHeight),\
						(endWidth, endHeight), (0, 0, 255), 12)
		roi = np.zeros([patch[2]-patch[0], patch[3]-patch[1], 3],\
						np.uint8)
		# Paint the patch
		if patchesLabels[i] == 1:
			roi[:,:,:] = (0,0,255)
		else:
			roi[:,:,:] = (0,255,0)
		cv2.addWeighted(frame[patch[0]:patch[2],patch[1]:patch[3],:],\
						0.8, roi, 0.2, 0, roi)
		frame[patch[0]:patch[2],patch[1]:patch[3],:] = roi
	return frame

def drawBoxes(frame,
			  patchesCoordinates,
			  patchesLabels):
	"""
	Draws a box or boxes over the frame
	:param frame: input cv2 image
	:param patchesCoordinates: a list containing sublists [iy, ix, y, x]
							  of coordinates
	:param patchesLabels: a list containing the labels of the coordinates
	"""
	for coord in patchesCoordinates:
		# Decode coordinate [iy, ix, y, x]
		iy, ix, y, x = coord[0], coord[1], coord[2], coord[3]
		# Draw box
		cv2.rectangle(frame, (ix, iy), (x, y), (255, 0, 0), 8)
	return frame
