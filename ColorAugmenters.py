"""
package: Images2Dataset
class: DataAugmentation
Email: lozuwaucb@gmail.com
Author: Rodrigo Loza
Description: Common data augmentation operations 
for an image.
Log:
	Novemeber, 2017 -> Re-estructured class.
	December, 2017 -> Researched most used data augmentation techniques.
	March, 2018 -> Coded methods.
	April, 2018 -> Redesigned the methods to support multiple bounding
								 boxes (traditional data augmentation tools.)
	April, 2018 -> Redefined list of augmenters:
	April, 2018 -> Separated generic image augmenters into color and geometric augmenters.

	---------------
	Color dimension
	---------------
	1. Invert color
		Invert the color space of the image.
	2. Histogram equalization
		Equalize the contrast of an image.
	3. Change brightness
		Change the brightness of an image.
	4. Random sharpening
		Randomly add sharpening to an image.
	5. Add gaussian noise
		Add normal noise to an image.
	6. Gaussian blur 
		Convolves the image with a Gaussian filter.
	7. Shift colors
		Swap the color spaces of an image.
	8. Fancy PCA
		Add a color perturbation based on the computation
		of color's space PCA.
"""
# Libraries
from interface import implements
import math
import random
import cv2
import numpy as np

try:
	from .ColorAugmentersMethods import *
except:
	from ColorAugmentersMethods import *

try:
	from .VectorOperations import *
except:
	from VectorOperations import *

try:
	from .AssertDataTypes import *
except:
	from AssertDataTypes import *

class ColorAugmenters(implements(ColorAugmentersMethods)):
	"""
	ImageAugmenters class. This class implements a set of data augmentation
	tools for bouding boxes.
	IMPORTANT
	- This class assumes input images are numpy tensors that follow the opencv
	color format BGR.
	"""
	def __init__(self):
		super(ColorAugmenters, self).__init__()
		self.assertion = AssertDataTypes()

	def invertColor(self,
									frame = None,
									CSpace = None):
		"""
		Inverts the color of an image.
		Args:
			frame: A tensor that contains an image.
			CSpace: A 3-sized tuple that contains booleans (B, G, R).
							If a boolean is set to true, then we invert that channel.
							If the 3 booleans are false, then we invert all the image.
		Returns:
			A tensor that has its color inverted.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (CSpace == None):
			CSpace = [True, True, True]
		# Check CSpace
		if (CSpace[0] == True):
			frame[:, :, 0] = cv2.bitwise_not(frame[:, :, 0])
		else:
			pass
		if (CSpace[1] == True):
			frame[:, :, 1] = cv2.bitwise_not(frame[:, :, 1])
		else:
			pass
		if (CSpace[2] == True):
			frame[:, :, 2] = cv2.bitwise_not(frame[:, :, 2])
		else:
			pass
		# Return tensor
		return frame

	def histogramEqualization(self,
														frame = None,
														equalizationType = None):
		"""
		Args:
			frame: A tensor that contains an image.
			equalizationType: An int that defines what type of histogram
						equalization algorithm to use.
		Returns:
			A frame whose channels have been equalized.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (len(frame.shape) != 3):
			raise ValueError("Frame needs to have at least 3 channels.")
		if (equalizationType == None):
			equalizationType = 0
		# Local variables
		equ = frame.copy()
		equChannel = []
		# Equalize hist
		if (equalizationType == 0):
			for channel in range(3):
				equChannel.append(cv2.equalizeHist(frame[:, :, channel]))
			equ[:, :, 0] = equChannel[0]
			equ[:, :, 1] = equChannel[1]
			equ[:, :, 2] = equChannel[2]
		elif (equalizationType == 1):
			for channel in range(3):
				clahe = cv2.createCLAHE(clipLimit=2.0)
				equ = clahe.apply(frame[:, :, channel])
				equChannel.append(equ)
			equ[:, :, 0] = equChannel[0]
			equ[:, :, 1] = equChannel[1]
			equ[:, :, 2] = equChannel[2]
		else:
			raise ValueError("equalizationType not understood.")
		return equ

	def changeBrightness(self,
											frame = None,
											coefficient = None):
		"""
		Change the brightness of a frame.
		Args:
			frame: A tensor that contains an image.
			coefficient: A float that changes the brightness of the image.
									Default is a random number in the range of 2.
		Returns:
			A tensor with its brightness property changed.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (coefficient == None):
			coefficient = np.random.rand()*0.75
		# Change brightness
		frame = frame*coefficient
		return frame

	def sharpening(self,
								frame = None,
								weight = None):
		"""
		Sharpens an image.
		Args:
			frame: A tensor that contains an image.
			weight: A float that contains the weight coefficient.
		Returns:
			A sharpened tensor.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (weight == None):
			prob = np.random.rand()
			comp = 1 - prob
		else:
			if (type(weight) == float):
				prob = weight
				comp = 1 - prob
		# Find edges
		edges = cv2.Laplacian(frame, cv2.CV_64F)
		edges = edges.astype(np.uint8)
		frame = frame.astype(np.uint8)
		# Add edges to original frame
		# print("DEBUG: ", edges.shape, frame.shape)
		# print("DEBUG: ", edges.dtype, frame.dtype)
		frameSharpened = cv2.addWeighted(edges, prob, frame, comp, 0)
		return frameSharpened

	def addGaussianNoise(self,
											frame = None,
											coefficient = None):
		"""
		Add gaussian noise to a tensor.
		Args:
			frame: A tensor that contains an image.
			coefficient: A float that contains the amount of noise to add
										to a frame.
		Returns:
			An altered frame that has gaussian noise.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (coefficient == None):
			coefficient = 0.2
		# Local variables
		height, width, depth = frame.shape
		# Create random noise
		gaussianNoise = np.random.rand(height*width*depth) * 255
		# Reshape noise
		gaussianNoise = np.array([int(i) for i in gaussianNoise], np.uint8)
		gaussianNoise = gaussianNoise.reshape([height, width, depth])
		# Cast types
		gaussianNoise = gaussianNoise.astype(np.uint8)
		frame = frame.astype(np.uint8)
		# Add noise to frame
		frame = cv2.addWeighted(frame, 1-coefficient, gaussianNoise, coefficient, 0)
		return frame

	def gaussianBlur(self,
										frame = None,
										sigma = None):
		"""
		Blur an image applying a gaussian filter with a random sigma(0, sigma_max)
		Sigma's default value is between 1 and 3.
		Args:
			frame: A tensor that contains an image.
			sigma: A float that contains the value of the gaussian filter.
		Returns:
			A tensor with a rotation of the original image.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if sigma == None:
			sigma = float(random.random()*3) + 1
		# Apply gaussian filter
		frame = cv2.GaussianBlur(frame, (5, 5), sigma)
		return frame

	def shiftColors(self,
									frame = None):
		"""
		Shifts the colors of the frame.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that has shifted the order of its colors.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (len(frame.shape) != 3):
			raise Exception("Frame must have 3 dimensions")
		# Local variables
		colorsOriginal = [0, 1, 2]
		colorsShuffle = [0, 1, 2]
		# Shuffle list of colors
		while(colorsOriginal == colorsShuffle):
			np.random.shuffle(colorsShuffle)
		# Swap color dimensions
		frame[:, :, 0], frame[:, :, 1], \
		frame[:, :, 2] = frame[:, :, colorsShuffle[0]], \
										frame[:, :, colorsShuffle[1]], \
										frame[:, :, colorsShuffle[2]]
		return frame

	def fancyPCA(self,
							frame = None):
		"""
		Fancy PCA implementation.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that contains the altered image by fancy PCA.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (len(frame.shape) != 3):
			raise Exception("Frame must have 3 dimensions")
		# Local variables
		height, width, depth = frame.shape
		# Create matrix
		redCol = frame[:, :, 2].reshape(-1, 1)
		greenCol = frame[:, :, 1].reshape(-1, 1)
		blueCol = frame[:, :, 0].reshape(-1, 1)
		# Calculate the mean of every column and substract it
		uRed = np.mean(redCol)
		uGreen = np.mean(greenCol)
		uBlue = np.mean(blueCol)
		redCol = redCol - uRed
		greenCol = greenCol - uGreen
		blueCol = blueCol - uBlue
		# Define matrix
		matrix = np.zeros([redCol.shape[0], 3])
		matrix[:, 0] = list(redCol)
		matrix[:, 1] = list(greenCol)
		matrix[:, 2] = list(blueCol)
		# Normalize data
		# If the data is in the range 0-1, then normalize the image.
		# If the data is in the range 0-255, then don't normalize. 
		# Apply PCA
		cov = np.cov(matrix.T)
		eigvals, eigvects = np.linalg.eigh(cov)
		# pca = np.dot(np.sqrt(eigvals), eigvects)
		pca = np.sqrt(eigvals) * eigvects
		# print(frame.shape, cov.shape)
		# print(eigvects.shape, eigvals.shape)
		perturb = [int(each) for each in (pca*np.random.randn(3)*0.1).sum(axis=1)]
		# print("Perturbation: ", perturb)
		# Add perturbation vector to frame
		framePCA = frame.copy() + perturb
		return framePCA
