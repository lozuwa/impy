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

	def invertColor(self, frame = None, CSpace = None):
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
		# Assertions.
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (CSpace == None):
			CSpace = [True, True, True]
		if ((type(CSpace) == tuple) or (type(CSpace) == list)):
			pass
		else:
			raise TypeError("ERROR: CSpace parameter has to be either a tuple or "+\
											"a list: {}".format(type(CSpace)))
		# Check CSpace.
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
		if (not (frame.dtype == np.uint8)):
			print("WARNING: Image is not dtype uint8. Forcing type.")
			frame = frame.astype(np.uint8)
		# Return tensor.
		return frame

	def histogramEqualization(self, frame = None, equalizationType = None):
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
		if (type(equalizationType) != int):
			raise TypeError("ERROR: equalizationType has to be of type int.")
		# Local variables
		equ = np.zeros(frame.shape, np.uint8)
		# Equalize hist
		if (equalizationType == 0):
			for channel in range(3):
				equ[:, :, channel] = cv2.equalizeHist(frame[:, :, channel])
		elif (equalizationType == 1):
			clahe = cv2.createCLAHE(clipLimit=2.0)
			for channel in range(3):
				equ[:, :, channel] = clahe.apply(frame[:, :, channel])
		else:
			raise ValueError("ERROR: equalizationType not understood.")
		if (not (equ.dtype == np.uint8)):
			print("WARNING: Image is not dtype uint8. Forcing type.")
			equ = equ.astype(np.uint8)
		return equ

	def changeBrightness(self, frame = None, coefficient = None):
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
		if (len(frame.shape) == 3):
			channels = 3
		elif (len(frame.shape) == 2):
			channels = 1
		else:
			raise Exception("ERROR: Frame has to be either 1 or 3 channels.")
		if (coefficient == None):
			coefficient = np.random.rand()*2
		if (type(coefficient) != float):
			raise TypeError("ERROR: Coefficient parameter has to be of type float.")
		# Change brightness
		if (channels == 3):
			for i in range(channels):
				frame[:, :, i] = cv2.multiply(frame[:, :, i], coefficient)
		elif (channels == 1):
			frame[:, :] = cv2.multiply(frame[:, :], coefficient)
		# Force cast in case of overflow
		if (not (frame.dtype == np.uint8)):
			print("WARNING: Image is not dtype uint8. Forcing type.")
			frame = frame.astype(np.uint8)
		return frame

	def sharpening(self, frame = None, weight = None):
		"""
		Sharpens an image using the following system:
		frame = I(x, y, d)
		gray_frame(xi, yi) = sum(I(xi, yi, d) * [0.6, 0.3, 0.1])
		hff_kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
		edges(x, y) = hff_kernel * gray_frame
		weight = 2.0
		sharpened(x, y, di) = (edges x weight) + frame(x, y, di)
		Args:
			frame: A tensor that contains an image.
			weight: A float that contains the weight coefficient.
		Returns:
			A sharpened tensor.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (len(frame.shape) == 3):
			channels = 3
		elif (len(frame.shape) == 2):
			channels = 1
		else:
			raise Exception("ERROR: Frame not understood.")
		if (weight == None):
			weight = 2.0
		if (type(weight) != float):
			raise TypeError("ERROR: Weight has to be a float.")
		# Local variables
		hff_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
		# Logic
		if (channels == 3):
			gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			edges = cv2.filter2D(gray_frame, -1, hff_kernel)
			edges = cv2.multiply(edges, weight)
			sharpened = np.zeros(frame.shape, np.uint8)
			for i in range(channels):
				sharpened[:, :, i] = cv2.add(frame[:, :, i], edges)
		else:
			edges = cv2.filter2D(frame, -1, hff_kernel)
			edges = cv2.multiply(edges, weight)
			sharpened[:, :] = cv2.add(frame[:, :], edges)
		if (not (sharpened.dtype == np.uint8)):
			print("WARNING: Image is not dtype uint8. Forcing type.")
			sharpened = sharpened.astype(np.uint8)
		return sharpened

	def addGaussianNoise(self, frame = None, coefficient = None):
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
		if (type(coefficient) != float):
			raise TypeError("ERROR: Coefficient parameter has to be of type float.")
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
		if (not (frame.dtype == np.uint8)):
			print("WARNING: Image is not dtype uint8. Forcing type.")
			frame = frame.astype(np.uint8)
		return frame

	def gaussianBlur(self, frame = None, kernelSize = None, sigma = None):
		"""
		Blur an image applying a gaussian filter with a random sigma(0, sigma_max)
		Sigma's default value is between 1 and 3.
		Args:
			frame: A tensor that contains an image.
			kernelSize: A list or tuple that contains the size of the kernel
									that will be convolved with the image.
			sigma: A float that contains the value of the gaussian filter.
		Returns:
			A tensor with a rotation of the original image.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (kernelSize == None):
			kernelSize = [5,5]
		if ((type(kernelSize) == list) or (type(kernelSize) == tuple)):
			pass
		else:
			raise TypeError("Kernel size must be of type list or tuple.")
		if (type(kernelSize) == list):
			kernelSize = tuple(kernelSize)
		if (len(kernelSize) != 2):
			raise ValueError("Kernel size must be of size 2.")
		if ((kernelSize[0] > 8) or (kernelSize[1] > 8)):
			raise ValueError("Kernel size is constrained to be of max size 8.")
		if (sigma == None):
			sigma = float(random.random()*3) + 1
		if (type(sigma) != float):
			raise TypeError("ERROR: Sigma parameter has to be of type float.")
		# Logic.
		blurredFrame = cv2.GaussianBlur(frame, kernelSize, sigma)
		if (not (blurredFrame.dtype == np.uint8)):
			print("WARNING: Image is not dtype uint8. Forcing type.")
			blurredFrame = blurredFrame.astype(np.uint8)
		return blurredFrame

	def averageBlur(self, frame = None, kernelSize = None):
		"""
		Convolves the image with an average filter.
		Args:
			frame: A tensor that contains an image.
			kernelSize: A tuple or list that contains the size 
									of the kernel that will be convolved with
									the image.
		Returns:
			A tensor with a blurred image.
		"""
		# Assertions.
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (kernelSize == None):
			kernelSize = [5,5]
		if ((type(kernelSize) == list) or (type(kernelSize) == tuple)):
			pass
		else:
			raise Exception("Kernel size must be a list or a tuple.")
		if (type(kernelSize) == list):
			kernelSize = tuple(kernelSize)
		if (len(kernelSize) != 2):
			raise ValueError("Kernel size must be a list or tuple of length 2.")
		if ((kernelSize[0] > 8) or (kernelSize[1] > 8)):
			raise ValueError("Kernel size is constrained to be of max size 8.")
		# Local variables.
		m = kernelSize[0]*kernelSize[1]
		averageKernel = (1 / m) * np.ones(kernelSize, np.float32)
		# Logic.
		blurredFrame = cv2.filter2D(frame, -1, averageKernel)
		if (not (blurredFrame.dtype == np.uint8)):
			print("WARNING: Image is not dtype uint8. Forcing type.")
			blurredFrame = blurredFrame.astype(np.uint8)
		# Return blurred image.
		return blurredFrame

	def medianBlur(self, frame = None, coefficient = None):
		"""
		Convolves an image with a median blur kernel.
		Args:
			frame: A tensor that contains an image.
			coefficient: An odd integer.
		Returns:
			A median blurred frame.
		"""
		# Assertions.
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (coefficient == None):
			coefficient = 5
		if (type(coefficient) != int):
			raise TypeError("Coefficient must be an integer.")
		if (coefficient%2==0):
			raise ValueError("Coefficient must be an odd number.")
		if (coefficient > 9):
			raise ValueError("Coefficient is constrained to be max 9.")
		# Logic.
		blurredFrame = cv2.medianBlur(frame, coefficient)
		# Return blurred tensor.
		return blurredFrame

	def bilateralBlur(self, frame = None, d = None, sigmaColor = None, sigmaSpace = None):
		"""
		Convolves an image with a bilateral filter.
		Args:
			d: Diameter of each pixel neighborhood.
			sigmaColor: Filter color space.
			sigmaSpace: Filter the coordinate space.
		Returns:
			An image blurred by a bilateral filter.
		"""
		# Assertions.
		if (d == None):
			d = 5
		if (type(d) != int):
			raise ValueError("d has to be of type int.")
		if (d > 9):
			raise ValueError("d is allowed to be maximum 9.")
		if (sigmaColor == None):
			sigmaColor = 75
		if (type(sigmaColor) != int):
			raise ValueError("sigmaColor has to be of type int.")
		if (sigmaColor > 250):
			raise ValueError("Sigma color is allowed to be maximum 250.")
		if (sigmaSpace == None):
			sigmaSpace = 75
		if (type(sigmaSpace) != int):
			raise ValueError("sigmaSpace has to be of type int.")
		if (sigmaSpace > 200):
			raise ValueError("Sigma space is allowed to be maximum 200.")		
		# Logic.
		blurredFrame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)
		# Return blurred frame.
		return blurredFrame

	def shiftColors(self, frame = None):
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
		if (not (frame.dtype == np.uint8)):
			print("WARNING: Image is not dtype uint8. Forcing type.")
			frame = frame.astype(np.uint8)
		return frame

	def fancyPCA(self, frame = None):
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
		if (not (frame.dtype == np.uint8)):
			print("WARNING: Image is not dtype uint8. Forcing type.")
			frame = frame.astype(np.uint8)		
		return framePCA
