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

	---------------
	Space dimension
	---------------
	1. Scaling
		Resize the image to (h' x w').
	2. Translate
		Translate an image.
	3. Jitter boxes
		Draws random color boxes inside the image. 
	4. Flip horizontally
		Flip the image horizontally.
	5. Flip vertically
		Flip the image vertically.
	6. Rotation
		Randomly rotates the bounding boxes.

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
	from .ImageAugmentersMethods import *
except:
	from ImageAugmentersMethods import *

try:
	from .VectorOperations import *
except:
	from VectorOperations import *

try:
	from .AssertDataTypes import *
except:
	from AssertDataTypes import *

class ImageAugmenters(implements(ImageAugmentersMethods)):
	"""
	ImageAugmenters class. This class implements a set of data augmentation
	tools for bouding boxes.
	IMPORTANT
	- This class assumes input images are numpy tensors that follow the opencv
	color format BGR.
	"""
	def __init__(self):
		super(ImageAugmenters, self).__init__()
		self.assertion = AssertDataTypes()

	def scale(self,
						frame = None,
						size = None,
						interpolationMethod = None):
		"""
		Scales an image to another size.
		Args:
			frame: A tensor that contains an image.
			boundingBoxes: A list of lists that contains the coordinates of the bounding
											boxes that are part of the image.
			size: A tuple that contains the resizing values.
			interpolationMethod: Set the type of interpolation method. 
														(INTER_NEAREST -> 0,
														INTER_LINEAR -> 1, 
														INTER_CUBIC -> 2, 
														INTER_LANCZOS4 -> 4)
		Returns:
			An image that has been scaled.
		"""
		# Local variable assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (size == None):
			raise ValueError("size cannot be empty.")
		elif (type(size) != tuple):
			raise ValueError("size has to be a tuple (width, height)")
		else:
			if (len(size) != 2):
				raise ValueError("size must be a tuple of size 2 (width, height)")
			else:
				resizeWidth, resizeHeight = size
				if (resizeWidth == 0 or resizeHeight == 0):
					raise ValueError("Neither width nor height can be 0.")
		if (interpolationMethod == None):
			interpolationMethod = 2
		# Local variables
		height, width, depth = frame.shape
		reduX = height / resizeHeight
		reduY = width / resizeWidth
		# Scale image
		frame = cv2.resize(frame.copy(), size, interpolationMethod)
		# Return values
		return frame

	def translate(self,
								frame = None,
								offset = None):
		"""
		Given an image and its bounding boxes, this method translates the bounding boxes
		to create an alteration of the image.
		Args:
			frame: A tensor that contains an image.
			offset: A tuple that contains the amoung of space to move on each axis.
							(widthXheight)
		Returns:
			A translated tensor by offset.
		"""
		# Local variables
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (len(frame.shape) == 3):
			height, width, depth = frame.shape
		elif (len(frame.shape) == 2):
			height, width = frame.shape
		else:
			raise ValueError("Type of data not understood.")
		if (offset == None):
			raise ValueError("Offset cannot be empty.")
		if (len(offset) == 2):
			tx, ty = offset
		elif (len(offset) == 1):
			tx, ty = offset, offset
		else:
			raise ValueError("offset is not understood.")
		# Translate image
		M = np.float32([[1, 0, 100], [0, 1, 50]])
		frame = cv2.warpAffine(frame, M, (width, height))
		return frame

	def jitterBoxes(self,
									frame = None,
									size = None,
									quantity = None,
									color = None):
		"""
		Draws random jitter boxes in the bounding boxes.
		Args:
			frame: A tensor that contains an image.
			size: A tuple that contains the size of the jitter boxes to draw.
			quantity: An int that tells how many jitter boxes to create inside 
							the frame.
			color: A 3-sized tuple that contains the RGB code for a color. Default
							is black (0,0,0)
		Returns:
			A tensor that contains an image altered by jitter boxes.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (quantity == None):
			quantity = 10
		if (size == None):
			raise Exception("Size cannot be empty.")
		# Local variables
		rows, cols, depth = frame.shape
		# Create boxes
		for i in range(quantity):
			y = int(random.random() * rows) - (rows // 3)
			x = int(random.random() * cols) - (cols // 3)
			# Draw boxes on top of the image
			frame = cv2.rectangle(frame, (x, y), (x+size[0], y+size[1]), color, -1)
		# Return frame
		return frame

	def horizontalFlip(self,
										frame = None):
		"""
		Flip a frame by its horizontal axis.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that has been flipped by its horizontal axis.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		# Flip
		frame = cv2.flip(frame, 1)
		return frame

	def verticalFlip(self,
									frame = None):
		"""
		Flip a bouding box by its vertical axis.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that has been flipped by its vertical axis.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		# Flip frame with opencv.
		frame = cv2.flip(frame, 0)
		return frame

	def rotation(self,
							frame = None,
							bndbox = None,
							theta = None):
		"""
		Rotate a frame clockwise by random degrees. Random degrees
		is a number that is between 20-360.
		Args:
			frame: A tensor that contains an image.
			bndbox: A tuple that contains the ix, iy, x, y coordinates 
							of the bounding box in the image.
			theta: An int that contains the amount of degrees to move.
							Default is random.
		Returns:
			A tensor that contains the rotated image and a tuple
			that contains the rotated coordinates of the bounding box.
		"""
		# Assertions
		if (self.assertion.assertNumpyType(frame) == False):
			raise ValueError("Frame has to be a numpy array.")
		if (bndbox == None):
			raise Exception("Bnbdbox cannot be empty")
		if (theta == None):
			theta = (random.random() * math.pi) + math.pi / 3
		# Local variables
		thetaDegrees = theta * 180 / math.pi
		rows, cols, depth = frame.shape
		# print("Degrees: ", thetaDegrees)
		# print("Rows, cols: ", rows//2, cols//2)
		# Decode the bouding box
		ix, iy, x, y = bndbox
		# print("Original: ", bndbox)
		# Fix the y coordinate since matrix transformations
		# assume 0,0 is at the left bottom corner.
		iy, y = rows-iy, rows-y
		# print(ix, iy, x, y)
		# Center the coordinates with respect to the 
		# center of the image.
		ix, iy, x, y = ix-(cols//2), iy-(rows//2), x-(cols//2), y-(rows//2)
		# print("Centered: ", ix, iy, x, y)
		# print(ix, iy, x, y)
		# Write down coordinates
		p0 = [ix, iy]
		p1 = [x, iy]
		p2 = [ix, y]
		p3 = [x, y]
		# Compute rotations on coordinates
		p0[0], p0[1] = VectorOperations.rotation_equations(p0[0], p0[1], theta)
		p1[0], p1[1] = VectorOperations.rotation_equations(p1[0], p1[1], theta)
		p2[0], p2[1] = VectorOperations.rotation_equations(p2[0], p2[1], theta)
		p3[0], p3[1] = VectorOperations.rotation_equations(p3[0], p3[1], theta)
		# Add centers to compensate
		p0[0], p0[1] = p0[0] + (cols//2), rows - (p0[1] + (rows//2))
		p1[0], p1[1] = p1[0] + (cols//2), rows - (p1[1] + (rows//2))
		p2[0], p2[1] = p2[0] + (cols//2), rows - (p2[1] + (rows//2))
		p3[0], p3[1] = p3[0] + (cols//2), rows - (p3[1] + (rows//2))
		# Rotate image
		M = cv2.getRotationMatrix2D((cols/2, rows/2), thetaDegrees, 1)
		frame = cv2.warpAffine(frame, M, (cols, rows))
		xs = [p0[0], p1[0], p2[0], p3[0]]
		ys = [p0[1], p1[1], p2[1], p3[1]]
		ix, x = min(xs), max(xs)
		iy, y = min(ys), max(ys)
		# print(p0, p1, p2, p3)
		# Make sure ix, iy, x, y are valid
		if ix < 0:
			# If ix is smaller, then it was impossible to place 
			# the coordinate inside the image because of the angle. 
			# In this case, the safest option is to set ix to 0.
			print("WARNING: ix is negative.", ix)
			ix = 0
		if iy < 0:
			# If iy is smaller, then it was impossible to place 
			# the coordinate inside the image because of the angle. 
			# In this case, the safest option is to set iy to 0.
			print("WARNING: iy is negative.", iy)
			iy = 0
		if x >= cols:
			print("WARNING: x was the width of the frame.", x, cols)
			x = cols - 1
		if y >= rows:
			print("WARNING: y was the height of the frame.", y, rows)
			y = rows - 1
		# Return frame and coordinates
		return frame, [ix, iy, x, y]

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
		edges = np.array(edges, np.uint8)
		# Add edges to original frame
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
		pca = np.sqrt(eigvals)*eigvects
		perturb = [int(each) for each in (pca*np.random.randn(3)*0.1).sum(axis=1)]
		# print("Perturbation: ", perturb)
		# Add perturbation vector to frame
		framePCA = frame.copy() + perturb
		return framePCA
