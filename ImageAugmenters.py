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

	Input to all methods: Given an image with its bounding boxes.
	---------------
	Space dimension
	---------------
	1. Scaling
		Resize the image to (h' x w') and maintain the bounding boxes' sizes.
	2. Crop
		Crop the bounding boxes using random coordinates.
	3. Pad (also translation)
		Include exterior pixels to bounding boxes.
	4. Flip horizontally
		Flip the bounding boxes horizontally.
	5. Flip vertically
		Flip the bounding boxes vertically.
	6. Rotation
		Randomly rotates the bounding boxes.
	7. Jitter boxes
		Draws random color boxes inside the bounding boxes. 

	---------------
	Color dimension
	---------------
	1. Change color space
		Change the color space of the image.
	2. Gaussian blur 
		Convolves the image with a Gaussian filter.
	3. Average blur
		Convolves the image with an average blur filter.
	4. Median blur
		Convolves the image with a median filter.
	5. Sharpen
		Convolves the image with a sharpening filter.
	6. Add gaussian noise
		Adds the image with a gaussian tensor of the same size. The 
		tensor is produced with a normal distribution. 
	7. Contrast
		Multiplies the image pixels with a value C that produces a 
		ligher or darker image.
	
"""
# Libraries
from interface import implements
import math
import random
import cv2
import numpy as np

try:
	from .BoundingBoxAugmentersMethods import *
except:
	from BoundingBoxAugmentersMethods import *

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

	def scale(self,
						frame = None,
						boundingBoxes = None,
						resizeSize = None,
						interpolationMethod = None):
		"""
		Scales an image with its bounding boxes to another size while maintaing the 
		coordinates of the bounding boxes.
		Args:
			frame: A tensor that contains an image.
			boundingBoxes: A list of lists that contains the coordinates of the bounding
											boxes that are part of the image.
			resizeSize: A tuple that contains the resizing values.
			interpolationMethod: Set the type of interpolation method. 
														(INTER_NEAREST -> 0,
														INTER_LINEAR -> 1, 
														INTER_CUBIC -> 2, 
														INTER_LANCZOS4 -> 4)
		Returns:
			An image that has been scaled and a list of lists that contains the new 
			coordinates of the bounding boxes.
		"""
		# Local variable assertions
		if (resizeSize == None):
			raise ValueError("resizeSize cannot be empty.")
		elif (type(resizeSize) != tuple):
			raise ValueError("resizeSize has to be a tuple (width, height)")
		else:
			if (len(resizeSize) != 2):
				raise ValueError("resizeSize must be a tuple of size 2 (width, height)")
			else:
				resizeWidth, resizeHeight = resizeSize
				if (resizeWidth == 0 or resizeHeight == 0):
					raise ValueError("Neither width nor height can be 0.")
		if (interpolationMethod == None):
			interpolationMethod = 2
		# Local variables
		height, width, depth = frame.shape
		reduX = height / resizeHeight
		reduY = width / resizeWidth
		# Scale image
		frame = cv2.resize(frame.copy(), resizeSize, interpolationMethod)
		# Fix bounding boxes
		for boundingBox in boundingBoxes:
			# Decode bounding box
			ix, iy, x, y = boundingBox
			# Update its values with the resizing factor
			ix, iy, x, y = ix // reduX, iy // reduY, x // reduX, y // reduY
			# Check variables are not the same as the right and bottom boundaries
			ix, iy, x, y = Data
		# Return values
		return frame, boundingBoxes

	def translate(self,
								frameHeight = None,
								frameWidth = None,
								boundingBoxes = None,
								offset = None):
		"""
		Given an image and its bounding boxes, this method translates the bounding boxes
		to create an alteration of the image.
		Args:
			frameHeight: An int that represents the height of the frame.
			frameWidth: An int that representst he width of the frame.
			boundingBoxes: A tuple of ints that contains the coordinates of the
												bounding box in the frame.
			offset: An int that contains the amount of space that will be used to limit
							the size of the cropped patch.
		Returns:
			An 8-sized tuple that contains the coordinates to crop the original frame
			and the new coordinates of the bounding box inside the cropped patch.
		Example:
			Given an image and its bounding boxes, translate the bounding boxes in the 
			allowed space the image .
				-------------------
				|                 |
				|     ---         |
				|     | |         |
				|     ---         |
				|                 |
				|                 |
				|                 |           
				-------------------
			So,
				-------------------
				|   ---------     |
				|   | ---   |     |
				|   | | |   |     |
				|   | ---   |     |
				|   |       |     |
				|   ---------     |
				|                 |           
				-------------------
		"""

	def jitterBoxes(self, frame = None, quantity = None):
		"""
		Draws random jitter boxes in the bounding boxes.
		Args:
			frame: A tensor that contains an image.
			quantity: An int that tells how many jitter boxes to create inside 
							the frame.
		Returns:
			A tensor that contains an image altered by jitter boxes.
		"""
		# Assertions
		try:
			if frame == None:
				raise Exception("Frame cannot be empty.")
		except:
			pass
		if quantity == None:
			quantity = 10
		# Local variables
		frameSize = frame.shape
		rows, cols = frameSize[0], frameSize[1]
		xj, yj = frameSize[0] // 8, frameSize[0] // 8
		# Create boxes
		for i in range(quantity):
			y = int(random.random() * rows) - (rows // 3)
			x = int(random.random() * cols) - (cols // 3)
			# Draw boxes on top of the image
			frame = cv2.rectangle(frame, (x, y), (x+xj, y+yj), (0, 0, 0), -1)
		# Return frame
		return frame

	def horizontalFlip(self, frame = None, boundingBoxes = None):
		"""
		Flip a bouding box by its horizontal axis.
		Args:
			frame: A tensor that contains an image with its bounding boxes.
		Returns:
			A tensor that has been flipped by its horizontal axis and a list
			of lists that contains the coordinates of the bounding boxes. 
		"""
				# Assertions
		try:
			if (frame == None):
				raise Exception("Frame parameter cannot be empty.")
		except:
			pass
		if (boundingBoxes == None):
			raise Exception("Bounding boxes parameter cannot be empty.")
		# Flip
		frame = cv2.flip(frame, 1)
		height, width, depth = frame.shape
		# Flip bounding boxes.
		for bndBox in boundingBoxes:
			aux = bndBox[0]
			bndBox[0] = width - bndBox[2]
			bndBox[2] = bndBox[0] + (bndBox[2] - aux)
		return frame, boundingBoxes

	def verticalFlip(self, frame = None, boundingBoxes = None):
		"""
		Flip a bouding box by its vertical axis.
		Args:
			frame: A tensor that contains a cropped bouding box from its frame.
		Returns:
			A tensor that has been flipped by its vertical axis and a list of lists 
			that contains the coordinates of the bounding boxes.
		"""
		# Assertions
		try:
			if (frame == None):
				raise Exception("Frame parameter cannot be empty.")
		except:
			pass
		if (boundingBoxes == None):
			raise Exception("Bounding boxes parameter cannot be empty.")
		# Flip frame with opencv.
		frame = cv2.flip(frame, 0)
		height, width, depth = frame.shape
		# Flip bounding boxes.
		for bndBox in boundingBoxes:
			aux = bndBox[1]
			bndBox[1] = height - bndBox[3]
			bndBox[3] = bndBox[1] + (bndBox[3] - aux)
		return frame, boundingBoxes

	def randomRotation(self, frame = None, bndbox = None, theta = None):
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
		try:
			if frame == None:
				raise Exception("Frame cannot be emtpy.")
		except:
			pass
		if bndbox == None:
			raise Exception("Bnbdbox cannot be empty")
		if theta == None:
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
		p0[0], p0[1] = DataAugmentation.rotation_equations(p0[0], p0[1], theta)
		p1[0], p1[1] = DataAugmentation.rotation_equations(p1[0], p1[1], theta)
		p2[0], p2[1] = DataAugmentation.rotation_equations(p2[0], p2[1], theta)
		p3[0], p3[1] = DataAugmentation.rotation_equations(p3[0], p3[1], theta)
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

	@staticmethod
	def rotation_equations(x, y, theta):
		"""
		Apply a 2D rotation matrix to a 2D coordinate by theta degrees.
		Args:
			x: An int that represents the x dimension of the coordinate.
			y: An int that represents the y dimension of the coordinate.
		"""
		x_result = int((x*math.cos(theta)) - (y*math.sin(theta)))
		y_result = int((x*math.sin(theta)) + (y*math.cos(theta)))
		return x_result, y_result

	def addRandomBlur(self, frame = None, sigma = None):
		"""
		Blur an image applying a gaussian filter with a random sigma(0, sigma_max)
		Sigma's default value is between 1 and 3.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor with a rotation of the original image.
		"""
		# Assertions
		try:
			if frame == None:
				raise Exception("Frame cannot be emtpy.")
		except:
			pass
		if sigma == None:
			sigma = float(random.random()*3) + 1
		# Apply gaussian filter
		frame = cv2.GaussianBlur(frame, (5, 5), sigma)
		return frame

	def shiftColors(self, frame = None):
		"""
		Shifts the colors of the frame.
		Args:
			frame: A tensor that contains an image.
		Returns:
			A tensor that has shifted the order of its colors.
		"""
		# Assertions
		try:
			if frame == None:
				raise Exception("Frame cannot be emtpy.")
		except:
			pass
		if len(frame.shape) != 3:
			raise Exception("Frame must have 3 dimensions")
		# Local variables
		colorsOriginal = [0, 1, 2]
		colorsShuffle = colorsOriginal
		# Shuffle list of colors
		while(colorsOriginal == colorsShuffle):
			shuffle(colorsShuffle)
		# Swap color dimensions
		frame[:, :, 0], frame[:, :, 1], \
		frame[:, :, 2] = frame[:, :, colorsShuffle[0]], \
										frame[:, :, colorsShuffle[1]], \
										frame[:, :, colorsShuffle[2]]
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
		try:
			if frame == None:
				raise Exception("Frame cannot be empty.")
		except:
			pass
		if len(frame.shape) != 3:
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
		frame = frame + perturb
		return frame
