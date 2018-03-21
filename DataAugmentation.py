"""
package: Images2Dataset
class: DataAugmentation
Author: Rodrigo Loza
Description: Common data augmentation operations 
for an image.
Log:
	Novemeber, 2017 -> Re-estructured class.
	December, 2017 -> Researched most used data augmentation techniques.
	March, 2018 -> Coded methods.
"""
# Libraries
from interface import implements
import math
import random
import cv2

from BoundingBoxDataAugmentationMethods import *

class DataAugmentation(implements(BoundingBoxDataAugmentationMethods)):
	"""
	DataAugmentation class. This class implements a set of data augmentation
	tools for bouding boxes and images.
	IMPORTANT
	- This class assumes input images are numpy tensors that follow the opencv
	color format BGR.
	"""
	def __init__(self):
		# Super class
		super(DataAugmentation, self).__init__()

	def centerCrop(self, frameHeight = None, frameWidth = None, bndbxCoordinates = None, offset = None):
		"""
		Crops the bounding box from the frame giving it a symetric 
		offset at each side so it is maintained in the center.
		Args:
			frameHeight: An int that represents the height of the frame.
			frameWidth: An int that representst he width of the frame.
			bndbxCoordinates: A tuple of ints that contains the coordinates of the
												bounding box in the frame.
			offset: An int that contains the amount of space to give at each side 
							of the bounding box.
		Returns:
			An 8-sized tuple that contains the coordinates to crop the original frame
			and the new coordinates of the bounding box inside the cropped patch.
		Example:
			Given an image and a bounding box, crop the bndbox with an offset to 
			maintain context of the object.
				---------------
				|             |
				|     ---     |
				|     | |     |
				|     ---     |
				---------------
			So,
				---------------
				|   --------  |
				|   | ---   | |
				|   | | |   | |
				|   | ---   | |
				---------------
		"""
		pass

	def cropWithTranslation(self, frameHeight = None, frameWidth = None, bndbxCoordinates = None, offset = None):
		"""
		Crops the bounding box from the frame giving it an asymetric 
		offset at each side so translation is simulated inside the context
		of the cropped patch.
		Args:
			frameHeight: An int that represents the height of the frame.
			frameWidth: An int that representst he width of the frame.
			bndbxCoordinates: A tuple of ints that contains the coordinates of the
												bounding box in the frame.
			offset: An int that contains the amount of space that will be used to limit
							the size of the cropped patch.
		Returns:
			An 8-sized tuple that contains the coordinates to crop the original frame
			and the new coordinates of the bounding box inside the cropped patch.
		Example:
			Given an image and a bounding box, crop the bndbox with an offset to 
			maintain context of the object.
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
		pass

	def jitterBoxes(self, frame = None, quantity = None):
		"""
		Creates random jitter boxes inside a bounding box cropped from its image.
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
			y = int(random.random() * rows) - rows // 3
			x = int(random.random() * cols) - cols // 3
			# Draw boxes on top of the image
			frame = cv2.rectangle(frame, (x, y), (x+xj, y+yj), (0, 0, 0), -1)
		# Return frame
		return frame

	def horizontalFlip(self, frame = None):
		"""
		Flip a bouding box by its horizontal axis.
		Args:
			frame: A tensor that contains a cropped bouding box from its frame.
		Returns:
			A tensor that has been flipped by its horizontal axis.
		"""
		# Flip
		frame = cv2.flip(frame, 0)
		return frame

	def verticalFlip(self, frame = None):
		"""
		Flip a bouding box by its vertical axis.
		Args:
			frame: A tensor that contains a cropped bouding box from its frame.
		Returns:
			A tensor that has been flipped by its vertical axis.
		"""
		# Flip
		frame = cv2.flip(frame, 1)
		return frame

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
			theta = random.random() * math.pi
		# Local variables
		thetaDegrees = theta * 180 / math.pi
		rows, cols, depth = frame.shape
		# Decode the bouding box
		ix, iy, x, y = bndbox
		# Fix the y coordinate since matrix transformations
		# assume 0,0 is at the left bottom corner.
		iy, y = rows-iy, rows-y
		# print(ix, iy, x, y)
		# Center the coordinates with respect to the 
		# center of the image.
		ix, iy, x, y = ix-(cols//2), iy-(rows//2), x-(cols//2), y-(rows//2)
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
		print(thetaDegrees)
		xs = [p0[0], p1[0], p2[0], p3[0]]
		ys = [p0[1], p1[1], p2[1], p3[1]]
		ix, x = min(xs), max(xs)
		iy, y = min(ys), max(ys)
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
		if frame == None:
			sigma = int(random.random()*3) + 1
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
		if frame.shape != 3:
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


